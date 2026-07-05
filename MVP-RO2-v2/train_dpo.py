# ---------------------------------------------------------------------------
# This file is adapted from the codebase accompanying the paper:
#   "Truthful or Fabricated? Using Causal Attribution to Mitigate
#    Reward Hacking in Explanations"
#   Paper:  https://openreview.net/pdf?id=nkdPLuKoL5
#   Source: https://github.com/PedroMLF/Reward-Hacking-in-Explanations
#
# Substantial changes have been made (in some cases) to suit our project's
# needs and requirements. Notably: the dataset schema is the standard TRL
# preference schema (prompt / chosen / rejected — no separate `query`), the
# policy model is Qwen3, and Hugging Face Hub pushing is optional.
# ---------------------------------------------------------------------------
"""LoRA DPO training on the harmlessness preference pairs.

Consumes the DatasetDict exported by DPO-train-dataset-curation/ (stage 06):
    DatasetDict({"train","valid"}) with columns prompt / chosen / rejected.

Run via train.sh (one command), or directly:
    python train_dpo.py --dataset_path <.../dpo_dataset> --model_name_or_path Qwen/Qwen3-4B ...
"""
import gzip
import json
import os
import shutil
from dataclasses import dataclass
from glob import glob

import random

import datasets
import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    enable_full_determinism,
    set_seed as hf_set_seed,
)
from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    TrlParser,
    get_peft_config,
)


# ── Self-contained helpers (this file has no dependency on any sibling module) ──
def set_seed(seed, full_determinism=False):
    """Seed python / numpy / torch / transformers for reproducibility."""
    print(f"Setting seed to {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    hf_set_seed(seed)
    if full_determinism:
        enable_full_determinism(seed)


def setup_hf_env():
    """Load a local .env (if present), point HF caches at HF_MODELS_CACHE_DIR when set, and
    log in when HF_TOKEN is available (gated models / Hub pushes). Never hard-code a token."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    cache = os.environ.get("HF_MODELS_CACHE_DIR")
    if cache:
        os.environ.setdefault("HF_HOME", cache)
        os.environ.setdefault("HF_DATASETS_CACHE", cache)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    token = os.environ.get("HF_TOKEN")
    if token:
        try:
            from huggingface_hub import login
            login(token=token)
        except Exception as e:  # never let an auth hiccup abort a local training run
            print(f"[hub] login skipped: {e}")


setup_hf_env()


def push_checkpoint_to_hub(args, ckpt_dir, step):
    """Push a single checkpoint dir to its own HF repo: <hub_model_id>-step-<N>."""
    from huggingface_hub import HfApi
    repo_id = f"{args.hub_model_id}-step-{step}"
    print(f"[push] Uploading step-{step} checkpoint to {repo_id} ...")
    api = HfApi()
    api.create_repo(repo_id, private=True, exist_ok=True, repo_type="model")
    api.upload_folder(folder_path=ckpt_dir, repo_id=repo_id, repo_type="model")
    print(f"[push] Done: https://huggingface.co/{repo_id}")


class PushStepCheckpointCallback(TrainerCallback):
    """Push each saved checkpoint to the Hub as soon as the trainer writes it.

    Only added to the trainer when --push_to_hub and --hub_model_id are set. The step-0
    baseline checkpoint isn't saved through the normal on_save path (it's written directly
    by StepwiseEvalCallback before training starts), so that one is pushed separately.
    """

    def on_save(self, args, state, control, **kwargs):
        step = state.global_step
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
        if not os.path.isdir(ckpt_dir):
            return
        push_checkpoint_to_hub(args, ckpt_dir, step)


class BestCheckpointCallback(TrainerCallback):
    def __init__(self):
        self.best_reward = float('-inf')
        self.best_loss = float('inf')
        self.best_checkpoint = None

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        current_reward = metrics.get("eval_rewards/accuracies")
        current_loss = metrics.get("eval_loss")
        # No eval metrics (e.g. empty valid split) → nothing to track this round.
        if current_reward is None:
            return
        if current_loss is None:
            current_loss = float('inf')

        if (current_reward > self.best_reward) or (current_reward == self.best_reward and current_loss < self.best_loss):
            self.best_reward = current_reward
            self.best_loss = current_loss
            self.best_checkpoint = state.global_step

    def get_best_checkpoint(self):
        return self.best_checkpoint

    def get_best_metrics(self):
        return {'accuracy': self.best_reward, 'loss': self.best_loss}

    def clean_checkpoints(self, output_dir):
        """Remove all non-best checkpoints and rename the best one to 'checkpoint-best'"""

        if self.best_checkpoint is None:
            print("No best checkpoint found.")
            return False

        best_checkpoint_path = os.path.join(output_dir, f"checkpoint-{self.best_checkpoint}")
        best_target_path = os.path.join(output_dir, "checkpoint-best")

        if not os.path.exists(best_checkpoint_path):
            print(f"Best checkpoint directory {best_checkpoint_path} not found.")
            return False

        checkpoint_dirs = [d for d in glob(os.path.join(output_dir, "checkpoint-*")) if os.path.isdir(d)]

        if os.path.isdir(best_target_path):
            print("Checkpoint-best already exists")
            return False

        try:
            shutil.copytree(best_checkpoint_path, best_target_path)
        except Exception as e:
            print(f"Error copying best checkpoint: {e}")
            return False

        for checkpoint_dir in checkpoint_dirs:
            try:
                assert os.path.basename(checkpoint_dir).split("-")[0] == "checkpoint"
                shutil.rmtree(checkpoint_dir)
                print(f"Removed checkpoint: {checkpoint_dir}")
            except Exception as e:
                print(f"Error removing checkpoint {checkpoint_dir}: {e}")

        print(f"Successfully renamed checkpoint-{self.best_checkpoint} to checkpoint-best")
        return True


class StepwiseEvalCallback(TrainerCallback):
    """Drive the eval/checkpoint schedule explicitly instead of via a fixed step interval:

    - step 0, evaluated/saved before any optimizer step (a pre-training baseline)
    - every `eval_fraction` of total training steps (default 20% -> ~5 evals mid-run)
    - the final training step, always, even if it doesn't land on a 20%-interval boundary

    `args.eval_strategy` / `args.save_strategy` are forced to "no" in main() so this callback
    is the single source of truth for when eval/save happens — no double-triggering against
    the trainer's built-in interval logic, and nothing here ever sets
    control.should_training_stop, so num_train_epochs alone still governs run length.
    """

    def __init__(self, eval_fraction=0.2):
        self.eval_fraction = eval_fraction
        self.eval_interval = None
        self.trainer = None  # wired in by main() right after DPOTrainer construction

    def on_train_begin(self, args, state, control, **kwargs):
        # state.max_steps is the trainer's own total-step count, derived from the train
        # dataset size, per_device_train_batch_size, gradient_accumulation_steps, and
        # num_train_epochs — the authoritative source, so our schedule can't drift off by one
        # from what the trainer actually runs.
        self.eval_interval = max(1, round(state.max_steps * self.eval_fraction))
        print(f"[eval schedule] {state.max_steps} total training steps -> eval/checkpoint every "
              f"{self.eval_interval} steps (~{self.eval_fraction:.0%} of training), "
              f"plus step 0 and the final step")

        # Baseline eval before any training happens. Trainer.evaluate() fires on_evaluate for
        # every registered callback (including BestCheckpointCallback), so this baseline is
        # tracked as a candidate "best" just like every other eval.
        metrics = self.trainer.evaluate()
        self.trainer.log_metrics("eval", metrics)

        ckpt_dir = os.path.join(args.output_dir, "checkpoint-0")
        self.trainer.save_model(ckpt_dir)
        if getattr(args, "push_to_hub", False) and getattr(args, "hub_model_id", None):
            push_checkpoint_to_hub(args, ckpt_dir, 0)

    def on_step_end(self, args, state, control, **kwargs):
        is_interval = state.global_step % self.eval_interval == 0
        is_last_step = state.global_step == state.max_steps
        if is_interval or is_last_step:
            control.should_evaluate = True
            control.should_save = True
        return control


class EvalMetricsLoggerCallback(TrainerCallback):
    """Append every eval's metrics to a JSONL file so they can be plotted later.

    One line per eval point (step 0, each ~20% interval, final step) with the global step,
    epoch, and all DPO eval metrics (eval_loss, eval_rewards/accuracies, eval_rewards/margins,
    eval_logps/*, …). Plain (uncompressed) JSONL: it's tiny and you always want it readable —
    load with `pandas.read_json(path, lines=True)`.
    """

    def __init__(self, metrics_path):
        self.metrics_path = metrics_path

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        row = {"step": state.global_step, "epoch": state.epoch}
        row.update(metrics)
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class EvalGenerationCallback(TrainerCallback):
    """At every eval point, generate the policy's responses to the valid-split prompts and
    save them to a gzipped JSONL, one file per checkpoint: eval_generations/step-<N>.jsonl.gz.

    Decoding: thinking-mode sampling (temp 0.6 / top_p 0.95 / top_k 20, 1 sample/prompt) — the
    same preset the curation module used to generate the DPO data, so eval generations reflect
    real deployment behaviour. Seeded from training_args.seed each time (and the RNG state is
    saved/restored around generation) so training is not perturbed and the sampling is
    reproducible across checkpoints — differences then reflect the weights, plus sampling noise.

    Gzipped JSONL keeps the (long, repetitive <think>…</think>) text compact on disk while
    staying streamable and human-inspectable:
        import gzip, json
        with gzip.open(path, "rt") as f:
            rows = [json.loads(l) for l in f]
    """

    # Qwen3 thinking-mode decoding preset (matches DPO-train-dataset-curation/config.py).
    TEMPERATURE = 0.6
    TOP_P = 0.95
    TOP_K = 20

    def __init__(self, prompts, out_dir, tokenizer, seed, max_new_tokens, batch_size,
                 max_prompt_length, use_gradient_checkpointing, gradient_checkpointing_kwargs):
        self.prompts = prompts
        self.out_dir = out_dir
        self.tokenizer = tokenizer
        self.seed = seed
        self.max_new_tokens = max_new_tokens
        self.batch_size = max(1, batch_size)
        self.max_prompt_length = max_prompt_length
        self.use_gc = bool(use_gradient_checkpointing)
        self.gc_kwargs = gradient_checkpointing_kwargs
        self.trainer = None  # wired in by main() right after DPOTrainer construction
        os.makedirs(out_dir, exist_ok=True)

    def _render(self, prompt):
        # add_generation_prompt=True appends the assistant turn header so the model continues
        # as the assistant; enable_thinking=True keeps Qwen3 free to emit <think>…</think>.
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )

    def _generate(self, model):
        device = next(model.parameters()).device
        outputs = []
        for b in range(0, len(self.prompts), self.batch_size):
            batch = self.prompts[b:b + self.batch_size]
            rendered = [self._render(p) for p in batch]
            enc = self.tokenizer(
                rendered, return_tensors="pt", padding=True, add_special_tokens=False,
                truncation=True, max_length=self.max_prompt_length,
            ).to(device)
            with torch.no_grad():
                gen = model.generate(
                    **enc, max_new_tokens=self.max_new_tokens, do_sample=True,
                    temperature=self.TEMPERATURE, top_p=self.TOP_P, top_k=self.TOP_K,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            new_tokens = gen[:, enc["input_ids"].shape[1]:]
            for row_ids in new_tokens:
                text = self.tokenizer.decode(row_ids, skip_special_tokens=True)
                n_tok = int((row_ids != self.tokenizer.pad_token_id).sum())
                outputs.append((text, n_tok))
        return outputs

    def on_evaluate(self, args, state, control, **kwargs):
        if not self.prompts:
            return
        model = self.trainer.model
        step = state.global_step

        print(f"[eval-gen] step {step}: generating {len(self.prompts)} responses "
              f"(thinking-mode sampling, max_new_tokens={self.max_new_tokens}) ...")

        # Preserve training state: eval mode, KV cache on, gradient checkpointing off (it forces
        # use_cache=False → very slow generation), and the RNG. All restored in `finally`.
        was_training = model.training
        prev_use_cache = getattr(model.config, "use_cache", None)
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        model.eval()
        if self.use_gc:
            model.gradient_checkpointing_disable()
        model.config.use_cache = True
        hf_set_seed(self.seed)
        try:
            results = self._generate(model)
        finally:
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            torch.set_rng_state(cpu_rng)
            model.config.use_cache = prev_use_cache
            if self.use_gc:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=self.gc_kwargs)
            if was_training:
                model.train()

        path = os.path.join(self.out_dir, f"step-{step}.jsonl.gz")
        with gzip.open(path, "wt", encoding="utf-8") as f:
            for i, (prompt, (gen_text, n_tok)) in enumerate(zip(self.prompts, results)):
                f.write(json.dumps({
                    "step": step, "prompt_idx": i,
                    "prompt": prompt, "generation": gen_text, "gen_tokens": n_tok,
                }, ensure_ascii=False) + "\n")
        print(f"[eval-gen] step {step}: saved -> {path}")


def main(script_args, training_args, model_config):

    set_seed(training_args.seed, full_determinism=True)

    # Load dataset (the handoff from the curation module — columns: prompt/chosen/rejected)
    print("Loading dataset from: ", script_args.dataset_path)
    dataset = datasets.load_from_disk(script_args.dataset_path)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        cache_dir=os.environ.get("HF_MODELS_CACHE_DIR")
    )

    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    if tokenizer.pad_token is None:
        # Model-agnostic fallback (Qwen3 already defines a pad token, so this no-ops).
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'

    # Prepare training data
    def format_assistant_turn(content):
        # Format a single assistant turn WITHOUT routing through apply_chat_template:
        # Qwen3's chat template strips <think>...</think> from assistant messages, but for
        # thinking-mode DPO we need the full reasoning response preserved in the
        # chosen/rejected completions.
        return f"<|im_start|>assistant\n{content}<|im_end|>\n"

    def format_dpo_sample(sample):
        # Harmlessness pairs have a single user turn (the harmful prompt). The chosen/rejected
        # completions already contain the <think>…</think> reasoning from generation.
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": sample["prompt"]}],
            tokenize=False,
            enable_thinking=True,
        )
        chosen = format_assistant_turn(sample["chosen"])
        rejected = format_assistant_turn(sample["rejected"])
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

    # Capture the RAW valid prompts BEFORE the map: EvalGenerationCallback re-templates them
    # with add_generation_prompt=True to sample fresh responses, whereas the mapped `prompt`
    # column ends right before the assistant turn (so it can be concatenated with chosen/rejected).
    raw_valid_prompts = list(dataset["valid"]["prompt"]) if "valid" in dataset else []

    dataset = dataset.map(format_dpo_sample)
    dataset = dataset.select_columns(['prompt', 'chosen', 'rejected'])

    # Eval/checkpoint schedule is fully owned by StepwiseEvalCallback (step 0, every ~20% of
    # total training steps, and the final step) — force off the trainer's own interval-based
    # eval/save so the two mechanisms can't double-trigger or disagree. This also guarantees
    # no early stopping: num_train_epochs alone determines how long the run goes.
    training_args.eval_strategy = "no"
    training_args.save_strategy = "no"

    # LoRA config
    peft_config = get_peft_config(model_config)
    assert peft_config
    ref_model = None

    best_checkpoint_callback = BestCheckpointCallback()
    stepwise_eval_callback = StepwiseEvalCallback(eval_fraction=0.2)
    # Per-eval artifacts (both fire on every eval point, incl. step 0 and the final step):
    #   eval_metrics.jsonl        — DPO eval metrics per step (for plotting curves)
    #   eval_generations/step-*.jsonl.gz — the policy's sampled responses to the valid prompts
    metrics_logger = EvalMetricsLoggerCallback(
        os.path.join(training_args.output_dir, "eval_metrics.jsonl"))
    callbacks = [best_checkpoint_callback, stepwise_eval_callback, metrics_logger]

    gen_callback = None
    if getattr(script_args, "eval_generate", True) and raw_valid_prompts:
        gen_callback = EvalGenerationCallback(
            prompts=raw_valid_prompts,
            out_dir=os.path.join(training_args.output_dir, "eval_generations"),
            tokenizer=tokenizer,
            seed=training_args.seed,
            max_new_tokens=script_args.eval_gen_max_new_tokens,
            batch_size=script_args.eval_gen_batch_size,
            max_prompt_length=training_args.max_prompt_length,
            use_gradient_checkpointing=getattr(training_args, "gradient_checkpointing", False),
            gradient_checkpointing_kwargs=getattr(training_args, "gradient_checkpointing_kwargs", None),
        )
        callbacks.append(gen_callback)

    # Push-to-hub is optional: only wire the push callback when explicitly requested.
    if getattr(training_args, "push_to_hub", False) and getattr(training_args, "hub_model_id", None):
        callbacks.append(PushStepCheckpointCallback())

    trainer = DPOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=callbacks,
    )
    stepwise_eval_callback.trainer = trainer
    if gen_callback is not None:
        gen_callback.trainer = trainer
    trainer.train()

    # Get the best checkpoint and metrics
    best_step = best_checkpoint_callback.get_best_checkpoint()
    best_metrics = best_checkpoint_callback.get_best_metrics()
    print(f"Best checkpoint at step {best_step} with metrics: {best_metrics}")

    # Remove all non-best checkpoints and rename the best to checkpoint-best
    best_checkpoint_callback.clean_checkpoints(trainer.args.output_dir)


@dataclass
class ScriptArguments:
    dataset_path: str
    # Per-eval generation on the valid prompts (thinking-mode sampling), saved as
    # eval_generations/step-<N>.jsonl.gz. On by default; set --eval_generate False to skip it
    # (e.g. if the valid set is large and the extra per-eval decode time is too costly).
    eval_generate: bool = True
    eval_gen_max_new_tokens: int = 8192   # match the curation completion budget so CoT isn't truncated
    eval_gen_batch_size: int = 8


if __name__ == '__main__':
    parser = TrlParser((ScriptArguments, DPOConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()
    print(f"SCRIPT ARGS:\n{script_args}")
    print(f"TRAINING ARGS:\n{training_args}")
    print(f"MODEL CONFIG:\n{model_config}")
    main(script_args, training_args, model_config)
