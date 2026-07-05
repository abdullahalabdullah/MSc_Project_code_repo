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
    callbacks = [best_checkpoint_callback, stepwise_eval_callback]
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


if __name__ == '__main__':
    parser = TrlParser((ScriptArguments, DPOConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()
    print(f"SCRIPT ARGS:\n{script_args}")
    print(f"TRAINING ARGS:\n{training_args}")
    print(f"MODEL CONFIG:\n{model_config}")
    main(script_args, training_args, model_config)
