# ---------------------------------------------------------------------------
# This file is adapted from train_dpo.py (same lineage: the codebase accompanying
#   "Truthful or Fabricated? Using Causal Attribution to Mitigate
#    Reward Hacking in Explanations"
#   Paper:  https://openreview.net/pdf?id=nkdPLuKoL5
#   Source: https://github.com/PedroMLF/Reward-Hacking-in-Explanations
# )
# ---------------------------------------------------------------------------
"""LoRA SFT training on the CHOSEN responses of the preference pairs (controlled ablation).

This is the SFT arm of the DPO-vs-SFT ablation (SFT_Arm_Developer_Spec.md): identical to the
DPO arm in data, base model, LoRA config, sequence formatting, seeds, and checkpoint schedule,
but the loss is plain causal-LM on the chosen response only — DPO with the negative (rejected)
gradient removed. Loss is computed on response tokens only (prompt masked), including the
final EOS.

Consumes the DatasetDict produced by convert_dpo_to_sft.py:
    DatasetDict({"train","valid"}) with columns prompt / chosen.

Run via train.sh (METHOD=sft), or directly:
    python train_sft.py --dataset_path <.../sft_dataset> --model_name_or_path Qwen/Qwen3-4B ...
"""
import json
import os
import shutil
from dataclasses import dataclass, field
from glob import glob

import hashlib
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
    ModelConfig,
    SFTConfig,
    SFTTrainer,
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


def gpu_memory_stats():
    """Current + peak CUDA memory (GB). Cheap; returns {} on CPU-only.

    peak_reserved is the number that matters for OOM headroom: it includes allocator
    fragmentation and is what actually competes with new allocations. Read it after a run
    to decide whether per_device_train_batch_size can go up.
    """
    if not torch.cuda.is_available():
        return {}
    dev = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(dev).total_memory
    return {
        "gpu_mem_alloc_gb": round(torch.cuda.memory_allocated(dev) / 1e9, 2),
        "gpu_mem_peak_alloc_gb": round(torch.cuda.max_memory_allocated(dev) / 1e9, 2),
        "gpu_mem_reserved_gb": round(torch.cuda.memory_reserved(dev) / 1e9, 2),
        "gpu_mem_peak_reserved_gb": round(torch.cuda.max_memory_reserved(dev) / 1e9, 2),
        "gpu_mem_total_gb": round(total / 1e9, 2),
    }


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_dir(root, exclude=("provenance.json", ".verify_passed")):
    """Deterministic hash of a directory tree (same scheme as convert_dpo_to_sft.py: the
    provenance file and train_sft.sh's verification marker are excluded so the hash recorded
    at conversion time stays valid after those files land in the dataset dir)."""
    lines = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in sorted(filenames):
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, root)
            if os.path.basename(rel) in exclude:
                continue
            lines.append(f"{rel}  {sha256_file(path)}")
    return hashlib.sha256("\n".join(sorted(lines)).encode("utf-8")).hexdigest()


def record_dataset_provenance(dataset_path, output_dir):
    """Spec §2.1: make the data lineage auditable in the run metadata.

    Recomputes the dataset dir hash, merges it with the converter's provenance.json (source
    DPO file + derived SFT file hashes), and writes OUTPUT_DIR/dataset_provenance.json.
    Returns the record (also pushed into wandb.config by main() when W&B is live).
    """
    recomputed = sha256_dir(dataset_path)
    converter_prov = None
    prov_path = os.path.join(dataset_path, "provenance.json")
    if os.path.isfile(prov_path):
        with open(prov_path, encoding="utf-8") as f:
            converter_prov = json.load(f)
    record = {
        "dataset_path": dataset_path,
        "dataset_sha256_recomputed": recomputed,
        "converter_provenance": converter_prov,
        "hash_matches_converter": (
            converter_prov is not None and converter_prov.get("derived_sha256") == recomputed),
    }
    if converter_prov is None:
        print(f"WARNING: no provenance.json in {dataset_path} — was this dataset produced by "
              "convert_dpo_to_sft.py? Provenance will be incomplete.")
    elif not record["hash_matches_converter"]:
        print("WARNING: dataset hash does NOT match the converter's recorded derived_sha256 — "
              "the sft_dataset was modified after conversion. Do not trust this run's lineage.")
    elif converter_prov.get("canonical") is False:
        print("WARNING: training on a NON-canonical (--limit smoke) sft_dataset.")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "dataset_provenance.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"Dataset provenance -> {out_path}")
    return record


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
    """Track the best checkpoint by eval_loss (lower is better).

    SFT has no preference-reward metrics (the DPO arm selects by eval_rewards/accuracies);
    eval_loss on the shared valid split is the natural SFT analogue. Kept for parity with the
    DPO arm's checkpoint-best machinery — the dose–response analysis uses ALL pushed step
    checkpoints regardless.
    """

    def __init__(self):
        self.best_loss = float('inf')
        self.best_token_accuracy = None
        self.best_checkpoint = None

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        current_loss = metrics.get("eval_loss")
        # No eval metrics (e.g. empty valid split) → nothing to track this round.
        if current_loss is None:
            return
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_token_accuracy = metrics.get("eval_mean_token_accuracy")
            self.best_checkpoint = state.global_step

    def get_best_checkpoint(self):
        return self.best_checkpoint

    def get_best_metrics(self):
        return {'loss': self.best_loss, 'mean_token_accuracy': self.best_token_accuracy}

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
    - the final training step, always, even if it doesn't land on an interval boundary

    `args.eval_strategy` / `args.save_strategy` are forced to "no" in main() so this callback
    is the single source of truth for when eval/save happens — no double-triggering against
    the trainer's built-in interval logic, and nothing here ever sets
    control.should_training_stop, so num_train_epochs alone still governs run length.
    """

    def __init__(self, eval_fraction=0.2):
        self.eval_fraction = eval_fraction
        self.eval_interval = None
        self.trainer = None  # wired in by main() right after SFTTrainer construction

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

    One line per eval point (step 0, each interval, final step) with the global step, epoch,
    and all SFT eval metrics (eval_loss, eval_mean_token_accuracy, …). Plain (uncompressed)
    JSONL: it's tiny and you always want it readable — load with
    `pandas.read_json(path, lines=True)`.
    """

    def __init__(self, metrics_path):
        self.metrics_path = metrics_path
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        open(metrics_path, "w").close()  # fresh file per run (this pipeline doesn't resume)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        row = {"step": state.global_step, "epoch": state.epoch}
        row.update(metrics)
        # GPU memory alongside each eval point: after a (smoke) run, compare
        # gpu_mem_peak_reserved_gb against gpu_mem_total_gb to judge whether
        # per_device_train_batch_size can be raised.
        row.update(gpu_memory_stats())
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(script_args, training_args, model_config):

    set_seed(training_args.seed, full_determinism=True)

    # Load dataset (the handoff from convert_dpo_to_sft.py — columns: prompt/chosen)
    print("Loading dataset from: ", script_args.dataset_path)
    dataset = datasets.load_from_disk(script_args.dataset_path)
    for split, d in dataset.items():
        if "rejected" in d.column_names:
            raise SystemExit(
                f"ERROR: split '{split}' still has a 'rejected' column — this looks like the raw "
                "dpo_dataset. The SFT arm must train on the converted dataset so its provenance "
                "is recorded:  python convert_dpo_to_sft.py <path/to/dpo_dataset>")
        missing = {"prompt", "chosen"} - set(d.column_names)
        if missing:
            raise SystemExit(f"ERROR: split '{split}' is missing columns {missing} — "
                             "expected the convert_dpo_to_sft.py export (prompt/chosen).")

    # Auditable lineage (spec §2.1): source DPO file + derived SFT file hashes into the run dir.
    provenance = record_dataset_provenance(script_args.dataset_path, training_args.output_dir)

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

    # Prepare training data — the formatting MUST reproduce, token for token, what the DPO
    # trainer saw for the chosen sequence (verified offline by verify_sft_tokenization.py).
    def format_assistant_turn(content):
        # Format a single assistant turn WITHOUT routing through apply_chat_template:
        # Qwen3's chat template strips <think>...</think> from assistant messages, but we
        # need the full reasoning response preserved in the completion — same rationale and
        # same string as train_dpo.py.
        return f"<|im_start|>assistant\n{content}<|im_end|>\n"

    def format_sft_sample(sample):
        # Single user turn (the harmful prompt); chosen already contains the <think>…</think>
        # reasoning from generation. Prompt formatting is identical to format_dpo_sample.
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": sample["prompt"]}],
            tokenize=False,
            enable_thinking=True,
        )
        # DPOTrainer.tokenize_row appended eos_token_id after the formatted chosen string, so
        # the DPO arm trained on sequences ending "<|im_end|>\n" + <|im_end|> (a double-EOS
        # quirk). We append the same EOS explicitly to stay token-identical — do NOT "fix"
        # this, the ablation depends on matching what DPO actually trained on. With the eos
        # already at the end of the string, trl's own add_eos step is a no-op.
        completion = format_assistant_turn(sample["chosen"]) + tokenizer.eos_token
        return {"prompt": prompt, "completion": completion}

    dataset = dataset.map(format_sft_sample)
    dataset = dataset.select_columns(['prompt', 'completion'])

    # Eval/checkpoint schedule is fully owned by StepwiseEvalCallback (step 0, every interval,
    # and the final step) — force off the trainer's own interval-based eval/save so the two
    # mechanisms can't double-trigger or disagree. This also guarantees no early stopping:
    # num_train_epochs alone determines how long the run goes.
    training_args.eval_strategy = "no"
    training_args.save_strategy = "no"

    # Loss guarantees (spec §2.2), independent of CLI flags: loss on response tokens only
    # (prompt masked; trl masks via completion_mask), and one example per sequence (packing
    # would concatenate examples and break the DPO-matched tokenization).
    training_args.completion_only_loss = True
    assert not training_args.packing, "packing must stay off — it breaks the DPO-matched ablation"

    # LoRA config
    peft_config = get_peft_config(model_config)
    assert peft_config

    best_checkpoint_callback = BestCheckpointCallback()
    stepwise_eval_callback = StepwiseEvalCallback(eval_fraction=script_args.eval_fraction)
    # eval_metrics.jsonl — SFT eval metrics per eval point (step 0, each interval, final step)
    # for plotting training curves. Actual behavioural eval (safety + monitorability) is done
    # separately, offline, on the pushed checkpoints — not here.
    metrics_logger = EvalMetricsLoggerCallback(
        os.path.join(training_args.output_dir, "eval_metrics.jsonl"))
    callbacks = [best_checkpoint_callback, stepwise_eval_callback, metrics_logger]

    # Push-to-hub is optional: only wire the push callback when explicitly requested.
    if getattr(training_args, "push_to_hub", False) and getattr(training_args, "hub_model_id", None):
        callbacks.append(PushStepCheckpointCallback())

    trainer = SFTTrainer(
        model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=callbacks,
    )
    stepwise_eval_callback.trainer = trainer

    # Surface the dataset lineage in W&B too (no-op if W&B isn't the reporter).
    try:
        import wandb
        if wandb.run is not None:
            wandb.config.update({"dataset_provenance": provenance}, allow_val_change=True)
    except Exception:
        pass

    trainer.train()

    # Dump the trainer's FULL structured log history — every logging-step training loss +
    # learning rate + grad norm, and every eval point — as one JSON. This is the complete,
    # W&B-independent record for offline loss/LR/eval curves (eval_metrics.jsonl is the
    # eval-only subset; logs/train.log is the same data but unstructured text).
    log_history_path = os.path.join(training_args.output_dir, "train_log_history.json")
    with open(log_history_path, "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print(f"Full training log history -> {log_history_path}")

    # GPU memory summary for batch-size decisions: peak reserved vs total is the OOM-relevant
    # headroom. Saved to gpu_memory.json (per-eval snapshots are in eval_metrics.jsonl).
    mem = gpu_memory_stats()
    if mem:
        mem_path = os.path.join(training_args.output_dir, "gpu_memory.json")
        with open(mem_path, "w", encoding="utf-8") as f:
            json.dump(mem, f, indent=2)
        headroom = mem["gpu_mem_total_gb"] - mem["gpu_mem_peak_reserved_gb"]
        print(f"GPU memory: peak reserved {mem['gpu_mem_peak_reserved_gb']} GB / "
              f"{mem['gpu_mem_total_gb']} GB total -> {headroom:.1f} GB headroom "
              f"(details: {mem_path})")

    # Get the best checkpoint and metrics
    best_step = best_checkpoint_callback.get_best_checkpoint()
    best_metrics = best_checkpoint_callback.get_best_metrics()
    print(f"Best checkpoint at step {best_step} with metrics: {best_metrics}")

    # Remove all non-best checkpoints and rename the best to checkpoint-best
    best_checkpoint_callback.clean_checkpoints(trainer.args.output_dir)

    # Record WHICH checkpoint won — including its HF repo id when checkpoints were pushed — so
    # you can identify the best among the pushed "<hub_model_id>-step-<N>" repos after the fact.
    pushed = getattr(training_args, "push_to_hub", False) and getattr(training_args, "hub_model_id", None)
    best_hf_repo = f"{training_args.hub_model_id}-step-{best_step}" if (pushed and best_step is not None) else None
    best_info_path = os.path.join(training_args.output_dir, "best_checkpoint.txt")
    with open(best_info_path, "w", encoding="utf-8") as f:
        f.write("Best checkpoint (selected by eval_loss — lower is better; SFT has no reward metrics)\n")
        f.write(f"step: {best_step}\n")
        f.write(f"eval_loss: {best_metrics.get('loss')}\n")
        f.write(f"eval_mean_token_accuracy: {best_metrics.get('mean_token_accuracy')}\n")
        f.write(f"local: {os.path.join(training_args.output_dir, 'checkpoint-best')}\n")
        if best_hf_repo:
            f.write(f"hf_repo: {best_hf_repo}\n")
            f.write(f"hf_url: https://huggingface.co/{best_hf_repo}\n")
        else:
            f.write("hf_repo: (checkpoints were not pushed — run with --push_to_hub/HUB_MODEL_ID)\n")
    print(f"Best-checkpoint info -> {best_info_path}"
          + (f" (HF: {best_hf_repo})" if best_hf_repo else ""))

    # Also surface the winner in the W&B run summary (no-op if W&B isn't the reporter).
    try:
        import wandb
        if wandb.run is not None:
            wandb.run.summary["best_step"] = best_step
            wandb.run.summary["best_eval_loss"] = best_metrics.get("loss")
            wandb.run.summary["best_eval_mean_token_accuracy"] = best_metrics.get("mean_token_accuracy")
            if best_hf_repo:
                wandb.run.summary["best_hf_repo"] = best_hf_repo
    except Exception:
        pass


@dataclass
class ScriptArguments:
    dataset_path: str
    # Checkpoint/eval cadence as a fraction of total steps. 0.2 matches the DPO run's schedule
    # (step 0, every ~20%, final step). The spec allows DENSER checkpoints for the SFT arm's
    # dose–response curve — pass e.g. --eval_fraction 0.1 (EVAL_FRACTION=0.1 via train.sh).
    eval_fraction: float = field(default=0.2)


if __name__ == '__main__':
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()
    print(f"SCRIPT ARGS:\n{script_args}")
    print(f"TRAINING ARGS:\n{training_args}")
    print(f"MODEL CONFIG:\n{model_config}")
    main(script_args, training_args, model_config)
