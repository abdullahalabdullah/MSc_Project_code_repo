"""
RO2 MVP — DPO harmlessness training (TRL 0.8.x + Unsloth, bf16 LoRA).

Trains one DPO LoRA adapter on PKU-SafeRLHF conflict cases, starting from
`allenai/Llama-3.1-Tulu-3-8B-SFT-no-safety-data`, selects the best checkpoint
by validation reward-accuracy (with the Ferreira et al. tie-break), and pushes
the selected ADAPTER (not a merged model) to a private Hugging Face Hub repo.

This module is structured as importable functions so `sanity_check_dpo.py` can
reuse the exact same data-loading / formatting / training / selection / push
code path on a tiny slice of data, rather than duplicating logic.

Run the full training:
    python train_dpo.py --config configs/harmlessness.yaml

Strict Unsloth: there is no non-Unsloth fallback. If the pinned environment
(trl==0.8.6, transformers==4.44.0, peft==0.12.0, unsloth) is incompatible, the
import below fails loudly. Use `sanity_check_dpo.py` first to confirm the
environment works before committing the GPU allocation to a multi-hour run.
"""

# -----------------------------------------------------------------------------
# Import order matters: unsloth must be imported BEFORE transformers/trl/peft so
# its memory-efficient kernels and the DPO reference-model patch are applied.
# -----------------------------------------------------------------------------
import unsloth  # noqa: F401  (import-for-side-effects; must be first)
from unsloth import FastLanguageModel, PatchDPOTrainer

import argparse
import datetime as _dt
import gc
import json
import logging
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from huggingface_hub import HfApi, login
from datasets import load_dataset, Dataset, DatasetDict
from transformers import TrainingArguments
from trl import DPOTrainer  # TRL 0.8.x: DPO params are DPOTrainer kwargs, not a DPOConfig


# =============================================================================
# Logging
# =============================================================================
def make_logger(name: str = "train_dpo") -> logging.Logger:
    """Stream logs to stdout so the SLURM .out file captures everything."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s  %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


# =============================================================================
# Config
# =============================================================================
def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` onto a copy of `base`."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(config_path: str, override_path: str | None = None) -> dict:
    """Load a YAML config, optionally deep-merging an override file on top.

    The sanity-check config holds only overrides relative to the full config,
    so `sanity_check_dpo.py` calls this with both paths.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if override_path is not None:
        with open(override_path, "r", encoding="utf-8") as f:
            override = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, override)
    return cfg


# =============================================================================
# Credentials
# =============================================================================
def check_credentials(logger: logging.Logger) -> str:
    """Load .env and fail fast if HF_TOKEN is missing.

    Returns the HF token. OPENAI_API_KEY is only needed by the downstream eval
    pipeline (not training), so its absence is a warning, not an error.
    """
    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN is not set. Create a .env file in the MVP-RO2 root with "
            "HF_TOKEN=hf_... (write access to the target hub repo). Aborting "
            "before any GPU work begins."
        )
    if not os.environ.get("OPENAI_API_KEY"):
        logger.warning(
            "OPENAI_API_KEY is not set. Not needed for training, but the "
            "downstream RO1 eval/monitor step will require it later."
        )
    login(token=hf_token)
    logger.info("Hugging Face authentication OK (HF_TOKEN loaded from .env).")
    return hf_token


# =============================================================================
# Dataset: PKU-SafeRLHF -> conflict cases -> (prompt, chosen, rejected)
# =============================================================================
# PKU-Alignment/PKU-SafeRLHF schema (per the dataset card):
#   prompt, response_0, response_1,
#   is_response_0_safe (bool), is_response_1_safe (bool),
#   better_response_id (0/1, helpfulness), safer_response_id (0/1, safety)
_REQUIRED_COLUMNS = [
    "prompt",
    "response_0",
    "response_1",
    "is_response_0_safe",
    "is_response_1_safe",
    "better_response_id",
    "safer_response_id",
]


def _is_conflict(example: dict, mode: str) -> bool:
    """Return True if the example is a 'conflict case' under the given mode.

    - "better_ne_safer": the safer response is NOT the more helpful one.
    - "safe_xor":        exactly one of the two responses is safe.
    """
    if mode == "better_ne_safer":
        return int(example["better_response_id"]) != int(example["safer_response_id"])
    if mode == "safe_xor":
        return bool(example["is_response_0_safe"]) != bool(example["is_response_1_safe"])
    raise ValueError(f"Unknown conflict_filter_mode: {mode!r}")


def filter_conflict_cases(ds: Dataset, mode: str) -> Dataset:
    """Keep only conflict-case rows under `mode`."""
    return ds.filter(lambda ex: _is_conflict(ex, mode))


def to_dpo_format(example: dict, tokenizer) -> dict:
    """Convert one PKU-SafeRLHF row to TRL's (prompt, chosen, rejected) format.

    - prompt:  the user turn, chat-templated with the assistant generation
               prompt appended (ends at the assistant-turn boundary).
    - chosen:  the SAFER response, raw text only (no role markers / template
               tokens). TRL's DPOTrainer appends EOS during tokenization, so we
               do NOT add it here.
    - rejected: the less-safe response, raw text only.
    """
    safer_id = int(example["safer_response_id"])
    chosen_text = example[f"response_{safer_id}"]
    rejected_text = example[f"response_{1 - safer_id}"]

    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": example["prompt"]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return {
        "prompt": prompt_text,
        "chosen": chosen_text,
        "rejected": rejected_text,
    }


def load_and_prepare_dataset(cfg: dict, tokenizer, logger: logging.Logger) -> DatasetDict:
    """Load PKU-SafeRLHF, filter conflict cases, subsample, format, and split.

    Returns a DatasetDict with "train" and "validation" splits, each row in
    TRL DPO format {prompt, chosen, rejected}.
    """
    logger.info("Loading dataset %s [split=%s] ...", cfg["dataset"], cfg["dataset_split"])
    ds = load_dataset(cfg["dataset"], split=cfg["dataset_split"])
    logger.info("Raw dataset size: %d", len(ds))
    logger.info("Dataset columns: %s", ds.column_names)

    missing = [c for c in _REQUIRED_COLUMNS if c not in ds.column_names]
    if missing:
        raise RuntimeError(
            f"PKU-SafeRLHF is missing expected columns {missing}. "
            f"Found columns: {ds.column_names}. Check the dataset card / revision."
        )

    # Log conflict counts under BOTH definitions for transparency, then use the
    # configured one. The ~12,616 target disambiguates which definition is meant.
    n_better_ne_safer = len(filter_conflict_cases(ds, "better_ne_safer"))
    n_safe_xor = len(filter_conflict_cases(ds, "safe_xor"))
    logger.info(
        "Conflict-case counts:  better_ne_safer=%d  |  safe_xor=%d  "
        "(expected ~%d)",
        n_better_ne_safer, n_safe_xor, cfg["filtered_pool_size"],
    )

    mode = cfg["conflict_filter_mode"]
    filtered = filter_conflict_cases(ds, mode)
    logger.info(
        "Using conflict_filter_mode=%r -> filtered pool size = %d "
        "(config expected %d).",
        mode, len(filtered), cfg["filtered_pool_size"],
    )
    if len(filtered) < cfg["subsample_size"] + cfg["val_size"]:
        raise RuntimeError(
            f"Filtered pool ({len(filtered)}) is smaller than "
            f"subsample_size ({cfg['subsample_size']}). Cannot subsample."
        )

    # Subsample filtered pool -> subsample_size (subsample_seed).
    filtered = filtered.shuffle(seed=cfg["subsample_seed"])
    subsample = filtered.select(range(cfg["subsample_size"]))
    logger.info(
        "Subsampled %d examples (seed=%d) from the filtered pool.",
        len(subsample), cfg["subsample_seed"],
    )

    # Map to DPO format. remove_columns drops all original PKU columns so the
    # trainer sees only {prompt, chosen, rejected}.
    formatted = subsample.map(
        lambda ex: to_dpo_format(ex, tokenizer),
        remove_columns=subsample.column_names,
        desc="Formatting to DPO (prompt, chosen, rejected)",
    )

    # Split subsample -> train / validation (random_seed).
    split = formatted.train_test_split(
        test_size=cfg["val_size"], seed=cfg["random_seed"], shuffle=True
    )
    dsd = DatasetDict({"train": split["train"], "validation": split["test"]})
    logger.info(
        "Final splits: train=%d, validation=%d.",
        len(dsd["train"]), len(dsd["validation"]),
    )
    return dsd


# =============================================================================
# Pre-training inspections (spec: manual format check + prompt-length check)
# =============================================================================
def inspect_formatting(dataset: Dataset, n: int, logger: logging.Logger) -> None:
    """Print the first `n` fully-formatted (prompt, chosen, rejected) triples.

    Manual check: confirm the prompt ends at the assistant-turn boundary, and
    chosen/rejected contain only response text (no leaked role markers).
    """
    n = min(n, len(dataset))
    logger.info("=" * 78)
    logger.info("FORMAT CHECK — first %d (prompt, chosen, rejected) triples:", n)
    logger.info("=" * 78)
    for i in range(n):
        ex = dataset[i]
        logger.info("--- example %d ---", i)
        logger.info("[PROMPT]\n%s", ex["prompt"])
        logger.info("[CHOSEN]\n%s", ex["chosen"])
        logger.info("[REJECTED]\n%s", ex["rejected"])
        logger.info("-" * 40)


def check_prompt_lengths(
    dataset: Dataset, tokenizer, max_prompt_length: int, logger: logging.Logger
) -> dict:
    """Tokenize prompts and report the length distribution vs max_prompt_length.

    Loudly warns if a non-trivial fraction of prompts would be truncated.
    Returns a small stats dict (also handy for the sanity-check assertions).
    """
    lengths = [
        len(tokenizer(ex["prompt"], add_special_tokens=False)["input_ids"])
        for ex in dataset
    ]
    arr = np.asarray(lengths)
    n_over = int((arr > max_prompt_length).sum())
    frac_over = n_over / max(len(arr), 1)
    stats = {
        "min": int(arr.min()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": int(arr.max()),
        "n_over_max_prompt_length": n_over,
        "frac_over_max_prompt_length": frac_over,
        "max_prompt_length": max_prompt_length,
    }
    logger.info("=" * 78)
    logger.info("PROMPT-LENGTH CHECK (tokens), max_prompt_length=%d:", max_prompt_length)
    logger.info(
        "  min=%d  p50=%.0f  p90=%.0f  p95=%.0f  p99=%.0f  max=%d",
        stats["min"], stats["p50"], stats["p90"], stats["p95"], stats["p99"], stats["max"],
    )
    logger.info(
        "  %d / %d prompts (%.1f%%) exceed max_prompt_length and would be truncated.",
        n_over, len(arr), 100.0 * frac_over,
    )
    if frac_over > 0.05:
        logger.warning(
            "More than 5%% of prompts exceed max_prompt_length=%d. Consider "
            "raising max_prompt_length to 1024 (and max_length to ~2560-3072) "
            "in the config before the full run.",
            max_prompt_length,
        )
    logger.info("=" * 78)
    return stats


# =============================================================================
# Model + tokenizer (Unsloth)
# =============================================================================
def load_model_and_tokenizer(cfg: dict, logger: logging.Logger):
    """Load the base model in bf16 via Unsloth and attach LoRA adapters.

    No quantisation (load_in_4bit=False): the A6000's 48 GB is enough for full
    bf16 LoRA, and quantisation would be a methodological confound. The DPO
    reference model is handled internally by Unsloth (ref_model=None later).
    """
    logger.info("Loading base model via Unsloth: %s", cfg["base_model"])
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["base_model"],
        max_seq_length=cfg["max_length"],
        dtype=None,            # None -> auto-detect bf16 on the A6000
        load_in_4bit=False,    # no quantisation
    )

    if tokenizer.pad_token is None:
        # Llama 3.1 has no dedicated pad token; reuse EOS for padding.
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("tokenizer.pad_token was None; set to eos_token (%s).", tokenizer.eos_token)

    logger.info(
        "Attaching LoRA: r=%d alpha=%d dropout=%.3f targets=%s",
        cfg["lora_r"], cfg["lora_alpha"], cfg["lora_dropout"], cfg["target_modules"],
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_r"],
        target_modules=cfg["target_modules"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg["random_seed"],
    )
    return model, tokenizer


# =============================================================================
# Trainer
# =============================================================================
def build_trainer(cfg, model, tokenizer, train_ds, eval_ds, logger: logging.Logger):
    """Construct the TRL 0.8.x DPOTrainer.

    On TRL 0.8.x there is no DPOConfig: standard training args go into a
    transformers.TrainingArguments, while the DPO-specific knobs (beta,
    max_length, max_prompt_length) are passed directly to DPOTrainer.

    PatchDPOTrainer() is called first so Unsloth's reference-model-via-disabled-
    adapters trick is active (this is what lets us use ref_model=None).
    """
    PatchDPOTrainer()

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        learning_rate=float(cfg["learning_rate"]),
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        optim=cfg["optim"],
        bf16=cfg["bf16"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        logging_steps=cfg["logging_steps"],
        eval_strategy=cfg["eval_strategy"],
        eval_steps=cfg["eval_steps"],
        save_strategy=cfg.get("save_strategy", "steps"),
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
        load_best_model_at_end=False,   # custom tie-aware selection runs after training
        remove_unused_columns=False,
        report_to="none",               # no wandb for the MVP; use the SLURM log
        seed=cfg["random_seed"],
        max_steps=int(cfg.get("max_steps", -1)),  # -1 = use epochs (full run)
    )

    logger.info(
        "TrainingArguments: epochs=%s max_steps=%s eff_batch=%d lr=%s beta=%s "
        "eval_steps=%d save_steps=%d",
        training_args.num_train_epochs, training_args.max_steps,
        cfg["per_device_train_batch_size"] * cfg["gradient_accumulation_steps"],
        training_args.learning_rate, cfg["beta"],
        training_args.eval_steps, training_args.save_steps,
    )

    trainer = DPOTrainer(
        model,
        ref_model=None,             # Unsloth handles the reference model internally
        args=training_args,
        beta=cfg["beta"],                       # TRL 0.8.x: DPO knobs are DPOTrainer kwargs
        max_length=cfg["max_length"],
        max_prompt_length=cfg["max_prompt_length"],
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )
    return trainer


# =============================================================================
# Checkpoint selection (Ferreira et al. B.2 tie-break)
# =============================================================================
def select_best_checkpoint(log_history: list, logger: logging.Logger):
    """Pick the best checkpoint from trainer.state.log_history.

    Two-stage selection with tie-breaks:
      1. max eval_rewards/accuracies
      2. among ties (|diff| < 1e-9): min eval_loss
      3. still tied: highest step (latest / most-trained)

    Returns (best_step, {"eval_rewards/accuracies", "eval_loss"}), or
    (None, None) if no eval points were recorded.
    """
    # Collect eval entries keyed by step.
    evals = {}
    for entry in log_history:
        if "eval_rewards/accuracies" in entry and "eval_loss" in entry:
            step = int(entry["step"])
            evals[step] = {
                "acc": float(entry["eval_rewards/accuracies"]),
                "loss": float(entry["eval_loss"]),
            }

    if not evals:
        logger.warning("No eval checkpoints found in log_history; cannot select.")
        return None, None

    max_acc = max(v["acc"] for v in evals.values())
    acc_ties = {s: v for s, v in evals.items() if abs(v["acc"] - max_acc) < 1e-9}
    min_loss = min(v["loss"] for v in acc_ties.values())
    loss_ties = {s: v for s, v in acc_ties.items() if abs(v["loss"] - min_loss) < 1e-9}
    best_step = max(loss_ties.keys())  # highest step among remaining ties

    # Log the full selection table.
    logger.info("=" * 78)
    logger.info("CHECKPOINT SELECTION TABLE")
    logger.info("  %-8s %-22s %-14s %-8s", "step", "eval_rewards/accuracies", "eval_loss", "selected")
    for step in sorted(evals.keys()):
        v = evals[step]
        logger.info(
            "  %-8d %-22.6f %-14.6f %-8s",
            step, v["acc"], v["loss"], "YES" if step == best_step else "",
        )
    logger.info("=" * 78)
    logger.info(
        "Selected checkpoint-%d  (acc=%.6f, loss=%.6f).",
        best_step, evals[best_step]["acc"], evals[best_step]["loss"],
    )

    return best_step, {
        "eval_rewards/accuracies": evals[best_step]["acc"],
        "eval_loss": evals[best_step]["loss"],
    }


# =============================================================================
# Push selected adapter to the Hub (adapter-only)
# =============================================================================
def _build_model_card(cfg: dict, best_step: int, metrics: dict) -> str:
    """Auto-generated model card recording the training provenance."""
    today = _dt.date.today().isoformat()
    acc = metrics.get("eval_rewards/accuracies") if metrics else None
    loss = metrics.get("eval_loss") if metrics else None
    return f"""---
base_model: {cfg['base_model']}
library_name: peft
tags:
- dpo
- lora
- harmlessness
- cot-monitorability
---

# tulu3-8b harmlessness DPO (RO2 MVP)

LoRA adapter trained with DPO on **PKU-SafeRLHF conflict cases** (harmlessness
objective), as part of the RO2 MVP on the effect of post-training on CoT
monitorability.

- **Base model:** `{cfg['base_model']}`
- **Training data:** `{cfg['dataset']}` conflict cases
  (`conflict_filter_mode={cfg['conflict_filter_mode']}`),
  subsampled to {cfg['subsample_size']} (seed {cfg['subsample_seed']}),
  with {cfg['val_size']} held out for validation.
- **Selected checkpoint step:** {best_step}
- **eval_rewards/accuracies:** {acc}
- **eval_loss:** {loss}
- **Date trained:** {today}

This repo contains the **LoRA adapter only** (merge into the base model for
inference). DPO hyperparameters follow Ferreira et al. (2025), Appendix B.2.
"""


def push_selected_adapter(cfg, best_step, metrics, hub_id: str, logger: logging.Logger):
    """Reload the selected adapter and push it (+tokenizer) to a private repo.

    Adapter-only: model.push_to_hub on a PeftModel uploads adapter weights and
    adapter_config, not a merged model. On any failure, saves locally to a
    clearly-named *_PUSH_FAILED dir and logs a loud ERROR with traceback.
    """
    output_dir = Path(cfg["output_dir"])
    checkpoint_dir = output_dir / f"checkpoint-{best_step}"
    if not checkpoint_dir.exists():
        raise RuntimeError(
            f"Selected checkpoint dir does not exist: {checkpoint_dir}. "
            f"Cannot push. Contents of {output_dir}: "
            f"{[p.name for p in output_dir.glob('checkpoint-*')]}"
        )

    logger.info("Reloading selected adapter from %s for push ...", checkpoint_dir)
    # Free the trainer's model/optimizer before reloading the base model.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        # Unsloth loads base + adapter from the saved checkpoint dir (its
        # adapter_config records base_model_name_or_path).
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(checkpoint_dir),
            max_seq_length=cfg["max_length"],
            dtype=None,
            load_in_4bit=False,
        )

        logger.info("Pushing adapter + tokenizer to %s (private) ...", hub_id)
        model.push_to_hub(hub_id, private=True)
        tokenizer.push_to_hub(hub_id, private=True)

        # Upload the auto-generated model card.
        api = HfApi()
        card = _build_model_card(cfg, best_step, metrics)
        api.upload_file(
            path_or_fileobj=card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=hub_id,
            repo_type="model",
        )

        # Log the resulting repo id + latest commit hash.
        info = api.repo_info(repo_id=hub_id, repo_type="model")
        commit = getattr(info, "sha", "unknown")
        logger.info(
            "PUSH OK -> repo_id=%s  commit=%s  (selected checkpoint-%d)",
            hub_id, commit, best_step,
        )
        return {"pushed": True, "repo_id": hub_id, "commit": commit}

    except Exception:  # noqa: BLE001  (we want to catch ANY push failure)
        fallback_dir = output_dir / f"best_checkpoint_{best_step}_PUSH_FAILED"
        logger.error("push_to_hub FAILED. Traceback:\n%s", traceback.format_exc())
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=str(checkpoint_dir),
                max_seq_length=cfg["max_length"],
                dtype=None,
                load_in_4bit=False,
            )
            model.save_pretrained(str(fallback_dir))
            tokenizer.save_pretrained(str(fallback_dir))
            logger.error(
                "ERROR: push_to_hub failed, saved adapter+tokenizer locally to "
                "%s — re-run the push manually once credentials/network are fixed.",
                fallback_dir,
            )
        except Exception:  # noqa: BLE001
            logger.error(
                "ERROR: local fallback save ALSO failed. The selected adapter "
                "still exists at %s — push it manually.\n%s",
                checkpoint_dir, traceback.format_exc(),
            )
        return {"pushed": False, "repo_id": hub_id, "fallback_dir": str(fallback_dir)}


# =============================================================================
# Orchestrator
# =============================================================================
def run_training(cfg: dict, hub_id: str, logger: logging.Logger) -> dict:
    """End-to-end: credentials -> model -> data -> inspect -> train -> select -> push.

    Returns a summary dict (used by sanity_check_dpo.py to assert on results):
      {best_step, metrics, log_history, prompt_length_stats, push_result,
       train_size, val_size}
    """
    check_credentials(logger)

    model, tokenizer = load_model_and_tokenizer(cfg, logger)

    dsd = load_and_prepare_dataset(cfg, tokenizer, logger)

    # Pre-training inspections (spec).
    inspect_formatting(dsd["train"], n=8, logger=logger)
    prompt_length_stats = check_prompt_lengths(
        dsd["train"], tokenizer, cfg["max_prompt_length"], logger
    )

    trainer = build_trainer(cfg, model, tokenizer, dsd["train"], dsd["validation"], logger)

    logger.info("Starting DPO training ...")
    trainer.train()
    logger.info("Training complete.")

    log_history = list(trainer.state.log_history)
    best_step, metrics = select_best_checkpoint(log_history, logger)

    push_result = None
    if best_step is not None:
        # Release the trainer (and its model) before reloading for the push.
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        push_result = push_selected_adapter(cfg, best_step, metrics, hub_id, logger)
    else:
        logger.error("No checkpoint selected (no eval points). Skipping push.")

    return {
        "best_step": best_step,
        "metrics": metrics,
        "log_history": log_history,
        "prompt_length_stats": prompt_length_stats,
        "push_result": push_result,
        "train_size": len(dsd["train"]),
        "val_size": len(dsd["validation"]),
    }


# =============================================================================
# Entry point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="RO2 MVP — DPO harmlessness training.")
    parser.add_argument(
        "--config",
        default="configs/harmlessness.yaml",
        help="Path to the training config YAML.",
    )
    args = parser.parse_args()

    logger = make_logger()
    cfg = load_config(args.config)
    logger.info("Loaded config from %s", args.config)

    summary = run_training(cfg, cfg["hub_model_id"], logger)

    logger.info(
        "DONE. best_step=%s  metrics=%s  push=%s",
        summary["best_step"], summary["metrics"],
        summary["push_result"],
    )


if __name__ == "__main__":
    main()
