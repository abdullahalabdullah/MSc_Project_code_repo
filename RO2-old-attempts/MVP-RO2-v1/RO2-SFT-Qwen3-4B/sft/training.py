"""Training orchestration: build the SFTConfig + SFTTrainer and run the SFT.

API notes (verified against trl 0.22.2 / transformers 4.56.2 / peft 0.18.1):
  - `SFTConfig.max_length` (NOT the spec's `max_seq_length`, which was removed).
  - `SFTTrainer(..., processing_class=tokenizer)` (NOT `tokenizer=`).
  - We feed an ALREADY-tokenized dataset ({input_ids, attention_mask, labels})
    and set `dataset_kwargs={"skip_prepare_dataset": True}` so TRL does not
    re-tokenize or strip our labels; a `DataCollatorForSeq2Seq` pads dynamically
    and pads `labels` with -100.
  - LoRA is applied here (get_peft_model) so we can call
    `enable_input_require_grads()` — needed for grads to flow with gradient
    checkpointing — and we pass `peft_config=None` to the trainer.
"""

import glob
import logging
import os

from peft import get_peft_model
from transformers import DataCollatorForSeq2Seq
from trl import SFTConfig, SFTTrainer

from . import data as data_mod
from .credentials import check_credentials
from .model import build_lora_config, load_model, load_tokenizer


def build_sft_config(cfg: dict) -> SFTConfig:
    """Assemble the SFTConfig from the YAML config (spec §5, corrected)."""
    return SFTConfig(
        output_dir=cfg["output_dir"],

        # Duration / checkpointing
        num_train_epochs=cfg["num_train_epochs"],
        max_steps=int(cfg.get("max_steps", -1)),
        save_strategy=cfg.get("save_strategy", "epoch"),
        save_steps=cfg.get("save_steps", 500),
        eval_strategy=cfg.get("eval_strategy", "no"),
        eval_steps=cfg.get("eval_steps", 500),
        save_total_limit=cfg.get("save_total_limit", None),
        save_only_model=True,            # clean adapter-only checkpoints (no optim state)
        load_best_model_at_end=False,    # keep all epoch checkpoints

        # Batch size
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 2),

        # Optimiser / schedule
        optim=cfg.get("optim", "adamw_torch_fused"),
        learning_rate=float(cfg["learning_rate"]),
        weight_decay=cfg["weight_decay"],
        max_grad_norm=cfg["max_grad_norm"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_ratio=cfg["warmup_ratio"],

        # Precision / memory
        bf16=cfg.get("bf16", True),
        fp16=False,
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # Sequence length (record; truncation already applied in tokenize_and_mask)
        max_length=cfg["max_length"],
        packing=False,                   # avoid cross-example contamination

        # We pre-tokenized + masked ourselves -> skip TRL's dataset processing.
        dataset_kwargs={"skip_prepare_dataset": True},
        label_names=["labels"],

        # Logging
        logging_steps=cfg["logging_steps"],
        report_to="none",
        seed=cfg.get("seed", 42),
    )


def _prepare_data(cfg, tokenizer, logger):
    """PKU -> conflict -> safer -> (optional subsample) -> tokenize+mask -> verify -> split."""
    pairs = data_mod.load_pku_conflict(cfg, logger)

    max_train_samples = cfg.get("max_train_samples")
    if max_train_samples:
        n = min(int(max_train_samples), len(pairs))
        pairs = pairs.shuffle(seed=cfg.get("split_seed", 42)).select(range(n))
        logger.info("Subsampled to %d examples (max_train_samples).", n)

    tokenized = data_mod.build_tokenized_dataset(
        pairs, tokenizer,
        max_length=cfg["max_length"],
        enable_thinking=cfg.get("enable_thinking_train", False),
        logger=logger,
    )
    data_mod.verify_masking(tokenized, tokenizer, logger, k=3)

    dsd = data_mod.split_train_eval(
        tokenized, cfg.get("val_fraction", 0.0), cfg.get("split_seed", 42), logger
    )
    return dsd


def run_training(cfg: dict, logger: logging.Logger) -> dict:
    """End-to-end SFT: credentials -> model/tokenizer -> data -> train -> save.

    Returns a summary dict consumed by the sanity gate:
      {output_dir, checkpoint_dirs, log_history, train_size, eval_size}.
    """
    check_credentials(logger)

    tokenizer = load_tokenizer(cfg)
    base_model = load_model(cfg, logger)

    # Apply LoRA here so we can enable input grads for gradient checkpointing.
    lora_config = build_lora_config(cfg)
    model = get_peft_model(base_model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    dsd = _prepare_data(cfg, tokenizer, logger)

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, padding=True, label_pad_token_id=-100, pad_to_multiple_of=8
    )

    args = build_sft_config(cfg)
    logger.info(
        "Effective batch size = %d (%d x %d). Epochs=%s, max_steps=%s, max_length=%d.",
        cfg["per_device_train_batch_size"] * cfg["gradient_accumulation_steps"],
        cfg["per_device_train_batch_size"], cfg["gradient_accumulation_steps"],
        args.num_train_epochs, args.max_steps, args.max_length,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dsd["train"],
        eval_dataset=dsd.get("eval"),
        processing_class=tokenizer,
        data_collator=collator,
        peft_config=None,                # model is already a PeftModel
    )

    logger.info("Starting SFT training ...")
    trainer.train()
    logger.info("Training complete.")

    # Save the tokenizer alongside the checkpoints so each pushed repo is
    # self-contained at eval time.
    tokenizer.save_pretrained(cfg["output_dir"])

    checkpoint_dirs = sorted(
        glob.glob(os.path.join(cfg["output_dir"], "checkpoint-*")),
        key=lambda p: int(p.split("-")[-1]),
    )
    logger.info("Checkpoints written: %s", checkpoint_dirs)

    return {
        "output_dir": cfg["output_dir"],
        "checkpoint_dirs": checkpoint_dirs,
        "log_history": list(trainer.state.log_history),
        "train_size": len(dsd["train"]),
        "eval_size": len(dsd["eval"]) if "eval" in dsd else 0,
    }
