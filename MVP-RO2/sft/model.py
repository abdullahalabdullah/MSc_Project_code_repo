"""Model + tokenizer loading and the LoRA config.

Differences from the spec's illustrative snippet (corrected):
  - NO `device_map="auto"`. That path is for inference / model-parallel and
    conflicts with the Trainer's own device placement and optimizer. For
    single-GPU LoRA training we load on CPU and let the Trainer move the model
    to the GPU (accelerate handles multi-GPU).
  - `use_cache=False` is required for gradient checkpointing.
  - `attn_implementation` is configurable; default "sdpa" needs no extra install.
"""

import logging

import torch
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tokenizer(cfg: dict):
    """Load the Qwen3 tokenizer; right padding is required for causal-LM SFT."""
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(cfg: dict, logger: logging.Logger):
    """Load the base model in bf16 with the configured attention implementation."""
    attn = cfg.get("attn_implementation", "sdpa")
    logger.info("Loading base model %s (bf16, attn_implementation=%s) ...",
                cfg["base_model"], attn)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        torch_dtype=torch.bfloat16,
        attn_implementation=attn,
    )
    # Required for gradient checkpointing; the Trainer re-enables cache for eval.
    model.config.use_cache = False
    return model


def build_lora_config(cfg: dict) -> LoraConfig:
    """PEFT LoRA config (spec §5)."""
    return LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=list(cfg["target_modules"]),
    )
