"""Merge a LoRA adapter into the base model and push the standalone merged model.

The MVP-RO1 eval pipeline loads models with vLLM **by hf_id** and cannot consume
an adapter-only repo directly. So for evaluation we merge the LoRA weights into
Llama-3.2-3B-Instruct and push a full, standalone 16-bit model that drops into
MVP-RO1's `models:` registry.
"""

import logging
import os

import torch
from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_adapter(base_model_id: str, adapter: str, out_dir: str,
                  logger: logging.Logger, dtype=torch.bfloat16) -> str:
    """Load base + adapter, merge, and save the standalone model + tokenizer.

    `adapter` may be a local checkpoint dir or a Hub adapter repo id.
    """
    logger.info("Loading base model %s (%s) ...", base_model_id, dtype)
    base = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=dtype)

    logger.info("Attaching adapter %s ...", adapter)
    model = PeftModel.from_pretrained(base, adapter)

    logger.info("Merging adapter into base weights (merge_and_unload) ...")
    merged = model.merge_and_unload()

    os.makedirs(out_dir, exist_ok=True)
    merged.save_pretrained(out_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(base_model_id).save_pretrained(out_dir)
    logger.info("Merged model + tokenizer saved to %s", out_dir)
    return out_dir


def push_merged(out_dir: str, repo_id: str, logger: logging.Logger,
                private: bool = True) -> str:
    """Upload the merged model dir to a (private) Hub repo for the eval pipeline."""
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    logger.info("Uploading merged model %s -> %s (private=%s) ...", out_dir, repo_id, private)
    api.upload_folder(folder_path=out_dir, repo_id=repo_id, repo_type="model")
    info = api.repo_info(repo_id=repo_id, repo_type="model")
    commit = getattr(info, "sha", "unknown")
    logger.info("PUSH OK -> %s (commit=%s)", repo_id, commit)
    return repo_id
