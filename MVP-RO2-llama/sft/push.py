"""Push the per-epoch LoRA adapters to private HF Hub repos (adapter-only).

Each checkpoint dir already contains only the adapter (save_only_model=True), so
we upload the adapter files + config + tokenizer + an auto model card — no model
loading or merging. Merging into the base model happens later at eval time.
"""

import datetime as _dt
import glob
import logging
import os
import traceback
from typing import Optional

from huggingface_hub import HfApi

from .config import epoch_repo_id

# Only adapter / config / tokenizer / card files — never optimizer or rng state.
_ALLOW_PATTERNS = [
    "adapter_*",          # adapter_model.safetensors, adapter_config.json
    "*.json",             # config.json, tokenizer_config.json, generation_config.json
    "tokenizer*",         # tokenizer.json, tokenizer.model
    "vocab*", "merges*", "*.txt",
    "special_tokens_map.json",
    "*.jinja",            # chat_template.jinja (newer transformers saves it separately)
    "README.md",
]


def discover_checkpoints(output_dir: str) -> list:
    """Return checkpoint dirs sorted by step (ascending = epoch order)."""
    return sorted(
        glob.glob(os.path.join(output_dir, "checkpoint-*")),
        key=lambda p: int(p.split("-")[-1]),
    )


def _model_card(cfg: dict, epoch_idx: int, step: int) -> str:
    today = _dt.date.today().isoformat()
    return f"""---
base_model: {cfg['base_model']}
library_name: peft
tags:
- sft
- lora
- harmlessness
- llama
---

# Llama-3.2-3B-Instruct harmlessness SFT (RO2 MVP) — epoch {epoch_idx}

LoRA adapter from **supervised fine-tuning** on the **chosen (more-harmless)
responses** of Anthropic/hh-rlhf `harmless-base` *single-turn* examples
(harmlessness objective). This is the SFT ablation for RO2; it trains only on the
preferred response, with no contrastive pressure (cf. the planned DPO run).

- **Base model:** `{cfg['base_model']}`
- **Training data:** `{cfg['dataset']}` (`{cfg.get('dataset_subset', 'harmless-base')}`,
  single-turn examples), target = the `chosen` assistant reply.
- **Checkpoint:** epoch {epoch_idx} (step {step}).
- **Date trained:** {today}

This repo contains the **LoRA adapter only**. To use:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("{cfg['base_model']}", torch_dtype="bfloat16")
model = PeftModel.from_pretrained(base, "<this-repo-id>")
tok = AutoTokenizer.from_pretrained("{cfg['base_model']}")
```
"""


def push_checkpoint(ckpt_dir: str, repo_id: str, cfg: dict, epoch_idx: int,
                    logger: logging.Logger) -> dict:
    """Push one checkpoint dir to `repo_id` (private), with a model card.

    On failure, logs a loud ERROR with traceback and returns pushed=False (the
    local checkpoint is untouched and can be pushed manually).
    """
    step = int(ckpt_dir.split("-")[-1])
    api = HfApi()
    try:
        api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
        logger.info("Pushing epoch %d (%s) -> %s (private) ...", epoch_idx, ckpt_dir, repo_id)
        api.upload_folder(
            folder_path=ckpt_dir,
            repo_id=repo_id,
            repo_type="model",
            allow_patterns=_ALLOW_PATTERNS,
        )
        api.upload_file(
            path_or_fileobj=_model_card(cfg, epoch_idx, step).encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
        )
        info = api.repo_info(repo_id=repo_id, repo_type="model")
        commit = getattr(info, "sha", "unknown")
        logger.info("PUSH OK -> %s (commit=%s, step=%d)", repo_id, commit, step)
        return {"pushed": True, "repo_id": repo_id, "commit": commit, "step": step}
    except Exception:  # noqa: BLE001 — report ANY push failure, keep going
        logger.error("push to %s FAILED. Traceback:\n%s", repo_id, traceback.format_exc())
        return {"pushed": False, "repo_id": repo_id, "step": step}


def push_all_checkpoints(cfg: dict, logger: logging.Logger,
                         tokenizer_dir: Optional[str] = None) -> list:
    """Discover all epoch checkpoints under output_dir and push each one.

    If `tokenizer_dir` is given, its tokenizer files are copied into every
    checkpoint dir first so each repo is self-contained at eval time.
    """
    output_dir = cfg["output_dir"]
    checkpoints = discover_checkpoints(output_dir)
    if not checkpoints:
        raise RuntimeError(f"No checkpoint-* dirs under {output_dir}. Train first.")
    logger.info("Found %d checkpoint(s): %s", len(checkpoints), checkpoints)

    if tokenizer_dir:
        _copy_tokenizer_into_checkpoints(tokenizer_dir, checkpoints, logger)

    results = []
    for epoch_idx, ckpt_dir in enumerate(checkpoints, start=1):
        repo_id = epoch_repo_id(cfg, epoch_idx)
        results.append(push_checkpoint(ckpt_dir, repo_id, cfg, epoch_idx, logger))

    n_ok = sum(1 for r in results if r["pushed"])
    logger.info("Pushed %d/%d adapter checkpoints.", n_ok, len(results))
    for r in results:
        logger.info("  epoch %s: %s", r.get("step"), r)
    return results


def _copy_tokenizer_into_checkpoints(tokenizer_dir, checkpoints, logger):
    """Copy tokenizer files from `tokenizer_dir` into each checkpoint dir."""
    import shutil

    tok_files = [
        f for f in os.listdir(tokenizer_dir)
        if f.startswith("tokenizer") or f in (
            "vocab.json", "merges.txt", "special_tokens_map.json",
            "added_tokens.json", "chat_template.jinja",
        )
    ]
    for ckpt in checkpoints:
        for f in tok_files:
            src = os.path.join(tokenizer_dir, f)
            dst = os.path.join(ckpt, f)
            if os.path.isfile(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
    logger.info("Copied tokenizer files %s into each checkpoint dir.", tok_files)
