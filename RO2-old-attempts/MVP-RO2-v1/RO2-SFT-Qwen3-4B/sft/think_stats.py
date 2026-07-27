"""Average chain-of-thought (think) token count over the held-out eval set.

This gives a *quantitative* CoT measure to complement the per-prompt sanity
check (sft/generate.py). The intended workflow:

  1) BEFORE/independent of training, run the BASE model -> store a baseline
     (outputs/think_stats/base.json): "the un-fine-tuned model thinks for ~N
     tokens on average over these eval prompts".
  2) After training, run the SAME counting on any checkpoint. It auto-loads the
     base baseline and reports what fraction of the base CoT length survived, so
     suppression shows up as a number, not a vibe.

The eval set is the SAME held-out split the trainer uses (seed + val_fraction
from the config), so counts are comparable across models. We split the
{prompt, chosen} pairs (not the tokenized set the trainer splits); the two
differ only by the <1% of examples dropped for full truncation, which is
immaterial for a held-out CoT baseline.
"""

import json
import logging
import os
import time

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from .data import load_pku_conflict, split_train_eval
from .generate import count_think_tokens

DEFAULT_STATS_DIR = "outputs/think_stats"


def label_for(adapter_dir) -> str:
    """Stable file/label name: 'base' for no adapter, else the dir/repo basename."""
    if adapter_dir is None or str(adapter_dir).lower() in {"base", "none", ""}:
        return "base"
    return os.path.basename(str(adapter_dir).rstrip("/")) or "model"


def build_eval_prompts(cfg: dict, logger: logging.Logger, max_samples=None):
    """Reconstruct the held-out eval prompts (same seed/val_fraction as training)."""
    pairs = load_pku_conflict(cfg, logger)
    dsd = split_train_eval(
        pairs, cfg.get("val_fraction", 0.10), cfg.get("split_seed", 42), logger
    )
    if "eval" not in dsd:
        raise RuntimeError("val_fraction<=0 -> no eval split; set val_fraction>0.")
    prompts = list(dsd["eval"]["prompt"])

    if max_samples and max_samples < len(prompts):
        rng = np.random.default_rng(cfg.get("split_seed", 42))
        idx = sorted(rng.choice(len(prompts), size=max_samples, replace=False).tolist())
        prompts = [prompts[i] for i in idx]
        logger.info("Subsampled eval prompts to %d (max_samples).", len(prompts))
    logger.info("Eval prompts for think-token counting: %d", len(prompts))
    return prompts


def _load_model(base_model_id: str, adapter_dir, logger: logging.Logger):
    base_only = adapter_dir is None or str(adapter_dir).lower() in {"base", "none", ""}
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if base_only:
        logger.info("Loaded BASE model %s (no adapter) for the CoT baseline.", base_model_id)
        model = base
    else:
        logger.info("Loaded base %s + adapter %s.", base_model_id, adapter_dir)
        model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model


def think_token_stats_on_eval(
    base_model_id: str,
    adapter_dir,
    tokenizer,
    prompts,
    logger: logging.Logger,
    max_new_tokens: int = 512,
    do_sample: bool = False,
    temperature: float = 0.6,
    top_p: float = 0.95,
    batch_size: int = 8,
    log_every: int = 5,
) -> dict:
    """Generate (thinking mode) for each eval prompt and aggregate think-token counts.

    Greedy by default for reproducible, model-to-model comparable counts; pass
    do_sample=True to mirror the RO1 eval decoding.
    """
    model = _load_model(base_model_id, adapter_dir, logger)

    # Left-pad for batched generation so the response slice is aligned per row.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
    if do_sample:
        gen_kwargs.update(temperature=temperature, top_p=top_p)

    counts, has_think_flags = [], []
    n_batches = (len(prompts) + batch_size - 1) // batch_size
    for b in range(n_batches):
        batch = prompts[b * batch_size:(b + 1) * batch_size]
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True, enable_thinking=True,
            )
            for p in batch
        ]
        enc = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs)
        gen = out[:, enc["input_ids"].shape[1]:]
        for row in gen:
            decoded = tokenizer.decode(row, skip_special_tokens=False)
            has_think, n = count_think_tokens(decoded, tokenizer)
            counts.append(n)
            has_think_flags.append(has_think)
        if (b + 1) % log_every == 0 or b + 1 == n_batches:
            logger.info("  [%d/%d batches] running mean think tokens = %.1f",
                        b + 1, n_batches, float(np.mean(counts)))

    arr = np.array(counts, dtype=float)
    label = label_for(adapter_dir)
    return {
        "label": label,
        "base_model": base_model_id,
        "adapter": None if label == "base" else str(adapter_dir),
        "n": int(len(arr)),
        "mean_think_tokens": float(arr.mean()),
        "median_think_tokens": float(np.median(arr)),
        "std_think_tokens": float(arr.std()),
        "min_think_tokens": int(arr.min()),
        "max_think_tokens": int(arr.max()),
        "frac_nonempty_think": float(np.mean(arr > 0)),
        "frac_has_think_tag": float(np.mean(has_think_flags)),
        "generation": {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature if do_sample else None,
            "top_p": top_p if do_sample else None,
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "per_example_think_tokens": [int(c) for c in counts],
    }


def save_stats(stats: dict, out_dir: str = DEFAULT_STATS_DIR) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{stats['label']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return path


def load_stats(label: str, out_dir: str = DEFAULT_STATS_DIR):
    path = os.path.join(out_dir, f"{label}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def log_summary(stats: dict, logger: logging.Logger, baseline=None) -> None:
    logger.info("=" * 70)
    logger.info("THINK-TOKEN STATS — %s  (n=%d, %s)",
                stats["label"], stats["n"],
                "sampled" if stats["generation"]["do_sample"] else "greedy")
    logger.info("  mean think tokens    : %.1f", stats["mean_think_tokens"])
    logger.info("  median               : %.1f", stats["median_think_tokens"])
    logger.info("  std / min / max      : %.1f / %d / %d",
                stats["std_think_tokens"], stats["min_think_tokens"],
                stats["max_think_tokens"])
    logger.info("  frac non-empty think : %.3f", stats["frac_nonempty_think"])
    if baseline is not None and baseline.get("mean_think_tokens", 0) > 0:
        retained = stats["mean_think_tokens"] / baseline["mean_think_tokens"]
        logger.info("-" * 70)
        logger.info("  base mean=%.1f -> this mean=%.1f  (%.1f%% of base CoT retained)",
                    baseline["mean_think_tokens"], stats["mean_think_tokens"],
                    retained * 100.0)
        if retained < 0.5:
            logger.warning("  CoT mean dropped >50%% vs base -> likely suppression.")
    logger.info("=" * 70)
