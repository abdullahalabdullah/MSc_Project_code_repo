"""Dataset: PKU-SafeRLHF -> conflict cases -> safer (chosen) response ->
chat-templated, completion-only-masked, tokenized SFT examples.

Why this module does the masking itself (and does NOT rely on
`SFTConfig(dataset_text_field=...)`):

  The spec (§6) asserts that passing a pre-formatted `text` field to SFTTrainer
  masks the prompt tokens automatically. It does NOT — a plain text field trains
  the loss on every token (prompt + response). Completion-only training requires
  an explicit mechanism. We build `{input_ids, attention_mask, labels}` here with
  the prompt scaffold set to -100, and `verify_masking()` proves it.

How the boundary is found robustly:

  Qwen3 renders a full user+assistant turn (enable_thinking=False) as
      <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{chosen}<|im_end|>\n
  We split that rendered string at the (single) assistant header marker with
  `rpartition`, tokenize the scaffold and the response SEPARATELY, then
  concatenate the ids. Because input_ids are built by concatenation, the mask
  boundary is exact by construction — no fragile "is this a token prefix?" guess,
  and no dependence on the empty-`<think>` block that Qwen3 injects ONLY into the
  generation prompt (which is why we do not use add_generation_prompt=True here).
"""

import logging
from typing import Optional

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset

# PKU-Alignment/PKU-SafeRLHF schema (per the dataset card).
_REQUIRED_COLUMNS = [
    "prompt",
    "response_0",
    "response_1",
    "is_response_0_safe",
    "is_response_1_safe",
    "safer_response_id",
]

# Qwen3 / ChatML assistant turn marker. The assistant *response* begins
# immediately after this literal string in the rendered conversation.
ASSISTANT_HEADER = "<|im_start|>assistant\n"
# Qwen3 wraps the assistant turn in a thinking block. In non-thinking mode it
# emits an EMPTY one (<think>\n\n</think>). We must NOT supervise it (see below).
THINK_CLOSE = "</think>"


# =============================================================================
# Load + filter + select the safer response
# =============================================================================
def _is_conflict(example: dict) -> bool:
    """Conflict case (spec §3.2): exactly one of the two responses is safe."""
    return bool(example["is_response_0_safe"]) != bool(example["is_response_1_safe"])


def _extract_safer(example: dict) -> dict:
    """Keep the prompt and the SAFER response as `chosen` (spec §3.3)."""
    safer_id = int(example["safer_response_id"])  # 0 or 1
    return {"prompt": example["prompt"], "chosen": example[f"response_{safer_id}"]}


def load_pku_conflict(cfg: dict, logger: logging.Logger) -> Dataset:
    """Load PKU-SafeRLHF, keep conflict cases, return {prompt, chosen}."""
    logger.info("Loading dataset %s [split=%s] ...", cfg["dataset"], cfg["dataset_split"])
    ds = load_dataset(cfg["dataset"], split=cfg["dataset_split"])
    logger.info("Raw dataset size: %d  | columns: %s", len(ds), ds.column_names)

    missing = [c for c in _REQUIRED_COLUMNS if c not in ds.column_names]
    if missing:
        raise RuntimeError(
            f"PKU-SafeRLHF is missing expected columns {missing}. "
            f"Found: {ds.column_names}. Check the dataset card / revision."
        )

    filtered = ds.filter(_is_conflict, desc="Filtering conflict cases")
    expected = cfg.get("expected_conflict_size")
    logger.info(
        "Conflict cases (is_response_0_safe != is_response_1_safe): %d  (expected ~%s)",
        len(filtered), expected,
    )
    if expected and abs(len(filtered) - int(expected)) > 0.1 * int(expected):
        logger.warning(
            "Conflict count %d differs from expected ~%s by >10%%. Proceeding "
            "anyway (the dataset revision may have changed).",
            len(filtered), expected,
        )

    pairs = filtered.map(
        _extract_safer,
        remove_columns=filtered.column_names,
        desc="Selecting safer (chosen) response",
    )
    return pairs


# =============================================================================
# Formatting + completion-only label masking
# =============================================================================
def _render_full(example: dict, tokenizer, enable_thinking: bool) -> str:
    """Render the full user+assistant conversation via the chat template."""
    messages = [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["chosen"]},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )


def tokenize_and_mask(
    example: dict, tokenizer, max_length: int, enable_thinking: bool = False
) -> dict:
    """Tokenize one example with completion-only label masking.

    Returns {input_ids, attention_mask, labels}; the prompt scaffold (everything
    up to and including the assistant header) is labelled -100. `labels` is None
    only-internally never — we return -100 ids. Examples whose response is fully
    truncated away (no supervised token survives) get `n_supervised=0` so the
    caller can drop them.
    """
    full = _render_full(example, tokenizer, enable_thinking)

    head, sep, response = full.rpartition(ASSISTANT_HEADER)
    if not sep:
        raise RuntimeError(
            "Assistant header marker "
            f"{ASSISTANT_HEADER!r} not found in the rendered chat template. "
            "The Qwen3 template may have changed; update ASSISTANT_HEADER in "
            "sft/data.py to match the tokenizer's assistant turn opening."
        )
    scaffold = head + sep  # prompt + assistant header (masked)

    # CRITICAL: fold the assistant turn's <think>...</think> block into the MASKED
    # scaffold so it is never a training target. Qwen3 injects an empty
    # <think>\n\n</think> in non-thinking mode; supervising it teaches the model
    # to emit an empty think block and collapses its chain-of-thought (observed:
    # CoT disappears by epoch 1). Masking it leaves the model's reasoning
    # behaviour intact and supervises ONLY the safe answer that follows </think>.
    lstripped = response.lstrip()
    lead_ws = response[: len(response) - len(lstripped)]
    if lstripped.startswith("<think>") and THINK_CLOSE in lstripped:
        before, _, after = lstripped.partition(THINK_CLOSE)  # before='<think>\n\n'
        after_stripped = after.lstrip()                       # the actual answer
        scaffold = scaffold + lead_ws + before + THINK_CLOSE \
            + after[: len(after) - len(after_stripped)]
        response = after_stripped

    # Tokenize the two pieces separately and concatenate -> exact mask boundary.
    scaffold_ids = tokenizer(scaffold, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]

    input_ids = scaffold_ids + response_ids
    labels = [-100] * len(scaffold_ids) + list(response_ids)

    # Right-truncate to max_length (rare; the preflight sets max_length so <1%
    # of examples are affected). Truncation keeps input_ids and labels aligned.
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

    n_supervised = sum(1 for l in labels if l != -100)
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "n_supervised": n_supervised,
    }


def build_tokenized_dataset(
    pairs: Dataset,
    tokenizer,
    max_length: int,
    enable_thinking: bool,
    logger: logging.Logger,
) -> Dataset:
    """Map tokenize_and_mask over the dataset and drop fully-truncated examples."""
    tokenized = pairs.map(
        lambda ex: tokenize_and_mask(ex, tokenizer, max_length, enable_thinking),
        remove_columns=pairs.column_names,
        desc="Tokenizing + masking (completion-only)",
    )

    before = len(tokenized)
    tokenized = tokenized.filter(lambda ex: ex["n_supervised"] > 0)
    dropped = before - len(tokenized)
    if dropped:
        logger.warning(
            "Dropped %d/%d examples whose response was fully truncated at "
            "max_length=%d (no supervised token survived). Consider raising "
            "max_length (run scripts/00_check_seq_length.py).",
            dropped, before, max_length,
        )
    tokenized = tokenized.remove_columns(["n_supervised"])
    return tokenized


# =============================================================================
# Verification of the label masking (spec §6 + checklist)
# =============================================================================
def verify_masking(
    tokenized: Dataset, tokenizer, logger: logging.Logger, k: int = 3
) -> None:
    """Decode the masked vs supervised regions of a few examples and assert the
    split is correct: prompt scaffold masked, only the response supervised."""
    k = min(k, len(tokenized))
    logger.info("=" * 78)
    logger.info("LABEL-MASKING CHECK — first %d examples (spec §6):", k)
    logger.info("=" * 78)
    for i in range(k):
        ex = tokenized[i]
        ids = ex["input_ids"]
        labels = ex["labels"]
        masked_ids = [t for t, l in zip(ids, labels) if l == -100]
        kept_ids = [t for t, l in zip(ids, labels) if l != -100]

        masked_text = tokenizer.decode(masked_ids, skip_special_tokens=False)
        kept_text = tokenizer.decode(kept_ids, skip_special_tokens=False)

        logger.info("--- example %d  (%d tokens: %d masked / %d supervised) ---",
                    i, len(ids), len(masked_ids), len(kept_ids))
        logger.info("[MASKED prompt scaffold]\n%s", masked_text)
        logger.info("[SUPERVISED response]\n%s", kept_text)
        logger.info("-" * 40)

        # Hard assertions.
        assert len(kept_ids) > 0, f"example {i}: nothing supervised (all -100)."
        assert len(masked_ids) > 0, f"example {i}: nothing masked."
        assert "<|im_start|>user" in masked_text, (
            f"example {i}: user prompt not in the masked region."
        )
        assert "<|im_start|>assistant" in masked_text, (
            f"example {i}: assistant header not in the masked region."
        )
        assert "<|im_start|>user" not in kept_text, (
            f"example {i}: user prompt leaked into the supervised region."
        )
        # CRITICAL (CoT preservation): Qwen3's enable_thinking=False template
        # injects an empty <think>\n\n</think> into the assistant turn. It MUST be
        # masked — supervising it teaches the model to emit an empty think block
        # and collapses its chain-of-thought. If a future template revision changes
        # the think-block format and the fold in tokenize_and_mask() stops matching,
        # this assertion turns that silent regression into a hard failure.
        assert "<think>" not in kept_text and "</think>" not in kept_text, (
            f"example {i}: a <think> block leaked into the supervised region. "
            "The empty think scaffold is being trained on -> CoT will be "
            "suppressed. Check the fold in tokenize_and_mask() against the "
            "current Qwen3 chat template."
        )
    logger.info("Label-masking check PASSED for %d examples.", k)
    logger.info("=" * 78)


# =============================================================================
# Train / eval split
# =============================================================================
def split_train_eval(ds: Dataset, val_fraction: float, seed: int,
                     logger: logging.Logger) -> DatasetDict:
    """90/10 (configurable) train/eval split. val_fraction<=0 -> no eval set."""
    if val_fraction and val_fraction > 0:
        split = ds.train_test_split(test_size=val_fraction, seed=seed, shuffle=True)
        dsd = DatasetDict({"train": split["train"], "eval": split["test"]})
        logger.info("Split: train=%d, eval=%d (val_fraction=%.2f, seed=%d).",
                    len(dsd["train"]), len(dsd["eval"]), val_fraction, seed)
    else:
        dsd = DatasetDict({"train": ds})
        logger.info("No eval split (val_fraction<=0). train=%d.", len(ds))
    return dsd


# =============================================================================
# Sequence-length preflight (spec §4) — used by scripts/00_check_seq_length.py
# =============================================================================
def sequence_length_report(
    pairs: Dataset, tokenizer, enable_thinking: bool, logger: logging.Logger,
    caps=(512, 1024, 2048, 4096, 8192),
) -> dict:
    """Tokenize the full formatted dataset and report the length distribution
    and per-cap truncation %, then recommend the lowest cap with <1% truncation.
    """
    def _measure(ex):
        full = _render_full(ex, tokenizer, enable_thinking)
        return {"length": len(tokenizer(full, add_special_tokens=False)["input_ids"])}

    measured = pairs.map(_measure, desc="Measuring sequence lengths")
    lengths = np.array(measured["length"])

    logger.info("=" * 78)
    logger.info("SEQUENCE-LENGTH REPORT (full formatted examples, spec §4):")
    logger.info("  n        : %d", len(lengths))
    logger.info("  mean     : %.0f", lengths.mean())
    logger.info("  median   : %.0f", np.median(lengths))
    logger.info("  95th pct : %.0f", np.percentile(lengths, 95))
    logger.info("  99th pct : %.0f", np.percentile(lengths, 99))
    logger.info("  max      : %.0f", lengths.max())

    recommended = None
    per_cap = {}
    for cap in caps:
        n_over = int((lengths > cap).sum())
        pct = n_over / len(lengths) * 100.0
        per_cap[cap] = pct
        logger.info("  > %5d tokens : %5d examples (%.2f%%)", cap, n_over, pct)
        if recommended is None and pct < 1.0:
            recommended = cap

    logger.info("-" * 78)
    if recommended is not None:
        logger.info("RECOMMENDED max_length = %d  (lowest cap with <1%% truncation).",
                    recommended)
    else:
        logger.warning("No cap in %s keeps truncation <1%%. Increase the cap list.",
                       list(caps))
    logger.info("=" * 78)
    return {"lengths": lengths, "per_cap_pct": per_cap, "recommended": recommended}
