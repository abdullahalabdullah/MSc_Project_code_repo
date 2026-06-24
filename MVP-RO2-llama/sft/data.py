"""Dataset: Anthropic/hh-rlhf (harmless-base) -> single-turn examples ->
chosen (more-harmless) response -> chat-templated, completion-only-masked,
tokenized SFT examples.

Why this module does the masking itself (and does NOT rely on
`SFTConfig(dataset_text_field=...)`):

  Passing a pre-formatted `text` field to SFTTrainer trains the loss on every
  token (prompt + response). Completion-only training requires an explicit
  mechanism. We build `{input_ids, attention_mask, labels}` here with the prompt
  scaffold set to -100, and `verify_masking()` proves it.

How the boundary is found robustly:

  Llama-3 renders a full user+assistant turn as
      ...<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>
      <|start_header_id|>assistant<|end_header_id|>\n\n{chosen}<|eot_id|>
  We split that rendered string at the (single) assistant header marker with
  `rpartition`, tokenize the scaffold and the response SEPARATELY, then
  concatenate the ids. Because input_ids are built by concatenation, the mask
  boundary is exact by construction — no fragile "is this a token prefix?" guess.
  The supervised response keeps its trailing <|eot_id|> so the model learns to
  stop.

How single-turn examples are selected:

  hh-rlhf stores a full conversation in the `chosen`/`rejected` strings, formatted
  as repeated "\\n\\nHuman:" / "\\n\\nAssistant:" turns. We keep only "single-turn"
  examples (exactly one Human prompt, i.e. one "\\n\\nHuman:" marker in `chosen`),
  matching the project's hh_rlhf_stats.py definition (~12,822 in harmless-base
  train). The training target is the assistant reply from the `chosen` transcript
  (the more-harmless of the two responses).
"""

import logging

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset

# Anthropic/hh-rlhf schema (per the dataset card): each row is a pair of full
# conversation transcripts; `chosen` is the more-harmless one.
_REQUIRED_COLUMNS = ["chosen", "rejected"]

# Conversation turn markers used by hh-rlhf transcripts.
_HUMAN_MARKER = "\n\nHuman:"
_ASSISTANT_MARKER = "\n\nAssistant:"

# Llama-3 assistant turn marker. The assistant *response* begins immediately
# after this literal string in the rendered conversation.
ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"


# =============================================================================
# Load + filter single-turn + select the chosen response
# =============================================================================
def _is_single_turn(example: dict) -> bool:
    """Single-turn (project definition): exactly one Human prompt in `chosen`."""
    return example["chosen"].count(_HUMAN_MARKER) == 1


def _parse_chosen(example: dict) -> dict:
    """Extract {prompt, chosen} from the single-turn `chosen` transcript.

    The transcript is "\\n\\nHuman: {prompt}\\n\\nAssistant: {reply}". We split off
    everything after the (single) Human marker, then partition on the Assistant
    marker to separate the user prompt from the assistant reply.
    """
    after_human = example["chosen"].split(_HUMAN_MARKER, 1)[1]
    prompt, _, reply = after_human.partition(_ASSISTANT_MARKER)
    return {"prompt": prompt.strip(), "chosen": reply.strip()}


def load_hh_harmless(cfg: dict, logger: logging.Logger) -> Dataset:
    """Load hh-rlhf harmless-base, keep single-turn examples, return {prompt, chosen}."""
    subset = cfg.get("dataset_subset", "harmless-base")
    logger.info("Loading dataset %s [data_dir=%s, split=%s] ...",
                cfg["dataset"], subset, cfg["dataset_split"])
    ds = load_dataset(cfg["dataset"], data_dir=subset, split=cfg["dataset_split"])
    logger.info("Raw dataset size: %d  | columns: %s", len(ds), ds.column_names)

    missing = [c for c in _REQUIRED_COLUMNS if c not in ds.column_names]
    if missing:
        raise RuntimeError(
            f"hh-rlhf is missing expected columns {missing}. "
            f"Found: {ds.column_names}. Check the dataset card / revision."
        )

    filtered = ds.filter(_is_single_turn, desc="Filtering single-turn examples")
    expected = cfg.get("expected_single_turn_size")
    logger.info(
        "Single-turn examples (one '\\n\\nHuman:' in `chosen`): %d  (expected ~%s)",
        len(filtered), expected,
    )
    if expected and abs(len(filtered) - int(expected)) > 0.1 * int(expected):
        logger.warning(
            "Single-turn count %d differs from expected ~%s by >10%%. Proceeding "
            "anyway (the dataset revision may have changed).",
            len(filtered), expected,
        )

    pairs = filtered.map(
        _parse_chosen,
        remove_columns=filtered.column_names,
        desc="Parsing prompt + chosen reply",
    )
    return pairs


# =============================================================================
# Formatting + completion-only label masking
# =============================================================================
def _render_full(example: dict, tokenizer) -> str:
    """Render the full user+assistant conversation via the chat template."""
    messages = [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["chosen"]},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def tokenize_and_mask(example: dict, tokenizer, max_length: int) -> dict:
    """Tokenize one example with completion-only label masking.

    Returns {input_ids, attention_mask, labels}; the prompt scaffold (everything
    up to and including the assistant header) is labelled -100. Only the assistant
    response (plus its trailing <|eot_id|>) is supervised. Examples whose response
    is fully truncated away (no supervised token survives) get `n_supervised=0` so
    the caller can drop them.
    """
    full = _render_full(example, tokenizer)

    head, sep, response = full.rpartition(ASSISTANT_HEADER)
    if not sep:
        raise RuntimeError(
            "Assistant header marker "
            f"{ASSISTANT_HEADER!r} not found in the rendered chat template. "
            "The Llama-3 template may have changed; update ASSISTANT_HEADER in "
            "sft/data.py to match the tokenizer's assistant turn opening."
        )
    scaffold = head + sep  # prompt + assistant header (masked)

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
    logger: logging.Logger,
) -> Dataset:
    """Map tokenize_and_mask over the dataset and drop fully-truncated examples."""
    tokenized = pairs.map(
        lambda ex: tokenize_and_mask(ex, tokenizer, max_length),
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
# Verification of the label masking
# =============================================================================
def verify_masking(
    tokenized: Dataset, tokenizer, logger: logging.Logger, k: int = 3
) -> None:
    """Decode the masked vs supervised regions of a few examples and assert the
    split is correct: prompt scaffold masked, only the response supervised."""
    k = min(k, len(tokenized))
    logger.info("=" * 78)
    logger.info("LABEL-MASKING CHECK — first %d examples:", k)
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
        assert "<|start_header_id|>user<|end_header_id|>" in masked_text, (
            f"example {i}: user prompt header not in the masked region."
        )
        assert "<|start_header_id|>assistant<|end_header_id|>" in masked_text, (
            f"example {i}: assistant header not in the masked region."
        )
        assert "<|start_header_id|>user<|end_header_id|>" not in kept_text, (
            f"example {i}: user prompt leaked into the supervised region."
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
# Sequence-length preflight — used by scripts/00_check_seq_length.py
# =============================================================================
def sequence_length_report(
    pairs: Dataset, tokenizer, logger: logging.Logger,
    caps=(512, 1024, 2048, 4096, 8192),
) -> dict:
    """Tokenize the full formatted dataset and report the length distribution
    and per-cap truncation %, then recommend the lowest cap with <1% truncation.
    """
    def _measure(ex):
        full = _render_full(ex, tokenizer)
        return {"length": len(tokenizer(full, add_special_tokens=False)["input_ids"])}

    measured = pairs.map(_measure, desc="Measuring sequence lengths")
    lengths = np.array(measured["length"])

    logger.info("=" * 78)
    logger.info("SEQUENCE-LENGTH REPORT (full formatted examples):")
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
