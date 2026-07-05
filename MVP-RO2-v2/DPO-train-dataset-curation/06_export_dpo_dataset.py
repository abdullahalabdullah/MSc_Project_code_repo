"""Stage 06 — export the final preference pairs as a training-ready DPO dataset.

This is the handoff to the (external) DPO trainer. It converts the module's
`final_pairs.joblib` (a list[dict] with rich metadata) into the standard TRL
preference schema and saves it as a HF DatasetDict:

    DatasetDict({
        "train": Dataset({prompt, chosen, rejected}),
        "valid": Dataset({prompt, chosen, rejected}),
    })
    -> saved via save_to_disk(config.DPO_DATASET_DIR)

- `prompt`   : the harmful WildJailbreak prompt (the trainer applies its own chat template).
- `chosen`   : the safest response; `rejected`: the most harmful response.
- By default `chosen`/`rejected` are the FULL responses (<think>…</think> + answer), matching
  what the model generates (config.EXPORT_FIELD="full"). Set EXPORT_FIELD="answer" to export
  only the visible post-</think> answer.

The split is deterministic (config.SEED) with config.VALID_FRACTION held out as "valid".

Runs on CPU. Input: config.FINAL_PAIRS_PATH. Output: config.DPO_DATASET_DIR (+ diagnostics).
"""
import random

import joblib
from datasets import Dataset, DatasetDict

import config
from utils import set_seed, update_diagnostics


def _pick_fields(pair):
    """Map one final_pairs dict to the {prompt, chosen, rejected} training row."""
    if config.EXPORT_FIELD == "answer":
        chosen, rejected = pair["chosen_answer"], pair["rejected_answer"]
    else:  # "full" (default)
        chosen, rejected = pair["chosen"], pair["rejected"]
    return {"prompt": pair["prompt"], "chosen": chosen, "rejected": rejected}


def _split_train_valid(rows, valid_fraction, seed):
    """Deterministic train/valid split.

    Guarantees a non-empty train split; holds out at least one valid example when there is
    more than one row (so downstream eval always has something), and never lets valid take
    the whole set.
    """
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    n = len(rows)
    if n <= 1:
        return rows, []                       # too few to split — everything trains
    n_valid = round(n * valid_fraction)
    n_valid = max(1, min(n_valid, n - 1))     # 1 <= n_valid <= n-1
    return rows[n_valid:], rows[:n_valid]


def main():
    config.ensure_output_dir()
    set_seed(config.SEED)

    if config.EXPORT_FIELD not in ("full", "answer"):
        raise ValueError(f"EXPORT_FIELD must be 'full' or 'answer', got {config.EXPORT_FIELD!r}")

    final_pairs = joblib.load(config.FINAL_PAIRS_PATH)
    print(f"Loaded {len(final_pairs)} final pairs from {config.FINAL_PAIRS_PATH}")
    if not final_pairs:
        raise SystemExit("No final pairs to export — run stages 01-05 first (final_pairs is empty).")

    rows = [_pick_fields(p) for p in final_pairs]
    train_rows, valid_rows = _split_train_valid(rows, config.VALID_FRACTION, config.SEED)

    # train is always non-empty (split guarantees it); reuse its schema for valid so an
    # empty valid split (tiny datasets / smoke runs) still saves instead of failing schema inference.
    train_ds = Dataset.from_list(train_rows)
    if valid_rows:
        valid_ds = Dataset.from_list(valid_rows)
    else:
        valid_ds = Dataset.from_dict(
            {c: [] for c in train_ds.column_names}, features=train_ds.features
        )
    dataset = DatasetDict({"train": train_ds, "valid": valid_ds})
    dataset.save_to_disk(config.DPO_DATASET_DIR)

    update_diagnostics(config.DIAGNOSTICS_PATH, "export", {
        "n_pairs": len(final_pairs),
        "n_train": len(train_rows),
        "n_valid": len(valid_rows),
        "valid_fraction": config.VALID_FRACTION,
        "export_field": config.EXPORT_FIELD,
        "columns": ["prompt", "chosen", "rejected"],
        "path": config.DPO_DATASET_DIR,
    })

    print(f"Exported DPO dataset -> {config.DPO_DATASET_DIR}")
    print(f"  splits: train={len(train_rows)}  valid={len(valid_rows)}  "
          f"(field={config.EXPORT_FIELD})")
    print("  columns: prompt, chosen, rejected")
    print(f"  load with: datasets.load_from_disk({config.DPO_DATASET_DIR!r})")


if __name__ == "__main__":
    main()
