"""Export (12) — training-ready DPO DatasetDict (CPU).

Identical handoff contract to the harmlessness module's stage 06: a HF DatasetDict
({"train","valid"}) with exactly three string columns (prompt / chosen / rejected),
saved via save_to_disk, loaded by train_dpo.py with load_from_disk — the trainer needs
NO changes. chosen/rejected are the verbatim full completions (incl. <think>…</think>),
matching what the model generates (the harmlessness EXPORT_FIELD="full" convention).

The split is deterministic (SPLIT_SEED) and stratified by direction (best-effort 50/50
in the holdout, mirroring the main file's balance).

Input : pairs/anti_syc_dpo.jsonl
Output: dpo_dataset/
"""
import argparse
import os
import random

import config
import env_setup
from utils import (check_stage_cache, read_jsonl, update_diagnostics, upstream_hash,
                   write_stage_marker)

STAGE = "12_export_dpo_dataset"


def _split_ids(pairs, valid_fraction, seed):
    """Deterministic pair_id -> split assignment, stratified by direction."""
    n = len(pairs)
    if n <= 1:
        return set()
    n_valid = max(1, min(round(n * valid_fraction), n - 1))
    rng = random.Random(seed)
    valid_ids = set()
    by_dir = {}
    for p in pairs:
        by_dir.setdefault(p["direction"], []).append(p["pair_id"])
    dirs = sorted(by_dir)
    # Half the holdout from each direction (mirrors the 50/50 main file); any
    # shortfall (odd counts / tiny runs) fills from the remaining pool.
    per_dir = n_valid // len(dirs) if dirs else 0
    for d in dirs:
        ids = sorted(by_dir[d])
        rng.shuffle(ids)
        valid_ids.update(ids[:min(per_dir, len(ids))])
    if len(valid_ids) < n_valid:
        rest = sorted(set(p["pair_id"] for p in pairs) - valid_ids)
        rng.shuffle(rest)
        valid_ids.update(rest[:n_valid - len(valid_ids)])
    return valid_ids


def _export(pairs, valid_ids, out_dir):
    from datasets import Dataset, DatasetDict
    rows = {"train": [], "valid": []}
    for p in pairs:
        split = "valid" if p["pair_id"] in valid_ids else "train"
        rows[split].append({
            "prompt": p["prompt"],
            "chosen": p["chosen"]["full_text"],
            "rejected": p["rejected"]["full_text"],
        })
    train_ds = Dataset.from_list(rows["train"])
    if rows["valid"]:
        valid_ds = Dataset.from_list(rows["valid"])
    else:  # tiny/smoke runs: keep the schema without failing inference on []
        valid_ds = Dataset.from_dict({c: [] for c in train_ds.column_names},
                                     features=train_ds.features)
    DatasetDict({"train": train_ds, "valid": valid_ds}).save_to_disk(out_dir)
    return len(rows["train"]), len(rows["valid"])


def main():
    ap = argparse.ArgumentParser(description="Export the DPO DatasetDict (CPU).")
    ap.add_argument("--force", action="store_true", help="Re-run even if cached.")
    args = ap.parse_args()

    env_setup.setup_env()
    config.ensure_output_dirs()

    params = {
        "upstream": upstream_hash("09_pairs"),
        "valid_fraction": config.VALID_FRACTION,
        "split_seed": config.SPLIT_SEED,
    }
    outputs = [os.path.join(config.DPO_DATASET_DIR, "dataset_dict.json")]
    if check_stage_cache(STAGE, params, outputs, force=args.force):
        return

    pairs = read_jsonl(config.PAIRS_PATH)
    if not pairs:
        raise SystemExit("No pairs to export — run stages up to 09 first.")

    valid_ids = _split_ids(pairs, config.VALID_FRACTION, config.SPLIT_SEED)
    n_train, n_valid = _export(pairs, valid_ids, config.DPO_DATASET_DIR)

    print(f"Exported -> {config.DPO_DATASET_DIR}  (train={n_train}, valid={n_valid})")
    print("  columns: prompt, chosen, rejected (full completions incl. <think>)")
    print(f"  load with: datasets.load_from_disk({config.DPO_DATASET_DIR!r})")

    update_diagnostics(config.DIAGNOSTICS_PATH, "export", {
        "n_pairs": len(pairs),
        "n_train": n_train,
        "n_valid": n_valid,
        "valid_fraction": config.VALID_FRACTION,
        "split_seed": config.SPLIT_SEED,
        "stratified_by": "direction",
        "columns": ["prompt", "chosen", "rejected"],
        "path": config.DPO_DATASET_DIR,
    })
    write_stage_marker(STAGE, params, {"train": n_train, "valid": n_valid})


if __name__ == "__main__":
    main()
