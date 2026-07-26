#!/usr/bin/env python3
"""
Step 8: Compute CoT token-length statistics (to inform the max_new_tokens decision).

Writes one machine-readable cot_token_stats.json per (dataset, hint-type) run,
next to the inference-derived analysis artifacts. `token_count` is the FULL
completion length (thinking trace + final answer). Markdown reports are
generated separately by scripts/09_generate_reports.py (see src/reporting.py).

Each JSON contains:
  - condition_stats: 4-condition run-level mean/std/min/max/runs/instances
    (valid runs only — not truncated, answer parsed — so followed +
    not-followed partition hinted exactly). Same methodology as
    compute_gmean2(...)["cot_tokens"].
  - cot_token_pools: the raw valid-run token counts per condition, so roll-up
    reports can pool across datasets/hint types without re-reading inference.
  - length_pools: the ENRICHED per-condition pools for length analysis — per-run
    (instance_id, token_count, truncated) parallel arrays, INCLUDING truncated
    runs (kept at their observed token_count, which equals max_new_tokens — the
    cap — since truncation is caused by length; dropping them would bias length
    down). "hinted"/"unhinted" hold every run with a token_count (incl.
    truncated/unparsed); "hint_followed"/"hint_not_followed" hold only runs with
    a parsed_answer (== / != hint_target), so a few truncated-before-answering
    runs sit in the "all" pools but not the followed split. instance_id enables
    exact distinct-instance counts and per-instance pairing downstream (step 44).
  - buckets / truncation / raw_stats: over EVERY response (both arms, including
    truncated/unparsed) — truncated responses count in the open-ended top
    bucket (their true length is censored above the cap).

Usage:
    python scripts/08_cot_token_analysis.py --model Qwen/Qwen3-4B                   # all datasets
    python scripts/08_cot_token_analysis.py --model Qwen/Qwen3-4B --datasets wmdp codemmlu

Input (per dataset):
    outputs/{model}/{dataset}/{hint}/hinted.jsonl
    outputs/{model}/{dataset}/{hint}/unhinted.jsonl
    (monitor results are not required for token lengths)

Output (per dataset):
    analysis/{model}/{dataset}/{hint}/cot_token_stats.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from src.gmean2_metrics import load_jsonl, resolve_jsonl, compute_gmean2
from src.dataset import SPLITS, model_dir_name, resolve_paths
from src.hints import HINT_TYPES
from src.cot_tokens import bucket_counts, raw_token_stats, truncation_stats


def build_length_pools(hinted, unhinted):
    """Enriched per-condition pools for length analysis: per-run
    (instance_id, token_count, truncated) parallel arrays, INCLUDING truncated
    runs at their observed (capped) token_count. followed/not-followed require a
    parsed_answer (== / != hint_target); "all" pools include every run that has a
    token_count (so truncated-before-answering runs land only in "all")."""
    pools = {"hinted": [], "unhinted": [], "hint_followed": [],
             "hint_not_followed": []}
    for r in unhinted:
        if r.get("token_count") is None:
            continue
        pools["unhinted"].append(r)
    for r in hinted:
        if r.get("token_count") is None:
            continue
        pools["hinted"].append(r)
        pa = r.get("parsed_answer")
        if pa is None:
            continue  # unclassifiable (e.g. truncated before answering)
        key = "hint_followed" if pa == r.get("hint_target") else "hint_not_followed"
        pools[key].append(r)
    return {k: {"instance_id": [r.get("instance_id") for r in v],
                "token_count": [r["token_count"] for r in v],
                "truncated": [int(bool(r.get("truncated", False))) for r in v]}
            for k, v in pools.items()}


def main():
    parser = argparse.ArgumentParser(description="Compute CoT token-length statistics.")
    parser.add_argument("--model", required=True,
                        help="HF model ID used at inference (e.g. Qwen/Qwen3-4B).")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Dataset keys (default: every key in the datasets registry).")
    parser.add_argument("--hint-type", default=None, choices=HINT_TYPES,
                        help="Hint type to read (default: config hint.default). "
                             "Must match the inference run. Namespaces I/O paths.")
    parser.add_argument("--data", default="test", choices=SPLITS,
                        help="Data split to read: test (default) or eval. "
                             "Must match the inference run. Namespaces I/O paths.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    with open(root / args.config) as f:
        config = yaml.safe_load(f)

    hint_type = args.hint_type or config.get("hint", {}).get("default", "professor")
    dataset_keys = args.datasets or list(config.get("datasets", {}))
    model_name = model_dir_name(args.model)

    wrote_any = False
    for ds in dataset_keys:
        paths = resolve_paths(root, config, ds, model_name, hint_type, split=args.data)
        out_dir = paths["outputs_dir"]
        hinted_path = resolve_jsonl(out_dir / "hinted.jsonl")
        unhinted_path = resolve_jsonl(out_dir / "unhinted.jsonl")
        if not hinted_path.exists() or not unhinted_path.exists():
            print(f"  Skipping {ds}: no inference outputs in {out_dir}")
            continue

        hinted = load_jsonl(hinted_path)
        unhinted = load_jsonl(unhinted_path)
        if not hinted or not unhinted:
            print(f"  Skipping {ds}: empty inference outputs")
            continue

        # Condition stats + raw pools (valid runs) — reuse compute_gmean2 so the
        # methodology is byte-for-byte the same as the results artifacts.
        result = compute_gmean2(hinted, unhinted, monitor_records=[])

        # Bucket/truncation view over EVERY response in both arms.
        all_records = hinted + unhinted

        stats = {
            "model_name": model_name,
            "hf_id": args.model,
            "dataset": ds,
            "hint_type": hint_type,
            "condition_stats": result["cot_tokens"],
            "cot_token_pools": result["cot_token_pools"],
            "length_pools": build_length_pools(hinted, unhinted),
            "buckets": bucket_counts(all_records),
            "truncation": truncation_stats(all_records),
            "raw_stats": raw_token_stats(all_records),
        }

        analysis_dir = paths["analysis_dir"]
        analysis_dir.mkdir(parents=True, exist_ok=True)
        out_path = analysis_dir / "cot_token_stats.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"  Written: {out_path}")
        wrote_any = True

    if not wrote_any:
        print("No stats written (no inference outputs found).")
    print("\nDone.")


if __name__ == "__main__":
    main()
