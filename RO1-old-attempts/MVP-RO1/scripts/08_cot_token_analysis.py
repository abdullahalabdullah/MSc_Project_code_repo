#!/usr/bin/env python3
"""
Step 8: CoT token-length analysis to inform the max_new_tokens decision.

The existing per-dataset gmean2_summary.md already reports a 4-condition CoT
token table for one dataset (the "overall" view for that run, since every
pipeline run is namespaced to a single dataset). This script aggregates ACROSS
datasets for a model and writes one combined report containing:

  1. Overall CoT token stats (pooled across all datasets), 4 conditions —
     computed the same per-instance-mean way as gmean2_summary.md so the numbers
     line up with that table.
  2. Per-dataset CoT token stats, 4 conditions — the cross-dataset comparison.
  3. Per-dataset token-length BUCKET counts over the raw per-sample token_count
     of every response (both arms), plus a truncation count/rate — the view that
     actually tells you where to set max_new_tokens.

Usage:
    python scripts/08_cot_token_analysis.py                     # all models, all datasets
    python scripts/08_cot_token_analysis.py --model qwen3_4b
    python scripts/08_cot_token_analysis.py --datasets mmlu gpqa wmdp

Input (per dataset, per model):
    outputs/{dataset}/{model}/hinted.jsonl
    outputs/{dataset}/{model}/unhinted.jsonl
    (monitor results are not required for token lengths)

Output:
    analysis/cot_token_report_{model}.md
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.gmean2_metrics import load_jsonl, compute_gmean2, _cot_stats
from src.dataset import resolve_paths
from src.cot_tokens import (
    DEFAULT_BUCKET_EDGES,
    bucket_labels,
    bucket_counts,
    raw_token_stats,
    truncation_stats,
)

CONDITIONS = [
    ("hinted",            "Hinted (X=1, all)"),
    ("unhinted",          "Unhinted (X=0, all)"),
    ("hint_followed",     "Hinted — hint followed (X=1, Y=1)"),
    ("hint_not_followed", "Hinted — hint not followed (X=1, Y=0)"),
]


def _fmt(v, decimals=1):
    if v is None:
        return "N/A"
    if decimals == 0:
        return f"{int(round(v))}"
    return f"{v:.{decimals}f}"


def _condition_table(cot: dict) -> list:
    """Render a 4-condition mean/std/min/max table from a cot_tokens-style dict."""
    lines = ["| Condition | Mean | Std | Min | Max |",
             "|-----------|------|-----|-----|-----|"]
    for prefix, label in CONDITIONS:
        lines.append(
            f"| {label} "
            f"| {_fmt(cot.get(f'{prefix}_mean'), 1)} "
            f"| {_fmt(cot.get(f'{prefix}_std'), 1)} "
            f"| {_fmt(cot.get(f'{prefix}_min'), 0)} "
            f"| {_fmt(cot.get(f'{prefix}_max'), 0)} |"
        )
    return lines


def analyze_model(root, config, dataset_keys, model_cfg) -> str | None:
    """Build the markdown report for one model, or None if it has no data."""
    model_name = model_cfg["name"]

    # Pools for the "overall" (across-dataset) condition table — per-instance
    # means, matching gmean2_summary.md methodology.
    pooled_means = {prefix: [] for prefix, _ in CONDITIONS}
    pooled_records = []  # every response (both arms, all datasets), for buckets/max

    per_dataset = []  # list of (dataset_key, cot_tokens, bucket_counts, raw_stats, trunc)

    for ds in dataset_keys:
        paths = resolve_paths(root, config, ds, model_name)
        out_dir = paths["outputs_dir"]
        hinted_path = out_dir / "hinted.jsonl"
        unhinted_path = out_dir / "unhinted.jsonl"
        if not hinted_path.exists() or not unhinted_path.exists():
            continue

        hinted = load_jsonl(hinted_path)
        unhinted = load_jsonl(unhinted_path)
        if not hinted or not unhinted:
            continue

        # Condition stats (per-instance mean) — reuse compute_gmean2 so they are
        # byte-for-byte the same methodology as gmean2_summary.md.
        result = compute_gmean2(hinted, unhinted, monitor_records=[])
        cot = result["cot_tokens"]
        for r in result["per_instance"]:
            for prefix in pooled_means:
                v = r.get(f"mean_cot_tokens_{prefix}")
                if v is not None:
                    pooled_means[prefix].append(v)

        # Raw per-sample buckets over every response in both arms. Truncated
        # responses are routed to the overflow bucket by bucket_counts (their
        # true length is censored above the cap).
        all_records = hinted + unhinted
        pooled_records.extend(all_records)

        per_dataset.append((
            ds,
            cot,
            bucket_counts(all_records),
            raw_token_stats(all_records),
            truncation_stats(all_records),
        ))

    if not per_dataset:
        return None

    overall_cot = {
        **_cot_stats(pooled_means["hinted"], "hinted"),
        **_cot_stats(pooled_means["unhinted"], "unhinted"),
        **_cot_stats(pooled_means["hint_followed"], "hint_followed"),
        **_cot_stats(pooled_means["hint_not_followed"], "hint_not_followed"),
    }

    labels = bucket_labels(DEFAULT_BUCKET_EDGES)
    lines = []
    w = lines.append

    w(f"# CoT token-length analysis: {model_name}\n")
    w(f"**HuggingFace ID:** `{model_cfg['hf_id']}`  ")
    w(f"**Datasets included:** {', '.join(ds for ds, *_ in per_dataset)}\n")

    # 1. Overall condition table -------------------------------------------------
    w("---\n")
    w("## 1. Overall CoT token length (pooled across all datasets)\n")
    w("Mean output tokens in the chain-of-thought, averaged per instance then "
      "across instances, pooled over every dataset above. Same methodology as "
      "each dataset's `gmean2_summary.md`.\n")
    lines.extend(_condition_table(overall_cot))
    w("")

    # 2. Per-dataset condition table --------------------------------------------
    w("---\n")
    w("## 2. CoT token length per dataset\n")
    w("Same four conditions, computed separately for each dataset.\n")
    for ds, cot, *_ in per_dataset:
        w(f"### {ds}\n")
        lines.extend(_condition_table(cot))
        w("")

    # 3. Per-dataset bucket table ------------------------------------------------
    w("---\n")
    w("## 3. Token-length buckets per dataset\n")
    w("Number of individual responses (every sample, both arms) whose "
      "`token_count` falls in each length band. **Truncated responses (those "
      f"that hit `max_new_tokens`) are counted in the top `{labels[-1]}` bucket "
      "regardless of their cap** — their true length is censored above the cap, "
      "so they represent the overflow tail, not a normal-length trace at the cap "
      "value. The `Truncated` column repeats that count explicitly. A high "
      "truncation rate means the cap is too low — use this to pick "
      "`max_new_tokens`.\n")

    header = "| Dataset | " + " | ".join(labels) + " | Total | Truncated (rate) | Max |"
    sep = "|" + "---|" * (len(labels) + 4)
    w(header)
    w(sep)
    # Per-dataset rows
    overall_bucket = bucket_counts(pooled_records)
    for ds, _cot, buckets, rstats, trunc in per_dataset:
        row = [ds] + [str(buckets[l]) for l in labels]
        rate = f"{trunc['rate']*100:.1f}%" if trunc["rate"] is not None else "N/A"
        row += [str(trunc["total"]), f"{trunc['truncated']} ({rate})", _fmt(rstats["max"], 0)]
        w("| " + " | ".join(row) + " |")
    # Overall (pooled) row
    pooled_max = max((r["token_count"] for r in pooled_records
                      if r.get("token_count") is not None), default=None)
    total_trunc = sum(t["truncated"] for *_, t in per_dataset)
    grand_total = sum(t["total"] for *_, t in per_dataset)
    rate = f"{(total_trunc/grand_total)*100:.1f}%" if grand_total else "N/A"
    row = ["**ALL**"] + [f"**{overall_bucket[l]}**" for l in labels]
    row += [f"**{grand_total}**", f"**{total_trunc} ({rate})**", f"**{_fmt(pooled_max, 0)}**"]
    w("| " + " | ".join(row) + " |")
    w("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Cross-dataset CoT token-length analysis.")
    parser.add_argument("--model", default=None,
                        help="Model name (default: every model in config).")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Dataset keys (default: mmlu + every key in the datasets registry).")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    with open(root / args.config) as f:
        config = yaml.safe_load(f)

    if args.datasets:
        dataset_keys = args.datasets
    else:
        # mmlu first (legacy path), then the registry keys, de-duplicated.
        dataset_keys = ["mmlu"] + [k for k in config.get("datasets", {}) if k != "mmlu"]

    models = config["models"]
    if args.model:
        models = [m for m in models if m["name"] == args.model]
        if not models:
            print(f"ERROR: model '{args.model}' not found in config.")
            sys.exit(1)

    analysis_root = root / config["paths"]["analysis_dir"]
    analysis_root.mkdir(parents=True, exist_ok=True)

    wrote_any = False
    for model_cfg in models:
        report = analyze_model(root, config, dataset_keys, model_cfg)
        if report is None:
            print(f"  Skipping {model_cfg['name']}: no inference outputs found.")
            continue
        out_path = analysis_root / f"cot_token_report_{model_cfg['name']}.md"
        out_path.write_text(report, encoding="utf-8")
        print(f"  Written: {out_path}")
        wrote_any = True

    if not wrote_any:
        print("No reports written (no inference outputs found for any model).")
    print("\nDone.")


if __name__ == "__main__":
    main()
