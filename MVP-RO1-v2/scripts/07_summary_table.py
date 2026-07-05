#!/usr/bin/env python3
"""
Print a per-model summary table from gmean2_summary.json files.

Scans outputs-and-results/analysis/{dataset}/{model}/{hint_type}/gmean2_summary.json
and prints one markdown table per model.

Usage:
    python scripts/07_summary_table.py
    python scripts/07_summary_table.py --hint-type professor
    python scripts/07_summary_table.py --hint-type professor --output summary.md
"""

import argparse
import json
from pathlib import Path

DATASET_ORDER = [
    "mmlu_redux",
    "gpqa_diamond",
    "wmdp",
    "codemmlu",
    "lsat_ar",
    "lsat_lr",
]


def load_summaries(analysis_dir: Path, hint_type: str) -> dict[str, dict[str, dict]]:
    """Return {model: {dataset: summary_dict}}."""
    data: dict[str, dict[str, dict]] = {}
    for ds_dir in analysis_dir.iterdir():
        if not ds_dir.is_dir() or ds_dir.name.startswith("."):
            continue
        dataset = ds_dir.name
        for model_dir in ds_dir.iterdir():
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            summary_path = model_dir / hint_type / "gmean2_summary.json"
            if not summary_path.exists():
                continue
            with open(summary_path) as f:
                summary = json.load(f)
            data.setdefault(model, {})[dataset] = summary
    return data


def fmt_acc(mean, std) -> str:
    return f"{mean:.3f} ({std:.3f})"


def build_table(model: str, datasets: dict[str, dict]) -> str:
    header = (
        "| Dataset | Mean TE (before) | Mean R_effect (before) "
        "| n TE>0 (post L1) | g-mean² "
        "| Unhinted Acc (Std) | Hinted Acc (Std) |"
    )
    sep = (
        "| --- | --- | --- | --- | --- | --- | --- |"
    )
    rows = [f"## {model}\n", header, sep]

    ordered = [d for d in DATASET_ORDER if d in datasets]
    others = sorted(d for d in datasets if d not in DATASET_ORDER)
    for dataset in ordered + others:
        s = datasets[dataset]
        row = (
            f"| {dataset} "
            f"| {s['mean_TE_before']:.4f} "
            f"| {s['mean_R_effect_before']:.4f} "
            f"| {s['n_level1_pass']} "
            f"| {s['gmean2']:.4f} "
            f"| {fmt_acc(s['mean_acc_u'], s['std_acc_u'])} "
            f"| {fmt_acc(s['mean_acc_h'], s['std_acc_h'])} |"
        )
        rows.append(row)

    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description="Print per-model summary tables.")
    parser.add_argument("--hint-type", default="professor",
                        help="Hint type to summarise (default: professor).")
    parser.add_argument("--analysis-dir", default="outputs-and-results/analysis",
                        help="Path to the analysis directory.")
    parser.add_argument("--output", default=None,
                        help="Output .md path (default: outputs-and-results/analysis/summary_{hint_type}.md).")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    analysis_dir = root / args.analysis_dir

    data = load_summaries(analysis_dir, args.hint_type)
    if not data:
        print(f"No gmean2_summary.json files found under {analysis_dir} "
              f"for hint_type='{args.hint_type}'.")
        return

    blocks = []
    for model in sorted(data):
        blocks.append(build_table(model, data[model]))

    output = "\n\n".join(blocks) + "\n"

    out_path = root / (args.output or f"outputs-and-results/analysis/summary_{args.hint_type}.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output)
    print(f"Written to {out_path}")
    print(output)


if __name__ == "__main__":
    main()
