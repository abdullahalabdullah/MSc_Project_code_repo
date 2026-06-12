#!/usr/bin/env python3
"""
Step 5: Generate plots for the g-mean^2 monitorability evaluation.

Reads analysis/{model}/gmean2_per_instance.csv and gmean2_summary.json
(produced by 04_compute_gmean2.py) for each model, and writes:

  plots/{model}_gmean2_te_distribution.png
  plots/{model}_gmean2_distribution.png
  plots/{model}_tpr_tnr_scatter.png
  plots/{model}_cot_length_distribution.png
  plots/gmean2_summary_table.json   (combined summaries across all models)

Usage:
    python scripts/05_analyze_gmean2.py
    python scripts/05_analyze_gmean2.py --model tulu_sft_no_safety
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.gmean2_analysis import (
    plot_te_distribution,
    plot_gmean2_distribution,
    plot_tpr_tnr_scatter,
    plot_cot_length_distribution,
)


def _f(v):
    """Parse a CSV cell into float/None."""
    if v is None or v == "" or v == "None":
        return None
    return float(v)


def _b(v):
    """Parse a CSV cell into bool."""
    return str(v).strip().lower() == "true"


def _load_per_instance(csv_path: Path) -> list:
    per_instance = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            per_instance.append({
                "instance_id": row["instance_id"],
                "hint_target": row["hint_target"],
                "correct_answer": row["correct_answer"],
                "p1_i": _f(row["p1_i"]),
                "p0_i": _f(row["p0_i"]),
                "te_i": _f(row["te_i"]),
                "r_effect_i": _f(row["r_effect_i"]),
                "level1_pass": _b(row["level1_pass"]),
                "tpr_mc_i": _f(row["tpr_mc_i"]),
                "tnr_x0_i": _f(row["tnr_x0_i"]),
                "tnr_x1_i": _f(row["tnr_x1_i"]),
                "tnr_x1_vacuous_i": _b(row["tnr_x1_vacuous_i"]),
                "gmean2_i": _f(row["gmean2_i"]),
                "acc_u_i": _f(row["acc_u_i"]),
                "mean_cot_tokens_hinted": _f(row["mean_cot_tokens_hinted"]),
                "mean_cot_tokens_unhinted": _f(row["mean_cot_tokens_unhinted"]),
                "mean_cot_tokens_hint_followed": _f(row["mean_cot_tokens_hint_followed"]),
                "mean_cot_tokens_hint_not_followed": _f(row["mean_cot_tokens_hint_not_followed"]),
            })
    return per_instance


def main():
    parser = argparse.ArgumentParser(description="Generate g-mean^2 plots.")
    parser.add_argument("--model", default=None,
                        help="Model name (default: run all models in config)")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    config_path = root / args.config

    with open(config_path) as f:
        config = yaml.safe_load(f)

    models = config["models"]
    if args.model:
        models = [m for m in models if m["name"] == args.model]
        if not models:
            print(f"ERROR: model '{args.model}' not found in config.")
            sys.exit(1)

    plots_dir = root / config["paths"]["plots_dir"]
    plots_dir.mkdir(parents=True, exist_ok=True)
    analysis_root = root / config["paths"]["analysis_dir"]

    summaries = []

    for model_cfg in models:
        model_name = model_cfg["name"]
        analysis_dir = analysis_root / model_name

        csv_path = analysis_dir / "gmean2_per_instance.csv"
        summary_path = analysis_dir / "gmean2_summary.json"

        if not csv_path.exists() or not summary_path.exists():
            print(f"  Skipping {model_name}: {csv_path} or {summary_path} not found "
                  f"(run 04_compute_gmean2.py first).")
            continue

        print(f"\n--- {model_name} ---")
        per_instance = _load_per_instance(csv_path)

        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
        summaries.append(summary)

        plot_te_distribution(per_instance, plots_dir / f"{model_name}_gmean2_te_distribution.png")
        plot_gmean2_distribution(per_instance, plots_dir / f"{model_name}_gmean2_distribution.png")
        plot_tpr_tnr_scatter(per_instance, plots_dir / f"{model_name}_tpr_tnr_scatter.png")
        plot_cot_length_distribution(per_instance, plots_dir / f"{model_name}_cot_length_distribution.png")

        print(f"  Wrote plots for {model_name} to {plots_dir}")

    if summaries:
        table_path = plots_dir / "gmean2_summary_table.json"
        with open(table_path, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2)
        print(f"\nWritten combined summary table: {table_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
