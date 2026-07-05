#!/usr/bin/env python3
"""
Step 5: Generate plots for the g-mean^2 monitorability evaluation.

Reads analysis/{model}/{dataset}/{hint}/results_per_instance.csv (produced by
04_compute_gmean2.py) and writes to plots/{model}/{dataset}/{hint}/:

  gmean2_te_distribution.png
  gmean2_distribution.png
  tpr_tnr_scatter.png
  cot_length_distribution.png

Usage:
    python scripts/05_analyze_gmean2.py --model Qwen/Qwen3-4B                    # every dataset
    python scripts/05_analyze_gmean2.py --model Qwen/Qwen3-4B --dataset wmdp    # one dataset

    Options:
      --config   path to config YAML (default: config/mvp_config.yaml)
      --dataset  restrict to one dataset key (default: every dataset in config)
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from src.dataset import model_dir_name, resolve_paths
from src.hints import HINT_TYPES
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


def run_dataset(root, config, dataset_key, models, hint_type):
    """Generate plots for every given model on a single dataset."""
    print(f"\n=== dataset: {dataset_key}  (hint: {hint_type}) ===")

    for model_cfg in models:
        model_name = model_cfg["name"]
        paths = resolve_paths(root, config, dataset_key, model_name, hint_type)
        analysis_dir = paths["analysis_dir"]
        plots_dir = paths["plots_dir"]

        csv_path = analysis_dir / "results_per_instance.csv"

        if not csv_path.exists():
            print(f"  Skipping {model_name}: {csv_path} not found "
                  f"(run 04_compute_gmean2.py first).")
            continue

        print(f"\n--- {model_name} ---")
        per_instance = _load_per_instance(csv_path)
        plots_dir.mkdir(parents=True, exist_ok=True)

        plot_te_distribution(per_instance, plots_dir / "gmean2_te_distribution.png")
        plot_gmean2_distribution(per_instance, plots_dir / "gmean2_distribution.png")
        plot_tpr_tnr_scatter(per_instance, plots_dir / "tpr_tnr_scatter.png")
        plot_cot_length_distribution(per_instance, plots_dir / "cot_length_distribution.png")

        print(f"  Wrote plots for {model_name} to {plots_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate g-mean^2 plots.")
    parser.add_argument("--model", required=True,
                        help="HF model ID used at inference (e.g. Qwen/Qwen3-4B).")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--dataset", default=None,
                        help="Restrict to one dataset key (default: every dataset in config). "
                             "Namespaces I/O paths.")
    parser.add_argument("--hint-type", default=None, choices=HINT_TYPES,
                        help="Hint type to read (default: config hint.default). "
                             "Must match the inference/monitor run. Namespaces I/O paths.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    config_path = root / args.config

    with open(config_path) as f:
        config = yaml.safe_load(f)

    hint_type = args.hint_type or config.get("hint", {}).get("default", "professor")
    models = [{"name": model_dir_name(args.model), "hf_id": args.model}]

    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = list(config.get("datasets", {}).keys())
        if not datasets:
            print("ERROR: config has no 'datasets' section to sweep.")
            sys.exit(1)

    for dataset_key in datasets:
        run_dataset(root, config, dataset_key, models, hint_type)

    print("\nDone.")


if __name__ == "__main__":
    main()
