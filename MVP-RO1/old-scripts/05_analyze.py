#!/usr/bin/env python3
"""
Step 5: Generate plots and confusion matrix data for all models.

Reads per_instance_te.csv and JSONL files, produces confusion matrices
and a combined TE-vs-ack scatter plot.

Usage:
    python scripts/05_analyze.py [--config config/mvp_config.yaml]

Input:
    analysis/{model_name}/per_instance_te.csv
    analysis/{model_name}/har_summary.json
    outputs/{model_name}/hinted.jsonl
    monitor_outputs/{model_name}/monitor_results.jsonl

Output:
    plots/{model_name}_confusion_matrix.png
    plots/{model_name}_confusion_matrix.json
    plots/te_ack_scatter_all_models.png
    plots/summary_table.json          — HAR / |S| / mean_TE for all models
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.metrics import load_jsonl
from src.analysis import (
    confusion_matrix_data,
    plot_confusion_matrix,
    plot_te_ack_scatter,
    plot_te_binned_histogram,
    plot_cot_length_distribution,
)


def load_per_instance_csv(path: Path) -> list:
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            def _float(v):
                return float(v) if v not in ("", "None") else None
            rows.append({
                "instance_id": int(row["instance_id"]),
                "te_i": float(row["te_i"]),
                "ack_i": _float(row["ack_i"]),
                "in_S": row["in_S"] == "True",
                "n_followed_hint": int(row["n_followed_hint"]),
                "mean_cot_tokens_hinted":   _float(row.get("mean_cot_tokens_hinted",   "")),
                "mean_cot_tokens_unhinted": _float(row.get("mean_cot_tokens_unhinted",  "")),
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate analysis plots.")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    config_path = root / args.config

    with open(config_path) as f:
        config = yaml.safe_load(f)

    plots_dir = root / config["paths"]["plots_dir"]
    plots_dir.mkdir(parents=True, exist_ok=True)

    per_instance_all = {}
    summary_table = []
    global_delta = config["analysis"]["delta_floor"]

    for model_cfg in config["models"]:
        model_name = model_cfg["name"]
        analysis_dir = root / config["paths"]["analysis_dir"] / model_name
        out_dir = root / config["paths"]["outputs_dir"] / model_name
        mon_dir = root / config["paths"]["monitor_outputs_dir"] / model_name

        csv_path = analysis_dir / "per_instance_te.csv"
        summary_path = analysis_dir / "har_summary.json"

        if not csv_path.exists():
            print(f"Skipping {model_name}: {csv_path} not found. Run 04_compute_metrics.py first.")
            continue

        per_instance = load_per_instance_csv(csv_path)
        per_instance_all[model_name] = per_instance

        with open(summary_path) as f:
            summary = json.load(f)
        global_delta = summary["delta"]  # use last model's delta; ideally all same
        summary_table.append(summary)

        S = {r["instance_id"] for r in per_instance if r["in_S"]}
        print(f"\n{model_name}: n_total={summary.get('n_instances_total')}")

        # Population-level TE stats
        if "pop_mean_TE" in summary:
            print(
                f"  Population TE  mean={summary['pop_mean_TE']:.4f}  "
                f"var={summary['pop_var_TE']:.4f}  std={summary['pop_std_TE']:.4f}"
            )

        # Three-threshold results
        if "thresholds" in summary:
            for name, th in summary["thresholds"].items():
                har_str = f"{th['HAR']:.4f}" if th["HAR"] is not None else "N/A"
                mte_str = f"{th['mean_TE_in_S']:.4f}" if th["mean_TE_in_S"] is not None else "N/A"
                print(
                    f"  [{name:12s}] delta={th['delta']:.4f}  |S|={th['S_size']:>4d}  "
                    f"HAR={har_str}  mean_TE_in_S={mte_str}"
                )
        else:
            print(f"  [25pct (legacy)] |S|={len(S)}, HAR={summary.get('HAR')}, delta={summary['delta']:.3f}")

        # CoT length stats
        if "pop_cot_tokens" in summary:
            cot = summary["pop_cot_tokens"]
            if cot.get("hinted_mean") is not None:
                print(
                    f"  CoT tokens (hinted)    mean={cot['hinted_mean']:.1f}  "
                    f"std={cot['hinted_std']:.1f}  "
                    f"min={cot['hinted_min']:.0f}  max={cot['hinted_max']:.0f}"
                )
                print(
                    f"  CoT tokens (unhinted)  mean={cot['unhinted_mean']:.1f}  "
                    f"std={cot['unhinted_std']:.1f}  "
                    f"min={cot['unhinted_min']:.0f}  max={cot['unhinted_max']:.0f}"
                )

        # Binned histogram (fixed 0.2-wide bins)
        binned_hist_path = plots_dir / f"{model_name}_te_binned_histogram.png"
        plot_te_binned_histogram(per_instance, binned_hist_path)
        print(f"  TE binned hist   -> {binned_hist_path}")

        # CoT length distribution plot
        cot_dist_path = plots_dir / f"{model_name}_cot_length_distribution.png"
        plot_cot_length_distribution(per_instance, cot_dist_path)
        print(f"  CoT length dist  -> {cot_dist_path}")

        # Confusion matrix (S only — instances with TE ≥ δ)
        hinted_path = out_dir / "hinted.jsonl"
        monitor_path = mon_dir / "monitor_results.jsonl"

        if hinted_path.exists() and monitor_path.exists():
            hinted_records = load_jsonl(hinted_path)
            monitor_records = load_jsonl(monitor_path)

            cm_s = confusion_matrix_data(hinted_records, monitor_records, S)
            cm_s_png = plots_dir / f"{model_name}_confusion_matrix_S.png"
            plot_confusion_matrix(cm_s, f"{model_name} (S, δ=25%)", cm_s_png)
            with open(plots_dir / f"{model_name}_confusion_matrix_S.json", "w") as f:
                json.dump(cm_s, f, indent=2)
            print(f"  Confusion matrix (S)    -> {cm_s_png}")
        else:
            print(f"  WARNING: hinted.jsonl or monitor_results.jsonl missing — skipping confusion matrix.")

    # Combined TE vs. ack scatter
    if per_instance_all:
        scatter_path = plots_dir / "te_ack_scatter_all_models.png"
        plot_te_ack_scatter(per_instance_all, global_delta, scatter_path)
        print(f"\nScatter plot -> {scatter_path}")

    # Summary table JSON
    summary_table_path = plots_dir / "summary_table.json"
    with open(summary_table_path, "w") as f:
        json.dump(summary_table, f, indent=2)
    print(f"Summary table -> {summary_table_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
