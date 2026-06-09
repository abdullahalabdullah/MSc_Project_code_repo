#!/usr/bin/env python3
"""
Step 4: Compute TE, delta, S, per-instance ack, and HAR for one or all models.

Reads hinted/unhinted JSONL and monitor JSONL, runs compute_te_har, and
writes per-instance CSV and a summary JSON per model.

Usage:
    python scripts/04_compute_metrics.py                       # all models
    python scripts/04_compute_metrics.py --model tulu_sft_no_safety

    Options:
      --config  path to config YAML (default: config/mvp_config.yaml)

Input:
    outputs/{model_name}/hinted.jsonl
    outputs/{model_name}/unhinted.jsonl
    monitor_outputs/{model_name}/monitor_results.jsonl  (optional)

Output:
    analysis/{model_name}/per_instance_te.csv    — one row per instance
    analysis/{model_name}/har_summary.json        — HAR, |S|, mean_TE, delta
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.metrics import load_jsonl, compute_te_har


def main():
    parser = argparse.ArgumentParser(description="Compute TE and HAR metrics.")
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

    delta_percentile = config["analysis"]["delta_percentile"]
    delta_floor = config["analysis"]["delta_floor"]

    for model_cfg in models:
        model_name = model_cfg["name"]
        print(f"\n--- {model_name} ---")

        out_dir = root / config["paths"]["outputs_dir"] / model_name
        mon_dir = root / config["paths"]["monitor_outputs_dir"] / model_name
        analysis_dir = root / config["paths"]["analysis_dir"] / model_name

        hinted_path = out_dir / "hinted.jsonl"
        unhinted_path = out_dir / "unhinted.jsonl"
        monitor_path = mon_dir / "monitor_results.jsonl"

        if not hinted_path.exists() or not unhinted_path.exists():
            print(f"  Skipping: inference outputs not found in {out_dir}")
            continue

        hinted = load_jsonl(hinted_path)
        unhinted = load_jsonl(unhinted_path)
        monitor = load_jsonl(monitor_path) if monitor_path.exists() else []

        if not monitor:
            print(f"  WARNING: no monitor results found. ack_i will be None for all instances.")

        results = compute_te_har(
            hinted, unhinted, monitor,
            delta_percentile=delta_percentile,
            delta_floor=delta_floor,
        )

        analysis_dir.mkdir(parents=True, exist_ok=True)

        # Write per-instance CSV
        csv_path = analysis_dir / "per_instance_te.csv"
        fieldnames = [
            "instance_id", "hint_target", "te_i",
            "n_hinted_valid", "n_unhinted_valid", "n_followed_hint",
            "ack_i", "in_S",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results["per_instance"])

        # Write summary JSON
        summary = {
            "model_name": model_name,
            "hf_id": model_cfg["hf_id"],
            "role": model_cfg.get("role", ""),
            "delta": results["delta"],
            "S_size": results["S_size"],
            "HAR": results["HAR"],
            "mean_TE": results["mean_TE"],
            "n_instances_total": results["n_instances_total"],
            "delta_percentile_used": delta_percentile,
            "delta_floor_used": delta_floor,
        }
        summary_path = analysis_dir / "har_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        har_str = f"{results['HAR']:.4f}" if results["HAR"] is not None else "N/A"
        mean_te_str = f"{results['mean_TE']:.4f}" if results["mean_TE"] is not None else "N/A"
        print(
            f"  HAR={har_str}  |S|={results['S_size']}  "
            f"mean_TE={mean_te_str}  delta={results['delta']:.4f}  "
            f"n_total={results['n_instances_total']}"
        )
        print(f"  Written: {csv_path}")
        print(f"  Written: {summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
