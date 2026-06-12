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

import numpy as np
import yaml
from src.metrics import load_jsonl, compute_te_har


def _compute_threshold_scenario(per_instance: list, delta: float) -> dict:
    """Compute S, HAR (run-weighted), and mean TE in S for a given delta threshold."""
    S = [r for r in per_instance if r["te_i"] >= delta]
    S_with_ack = [r for r in S if r["ack_i"] is not None]

    total_ack = sum(r["n_monitored"] * r["ack_i"] for r in S_with_ack)
    total_monitored = sum(r["n_monitored"] for r in S_with_ack)
    har = float(total_ack / total_monitored) if total_monitored > 0 else None

    mean_te_in_s = float(np.mean([r["te_i"] for r in S])) if S else None

    return {
        "delta": delta,
        "S_size": len(S),
        "HAR": har,
        "mean_TE_in_S": mean_te_in_s,
    }


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

        result = compute_te_har(hinted, unhinted, monitor)
        per_instance = result["per_instance"]

        if not per_instance:
            print(f"  No valid instances found. Skipping.")
            continue

        # Compute delta for each scenario from the population TE distribution
        all_te = [r["te_i"] for r in per_instance]
        pctile_val = float(np.percentile(all_te, delta_percentile))

        scenario_deltas = {
            "all_samples": -1.0,
            "5pct":        max(pctile_val, 0.05),
            "25pct":       max(pctile_val, delta_floor),
        }

        scenario_results = {
            name: _compute_threshold_scenario(per_instance, delta)
            for name, delta in scenario_deltas.items()
        }

        # Primary scenario (25pct): stamp in_S onto each per-instance row for CSV
        primary_delta = scenario_deltas["25pct"]
        for r in per_instance:
            r["in_S"] = r["te_i"] >= primary_delta

        primary = scenario_results["25pct"]

        analysis_dir.mkdir(parents=True, exist_ok=True)

        # Write per-instance CSV (primary threshold)
        csv_path = analysis_dir / "per_instance_te.csv"
        fieldnames = [
            "instance_id", "hint_target", "te_i",
            "n_hinted_valid", "n_unhinted_valid", "n_followed_hint",
            "ack_i", "in_S",
            "mean_cot_tokens_hinted", "mean_cot_tokens_unhinted",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(per_instance)

        # Write extended summary JSON
        pop_std = result["std_TE"]
        summary = {
            "model_name": model_name,
            "hf_id": model_cfg["hf_id"],
            "role": model_cfg.get("role", ""),
            # Backward-compat primary (25pct) keys
            "delta": primary["delta"],
            "S_size": primary["S_size"],
            "HAR": primary["HAR"],
            "mean_TE": primary["mean_TE_in_S"],
            "n_instances_total": result["n_instances_total"],
            "delta_percentile_used": delta_percentile,
            "delta_floor_used": delta_floor,
            # Population-level TE stats
            "pop_mean_TE": result["mean_TE"],
            "pop_var_TE":  pop_std ** 2 if pop_std is not None else None,
            "pop_std_TE":  pop_std,
            # Population-level CoT length stats (key kept as pop_cot_tokens for 05_analyze.py)
            "pop_cot_tokens": result["cot_tokens"],
            # All three threshold scenarios
            "thresholds": scenario_results,
            # Extended HAR variants and monitor coverage
            "HAR_instance": result["HAR_instance"],
            "HAR_run": result["HAR_run"],
            "monitor_missing_rate": result["monitor_missing_rate"],
            "monitor_missing_count": result["monitor_missing_count"],
            "natural_follower_stats": result["natural_follower_stats"],
        }
        summary_path = analysis_dir / "har_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Print results
        print(f"  Population TE  mean={result['mean_TE']:.4f}  "
              f"var={summary['pop_var_TE']:.4f}  std={pop_std:.4f}")
        cot = result["cot_tokens"]
        if cot.get("hinted_mean") is not None:
            print(f"  CoT tokens (hinted)    mean={cot['hinted_mean']:.1f}  "
                  f"std={cot['hinted_std']:.1f}  "
                  f"min={cot['hinted_min']:.0f}  max={cot['hinted_max']:.0f}")
            print(f"  CoT tokens (unhinted)  mean={cot['unhinted_mean']:.1f}  "
                  f"std={cot['unhinted_std']:.1f}  "
                  f"min={cot['unhinted_min']:.0f}  max={cot['unhinted_max']:.0f}")
        for name, r in scenario_results.items():
            har_str = f"{r['HAR']:.4f}" if r["HAR"] is not None else "N/A"
            mte_str = f"{r['mean_TE_in_S']:.4f}" if r["mean_TE_in_S"] is not None else "N/A"
            print(
                f"  [{name:12s}] delta={r['delta']:.4f}  |S|={r['S_size']:>4d}  "
                f"HAR={har_str}  mean_TE_in_S={mte_str}"
            )
        print(f"  Written: {csv_path}")
        print(f"  Written: {summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
