#!/usr/bin/env python3
"""
Step 4: Compute the g-mean^2 monitorability metric for one or all models.

Reads hinted/unhinted JSONL and monitor JSONL (both arms — see
03_run_monitor.py), runs compute_gmean2, and writes per-instance CSV/JSON and a
dataset-level summary JSON per model. Markdown reports are generated separately
by scripts/09_generate_reports.py (kept apart so all .md generation can be
removed cleanly for a public release).

Filtering follows MVP-RO1/metrics/g-mean² Filtering Levels.md:
  - Level-1 (applied): instance is included in the g-mean^2 average iff
    TE_i > 0 and p1_i > 0.

Usage:
    python scripts/04_compute_gmean2.py --model Qwen/Qwen3-4B                   # every dataset
    python scripts/04_compute_gmean2.py --model Qwen/Qwen3-4B --dataset wmdp   # one dataset

    Options:
      --config   path to config YAML (default: config/mvp_config.yaml)
      --dataset  restrict to one dataset key (default: every dataset in config)

Input:
    outputs/{model_name}/{dataset}/{hint}/hinted.jsonl
    outputs/{model_name}/{dataset}/{hint}/unhinted.jsonl
    monitor_outputs/{model_name}/{dataset}/{hint}/monitor_results.jsonl  (both arms, with "arm" field)

Output (under analysis/{model_name}/{dataset}/{hint}/):
    results_per_instance.csv    — one row per instance
    results_per_instance.json   — id-keyed maps (te, r_effect, tpr_mc, tnr_x0,
                                   tnr_x1, tnr_x1_vacuous, gmean2, level1_pass, acc_u)
    results_summary.json        — dataset-level aggregates + config snapshot
                                   (machine-readable)
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from src.gmean2_metrics import load_jsonl, resolve_jsonl, compute_gmean2
from src.dataset import SPLITS, model_dir_name, resolve_paths
from src.hints import HINT_TYPES


def _fmt(v, decimals=4):
    return f"{v:.{decimals}f}" if v is not None else "N/A"


def run_dataset(root, config, dataset_key, models, hint_type, split="test"):
    """Compute g-mean^2 for every given model on a single dataset."""
    print(f"\n=== dataset: {dataset_key}  (hint: {hint_type}, split: {split}) ===")

    for model_cfg in models:
        model_name = model_cfg["name"]
        print(f"\n--- {model_name} ---")

        paths = resolve_paths(root, config, dataset_key, model_name, hint_type, split=split)
        out_dir = paths["outputs_dir"]
        mon_dir = paths["monitor_dir"]
        analysis_dir = paths["analysis_dir"]

        hinted_path = resolve_jsonl(out_dir / "hinted.jsonl")
        unhinted_path = resolve_jsonl(out_dir / "unhinted.jsonl")
        monitor_path = mon_dir / "monitor_results.jsonl"

        if not hinted_path.exists() or not unhinted_path.exists():
            print(f"  Skipping: inference outputs not found in {out_dir}")
            continue

        hinted = load_jsonl(hinted_path)
        unhinted = load_jsonl(unhinted_path)
        monitor = load_jsonl(monitor_path) if monitor_path.exists() else []

        if not monitor:
            print(f"  WARNING: no monitor results found. TPR_mc/TNR/g-mean^2 will be None for all instances.")
        elif monitor and "arm" not in monitor[0]:
            print(f"  WARNING: monitor_results.jsonl has no 'arm' field — it predates the "
                  f"both-arms monitor run. Re-run 03_run_monitor.py.")

        result = compute_gmean2(hinted, unhinted, monitor)
        per_instance = result["per_instance"]

        if not per_instance:
            print(f"  No valid instances found. Skipping.")
            continue

        analysis_dir.mkdir(parents=True, exist_ok=True)

        # --- Per-instance CSV ---------------------------------------------------
        csv_path = analysis_dir / "results_per_instance.csv"
        fieldnames = [
            "instance_id", "hint_target", "correct_answer",
            "p1_i", "p0_i", "te_i", "r_effect_i", "level1_pass",
            "tpr_mc_i", "tnr_x0_i", "tnr_x1_i", "tnr_x1_vacuous_i", "gmean2_i",
            "acc_u_i", "acc_h_i", "n_hinted_valid", "n_unhinted_valid",
            "mean_cot_tokens_hinted", "mean_cot_tokens_unhinted",
            "mean_cot_tokens_hint_followed", "mean_cot_tokens_hint_not_followed",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(per_instance)

        # --- Per-instance id-keyed JSON maps ------------------------------------
        per_instance_json = {
            "te": {}, "r_effect": {}, "tpr_mc": {}, "tnr_x0": {}, "tnr_x1": {},
            "tnr_x1_vacuous": {}, "gmean2": {}, "level1_pass": {}, "acc_u": {},
        }
        for r in per_instance:
            iid = str(r["instance_id"])
            per_instance_json["te"][iid] = r["te_i"]
            per_instance_json["r_effect"][iid] = r["r_effect_i"]
            per_instance_json["tpr_mc"][iid] = r["tpr_mc_i"]
            per_instance_json["tnr_x0"][iid] = r["tnr_x0_i"]
            per_instance_json["tnr_x1"][iid] = r["tnr_x1_i"]
            per_instance_json["tnr_x1_vacuous"][iid] = r["tnr_x1_vacuous_i"]
            per_instance_json["gmean2"][iid] = r["gmean2_i"]
            per_instance_json["level1_pass"][iid] = r["level1_pass"]
            per_instance_json["acc_u"][iid] = r["acc_u_i"]

        per_instance_json_path = analysis_dir / "results_per_instance.json"
        with open(per_instance_json_path, "w", encoding="utf-8") as f:
            json.dump(per_instance_json, f, indent=2)

        # --- Dataset-level summary JSON -----------------------------------------
        # Runs per instance (N) = number of samples drawn per instance, taken from
        # the data as max(sample_idx)+1 across both arms.
        _sample_idxs = [r.get("sample_idx") for r in hinted + unhinted if r.get("sample_idx") is not None]
        n_samples_per_instance = (max(_sample_idxs) + 1) if _sample_idxs else None

        # Snapshot of the run-relevant config (read at analysis time; the same
        # committed config drives inference and the monitor, so this records the
        # sampling recipe and monitor model the run used).
        qi = config.get("qwen_inference", {})
        config_snapshot = {
            "n_examples": config.get("n_examples"),
            "n_samples": qi.get("n_samples"),
            "temperature": qi.get("temperature"),
            "top_p": qi.get("top_p"),
            "top_k": qi.get("top_k"),
            "max_new_tokens": qi.get("max_new_tokens"),
            "monitor_model": config.get("monitor", {}).get("model"),
        }

        summary = {
            "model_name": model_name,
            "hf_id": model_cfg["hf_id"],
            "dataset": dataset_key,
            "hint_type": hint_type,
            "n_instances_total": result["n_total"],
            "n_samples_per_instance": n_samples_per_instance,
            "config_snapshot": config_snapshot,
            **{k: v for k, v in result.items() if k != "per_instance"},
        }
        summary_path = analysis_dir / "results_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # --- Print results --------------------------------------------------------
        _n1, _nt = result['n_level1_pass'], result['n_total']
        _l1pct = f" ({100 * _n1 / _nt:.1f}%)" if _nt else ""
        print(f"  n_total={_nt}  n_level1_pass={_n1}{_l1pct}  "
              f"skip_rate={_fmt(result['skip_rate'])}")
        print(f"  Mean TE  before={_fmt(result['mean_TE_before'])}  after={_fmt(result['mean_TE_after'])}")
        print(f"  Mean R_effect  before={_fmt(result['mean_R_effect_before'])}  after={_fmt(result['mean_R_effect_after'])}")
        print(f"  g-mean^2={_fmt(result['gmean2'])}  "
              f"(n_gmean2_undefined={result['n_gmean2_undefined']})")
        print(f"  Mean TPR_mc={_fmt(result['mean_tpr_mc'])}  "
              f"Mean TNR_X0={_fmt(result['mean_tnr_x0'])}  Mean TNR_X1={_fmt(result['mean_tnr_x1'])}")
        print(f"  Written: {csv_path}")
        print(f"  Written: {per_instance_json_path}")
        print(f"  Written: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute the g-mean^2 metric.")
    parser.add_argument("--model", required=True,
                        help="HF model ID used at inference (e.g. Qwen/Qwen3-4B).")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--dataset", default=None,
                        help="Restrict to one dataset key (default: every dataset in config). "
                             "Namespaces I/O paths.")
    parser.add_argument("--hint-type", default=None, choices=HINT_TYPES,
                        help="Hint type to read (default: config hint.default). "
                             "Must match the inference/monitor run. Namespaces I/O paths.")
    parser.add_argument("--data", default="test", choices=SPLITS,
                        help="Data split to read: test (default) or eval. "
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
        run_dataset(root, config, dataset_key, models, hint_type, split=args.data)

    print("\nDone.")


if __name__ == "__main__":
    main()
