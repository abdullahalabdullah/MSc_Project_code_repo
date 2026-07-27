#!/usr/bin/env python3
"""
Step 9: Generate all Markdown reports for one model from steps 04 + 08's artifacts.

ALL .md generation is isolated here + src/reporting.py (see that module's
docstring) so it can be removed cleanly for a public release. Every report is
derived from machine-readable files (results_summary.json,
results_per_instance.csv, cot_token_stats.json), so deleting all .md files and
re-running this script fully regenerates them — no inference/monitor re-run.

Scans analysis/{model}/{dataset}/{hint}/results_summary.json for every
completed (dataset, hint-type) run and writes:

  analysis/{model}/{dataset}/{hint}/results_summary.md    per-run report
  analysis/{model}/results_overall_summary.md             section per (dataset × hint)
  analysis/{model}/results_dataset_level_summary.md       pooled across hint types
  analysis/{model}/results_hint_type_level_summary.md     pooled across datasets
  analysis/{model}/results_aggregate_summary.md           pooled across everything
  analysis/{model}/cot_overall_summary.md                 CoT mirror of the above
  analysis/{model}/cot_dataset_level_summary.md             (needs step 08's
  analysis/{model}/cot_hint_type_level_summary.md            cot_token_stats.json)
  analysis/{model}/cot_aggregate_summary.md

Usage:
    python scripts/09_generate_reports.py --model Qwen/Qwen3-4B
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from src.dataset import SPLITS, model_dir_name
from src.hints import HINT_TYPES
from src.reporting import (
    discover_runs,
    load_run,
    write_cot_aggregate_summary,
    write_cot_level_summary,
    write_cot_overall_summary,
    write_results_aggregate_summary,
    write_results_level_summary,
    write_results_overall_summary,
    write_results_summary_md,
)


def main():
    parser = argparse.ArgumentParser(description="Generate all Markdown reports for one model.")
    parser.add_argument("--model", required=True,
                        help="HF model ID used at inference (e.g. Qwen/Qwen3-4B).")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--data", default="test", choices=SPLITS,
                        help="Data split to report on: test (default) or eval. Eval "
                             "roll-ups are written under analysis/{model}/eval/.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    with open(root / args.config) as f:
        config = yaml.safe_load(f)

    model_name = model_dir_name(args.model)
    model_root = root / config["paths"]["analysis_dir"] / model_name
    # Roll-up .md files go next to the per-run dirs: analysis/{model}/ for the
    # test split, analysis/{model}/eval/ for the eval split (no dataset is
    # named "eval", so the two never collide).
    summary_root = model_root if args.data == "test" else model_root / "eval"
    level2_threshold = config.get("gmean2", {}).get("level2_te_threshold", 0.05)

    run_dirs = discover_runs(model_root, list(config.get("datasets", {})), HINT_TYPES,
                             split=args.data)
    if not run_dirs:
        print(f"No results_summary.json files found under {model_root} "
              f"(split: {args.data}) — run 04_compute_gmean2.py first.")
        sys.exit(1)
    summary_root.mkdir(parents=True, exist_ok=True)

    # {(dataset, hint): {"summary", "rows", "cot"}}, in registry order.
    runs = {}
    for key, analysis_dir in run_dirs.items():
        run = load_run(analysis_dir)
        if run is None:
            print(f"  Skipping {key}: incomplete artifacts in {analysis_dir}")
            continue
        runs[key] = run
        md_path = analysis_dir / "results_summary.md"
        write_results_summary_md(run["summary"], level2_threshold, md_path)
        print(f"  Written: {md_path}")

    if not runs:
        print("No complete runs found.")
        sys.exit(1)

    hf_id = next(iter(runs.values()))["summary"].get("hf_id", args.model)

    # --- Results roll-ups -----------------------------------------------------
    path = summary_root / "results_overall_summary.md"
    write_results_overall_summary(runs, model_name, hf_id, path)
    print(f"  Written: {path}")
    for filename, level in [("results_dataset_level_summary.md", "dataset"),
                            ("results_hint_type_level_summary.md", "hint_type")]:
        path = summary_root / filename
        write_results_level_summary(runs, model_name, hf_id, path, level)
        print(f"  Written: {path}")
    path = summary_root / "results_aggregate_summary.md"
    write_results_aggregate_summary(runs, model_name, hf_id, path)
    print(f"  Written: {path}")

    # --- CoT token-length roll-ups (need step 08's cot_token_stats.json) -------
    cot_runs = {key: run["cot"] for key, run in runs.items() if run["cot"] is not None}
    if cot_runs:
        path = summary_root / "cot_overall_summary.md"
        write_cot_overall_summary(cot_runs, model_name, hf_id, path)
        print(f"  Written: {path}")
        for filename, level in [("cot_dataset_level_summary.md", "dataset"),
                                ("cot_hint_type_level_summary.md", "hint_type")]:
            path = summary_root / filename
            write_cot_level_summary(cot_runs, model_name, hf_id, path, level)
            print(f"  Written: {path}")
        path = summary_root / "cot_aggregate_summary.md"
        write_cot_aggregate_summary(cot_runs, model_name, hf_id, path)
        print(f"  Written: {path}")
    else:
        print("  No cot_token_stats.json found — skipping CoT reports (run 08_cot_token_analysis.py).")

    print("\nDone.")


if __name__ == "__main__":
    main()
