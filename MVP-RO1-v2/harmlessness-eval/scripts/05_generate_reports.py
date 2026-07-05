#!/usr/bin/env python3
"""
Step 5 (harmlessness): Generate the Markdown report from step 04's metrics.json.

ALL .md generation is isolated here + src/reporting.py so it can be removed
cleanly for a public release. Derives only from metrics.json, so the .md can be
deleted and regenerated with one run (no inference/classifier re-run).

Usage:
    python harmlessness-eval/scripts/05_generate_reports.py --model Qwen/Qwen3-4B

Output:
    outputs-and-results/harmlessness/analysis/{model}/harmlessness_summary.md
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from src.dataset import model_dir_name
from src.reporting import load_metrics, write_summary_md

REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="Generate the harmlessness Markdown report.")
    parser.add_argument("--model", required=True,
                        help="HF model ID used at inference (e.g. Qwen/Qwen3-4B).")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    with open(config_path) as f:
        config = yaml.safe_load(f)

    h = config["harmlessness"]
    model_name = model_dir_name(args.model)
    analysis_dir = REPO_ROOT / h["paths"]["analysis_dir"] / model_name
    metrics_path = analysis_dir / "metrics.json"
    if not metrics_path.exists():
        print(f"ERROR: metrics not found at {metrics_path}. Run 04_compute_metrics.py first.")
        sys.exit(1)

    summary = load_metrics(metrics_path)
    meta = {
        "model_name": summary.get("model_name"),
        "hf_id": summary.get("hf_id"),
        "benchmark": summary.get("benchmark"),
        "classifier": summary.get("classifier"),
        **summary.get("config_snapshot", {}),
    }
    out_path = analysis_dir / "harmlessness_summary.md"
    write_summary_md(summary, meta, out_path)
    print(f"  Written: {out_path}")


if __name__ == "__main__":
    main()
