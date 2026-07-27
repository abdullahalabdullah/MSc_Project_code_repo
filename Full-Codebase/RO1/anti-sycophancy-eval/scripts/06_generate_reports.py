#!/usr/bin/env python3
"""
Step 6: Markdown reports (CPU). ALL .md generation for this eval lives here +
src/reporting.py (delete both + the driver lines to strip reports for release).

Two modes:
    python scripts/06_generate_reports.py --model Qwen/Qwen3-4B
        -> analysis/{model}/antisyc_summary.md            (from metrics.json)
    python scripts/06_generate_reports.py --base Qwen/Qwen3-4B --trained <trained-id>
        -> analysis/compare/{base}__vs__{trained}/antisyc_comparison.md
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from src.dataset import model_dir_name, resolve_paths
from src.reporting import write_comparison_md, write_summary_md


def main():
    parser = argparse.ArgumentParser(description="Render anti-sycophancy .md reports.")
    parser.add_argument("--model", default=None, help="Single-model summary mode.")
    parser.add_argument("--base", default=None, help="Comparison mode (with --trained).")
    parser.add_argument("--trained", default=None)
    parser.add_argument("--config", default="config/mvp_config.yaml")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    with open(root / args.config) as f:
        config = yaml.safe_load(f)

    if args.model:
        model_name = model_dir_name(args.model)
        paths = resolve_paths(root, config, model_name)
        m_path = paths["analysis_dir"] / "metrics.json"
        if not m_path.exists():
            sys.exit(f"ERROR: {m_path} not found — run 04_compute_metrics.py first.")
        with open(m_path) as f:
            metrics = json.load(f)
        out = paths["analysis_dir"] / "antisyc_summary.md"
        write_summary_md(metrics, model_name, out)
        print(f"-> {out}")
    elif args.base and args.trained:
        base_name = model_dir_name(args.base)
        trained_name = model_dir_name(args.trained)
        paths = resolve_paths(root, config)
        cmp_dir = paths["compare_dir"] / f"{base_name}__vs__{trained_name}"
        c_path = cmp_dir / "comparison.json"
        if not c_path.exists():
            sys.exit(f"ERROR: {c_path} not found — run 05_compare_models.py first.")
        with open(c_path) as f:
            comparison = json.load(f)
        out = cmp_dir / "antisyc_comparison.md"
        write_comparison_md(comparison, out)
        print(f"-> {out}")
    else:
        sys.exit("ERROR: pass either --model, or --base and --trained.")


if __name__ == "__main__":
    main()
