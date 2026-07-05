#!/usr/bin/env python3
"""
Step 4 (harmlessness): Compute Safety Rate + Refusal Rate from WildGuard verdicts.

Machine-readable only (JSON + per-prompt CSV); no Markdown. The .md report is
generated separately by scripts/05_generate_reports.py (kept apart so all .md
generation can be removed cleanly for a public release). Metrics are computed
overall and per subset (adversarial_harmful vs adversarial_benign).

Usage:
    python harmlessness-eval/scripts/04_compute_metrics.py --model Qwen/Qwen3-4B

Input:
    outputs-and-results/harmlessness/wildguard/{model}/classifications.jsonl
Output (under outputs-and-results/harmlessness/analysis/{model}/):
    metrics.json        — overall + per-subset safety/refusal rates + config snapshot
    per_prompt.csv      — one row per classified response
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from src.dataset import model_dir_name
from src.metrics import compute_metrics

REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="Compute harmlessness metrics for one model.")
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

    cls_path = REPO_ROOT / h["paths"]["wildguard_dir"] / model_name / "classifications.jsonl"
    if not cls_path.exists():
        print(f"ERROR: classifications not found at {cls_path}. Run 03_run_wildguard.py first.")
        sys.exit(1)

    classifications = []
    with open(cls_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                classifications.append(json.loads(line))
    if not classifications:
        print("ERROR: no classifications to score.")
        sys.exit(1)

    metrics = compute_metrics(classifications)

    qi = config.get("qwen_inference", {})
    summary = {
        "model_name": model_name,
        "hf_id": args.model,
        "benchmark": h["benchmark"]["hf"],
        "classifier": h["classifier"]["hf_id"],
        "config_snapshot": {
            "enable_thinking": h["generation"].get("enable_thinking"),
            "n_samples": h["generation"].get("n_samples"),
            "classify_target": h.get("classify_target"),
            "temperature": qi.get("temperature"),
            "top_p": qi.get("top_p"),
            "top_k": qi.get("top_k"),
            "max_new_tokens": qi.get("max_new_tokens"),
        },
        **metrics,
    }

    out_dir = REPO_ROOT / h["paths"]["analysis_dir"] / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    csv_path = out_dir / "per_prompt.csv"
    fieldnames = ["prompt_id", "subset", "sample_idx", "prompt_harmful", "refusal", "response_harmful"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(classifications)

    ov = metrics["overall"]
    print(f"  Overall: safety_rate={ov['safety_rate']}  refusal_rate={ov['refusal_rate']}  n={ov['n']}")
    for subset, m in metrics["by_subset"].items():
        print(f"    {subset}: safety={m['safety_rate']}  refusal={m['refusal_rate']}  n={m['n']}")
    print(f"  Written: {metrics_path}")
    print(f"  Written: {csv_path}")


if __name__ == "__main__":
    main()
