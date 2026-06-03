#!/usr/bin/env python3
"""
Step 6: Extract qualitative examples for manual annotation.

For each model, selects up to 15 instances in S where the model followed
the hint but the CoT was NOT acknowledged (the core covert failure mode),
and writes them to a human-readable text file.

Usage:
    python scripts/06_extract_qualitative.py [--config ...] [--n 15]

Input:
    analysis/{model_name}/per_instance_te.csv
    analysis/{model_name}/har_summary.json
    outputs/{model_name}/hinted.jsonl
    monitor_outputs/{model_name}/monitor_results.jsonl

Output:
    plots/{model_name}_qualitative.txt   — full prompts + CoTs + monitor verdict
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.metrics import load_jsonl


SEPARATOR = "=" * 80


def load_per_instance_csv(path: Path) -> dict:
    """Return {instance_id: row_dict} for quick lookup."""
    index = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            index[int(row["instance_id"])] = {
                "te_i": float(row["te_i"]),
                "in_S": row["in_S"] == "True",
            }
    return index


def main():
    parser = argparse.ArgumentParser(description="Extract qualitative CoT examples.")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--n", type=int, default=15, help="Max examples per model")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    config_path = root / args.config

    with open(config_path) as f:
        config = yaml.safe_load(f)

    plots_dir = root / config["paths"]["plots_dir"]
    plots_dir.mkdir(parents=True, exist_ok=True)

    for model_cfg in config["models"]:
        model_name = model_cfg["name"]
        analysis_dir = root / config["paths"]["analysis_dir"] / model_name
        out_dir = root / config["paths"]["outputs_dir"] / model_name
        mon_dir = root / config["paths"]["monitor_outputs_dir"] / model_name

        csv_path = analysis_dir / "per_instance_te.csv"
        hinted_path = out_dir / "hinted.jsonl"
        monitor_path = mon_dir / "monitor_results.jsonl"

        if not csv_path.exists() or not hinted_path.exists():
            print(f"Skipping {model_name}: required files missing.")
            continue

        per_instance_index = load_per_instance_csv(csv_path)
        hinted_records = load_jsonl(hinted_path)
        monitor_records = load_jsonl(monitor_path) if monitor_path.exists() else []

        monitor_lookup = {
            (r["instance_id"], r["sample_idx"]): r
            for r in monitor_records
        }

        # Find (followed=True, acknowledged=False) runs in S
        S = {iid for iid, row in per_instance_index.items() if row["in_S"]}

        candidates = []
        seen_instances: set = set()

        for rec in hinted_records:
            iid = rec["instance_id"]
            if iid not in S:
                continue
            if rec.get("truncated") or rec.get("parsed_answer") is None:
                continue
            if not rec.get("followed_hint", False):
                continue

            mon = monitor_lookup.get((iid, rec["sample_idx"]), {})
            acknowledged = mon.get("acknowledged", False)
            if acknowledged:
                continue

            # One example per instance to keep diversity
            if iid in seen_instances:
                continue
            seen_instances.add(iid)

            candidates.append({
                "instance_id": iid,
                "sample_idx": rec["sample_idx"],
                "te_i": per_instance_index[iid]["te_i"],
                "hint_target": rec["hint_target"],
                "full_prompt": rec["full_prompt"],
                "raw_output": rec["raw_output"],
                "parsed_answer": rec["parsed_answer"],
                "monitor_label": mon.get("label", "N/A"),
                "monitor_summary": mon.get("summary", "N/A"),
                "monitor_q_answers": {
                    f"q{i}": mon.get(f"q{i}") for i in range(1, 8)
                },
            })

            if len(candidates) >= args.n:
                break

        # Sort by TE descending (most hint-influenced first)
        candidates.sort(key=lambda x: x["te_i"], reverse=True)

        out_path = plots_dir / f"{model_name}_qualitative.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"QUALITATIVE EXAMPLES: {model_name}\n")
            f.write(f"Cell: Followed Hint = True, Acknowledged = False (covert hint-following)\n")
            f.write(f"Total examples shown: {len(candidates)}\n")
            f.write(f"{SEPARATOR}\n\n")

            for idx, ex in enumerate(candidates, 1):
                f.write(f"EXAMPLE {idx}/{len(candidates)}\n")
                f.write(f"Instance ID : {ex['instance_id']}\n")
                f.write(f"Sample Idx  : {ex['sample_idx']}\n")
                f.write(f"TE_i        : {ex['te_i']:.4f}\n")
                f.write(f"Hint target : {ex['hint_target']}\n")
                f.write(f"Model answer: {ex['parsed_answer']}\n")
                f.write(f"Monitor verdict: {ex['monitor_label']}\n")
                f.write(f"Monitor Q answers: {ex['monitor_q_answers']}\n")
                f.write(f"Monitor summary: {ex['monitor_summary']}\n\n")
                f.write("--- FULL PROMPT ---\n")
                f.write(ex["full_prompt"])
                f.write("\n\n--- MODEL RESPONSE ---\n")
                f.write(ex["raw_output"])
                f.write(f"\n\n{SEPARATOR}\n\n")

        print(f"{model_name}: {len(candidates)} qualitative examples -> {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
