#!/usr/bin/env python3
"""
Step 3: Run the GPT-4o monitor on all hinted outputs for one model.

Evaluates every non-truncated hinted response with the 7-question
structured monitor prompt and writes results to JSONL. Supports
resume-on-restart. Requires OPENAI_API_KEY in the environment.

Usage:
    OPENAI_API_KEY=<key> python scripts/03_run_monitor.py --model tulu_sft_no_safety

    Options:
      --config  path to config YAML (default: config/mvp_config.yaml)

Input:
    outputs/{model_name}/hinted.jsonl

Output:
    monitor_outputs/{model_name}/monitor_results.jsonl
    monitor_outputs/{model_name}/monitor_progress.json  (checkpoint state)
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from openai import OpenAI
from src.monitor import classify
from src.metrics import load_jsonl


def main():
    parser = argparse.ArgumentParser(description="Run GPT-4o monitor for one model.")
    parser.add_argument("--model", required=True,
                        help="Model name as defined in config (e.g. tulu_sft_no_safety)")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    config_path = root / args.config

    with open(config_path) as f:
        config = yaml.safe_load(f)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    mon_cfg = config["monitor"]
    model_name = args.model

    hinted_path = root / config["paths"]["outputs_dir"] / model_name / "hinted.jsonl"
    if not hinted_path.exists():
        print(f"ERROR: {hinted_path} not found. Run 02_run_inference.py first.")
        sys.exit(1)

    mon_dir = root / config["paths"]["monitor_outputs_dir"] / model_name
    mon_dir.mkdir(parents=True, exist_ok=True)
    out_path = mon_dir / "monitor_results.jsonl"
    progress_path = mon_dir / "monitor_progress.json"

    # Resume: load already-processed (instance_id, sample_idx) pairs
    done_keys: set = set()
    if progress_path.exists():
        with open(progress_path) as f:
            stored = json.load(f).get("done_keys", [])
            done_keys = {tuple(k) for k in stored}
        print(f"Resuming: {len(done_keys)} monitor calls already done.")

    hinted_records = load_jsonl(hinted_path)
    print(f"Loaded {len(hinted_records)} hinted records for {model_name}.")

    failure_count = 0
    total = 0

    with open(out_path, "a", encoding="utf-8") as out_f:
        for record in hinted_records:
            key = (record["instance_id"], record["sample_idx"])

            if key in done_keys:
                continue

            # Skip unusable records
            if record.get("truncated", False) or record.get("parsed_answer") is None:
                continue

            result = classify(
                full_cued_prompt=record["full_prompt"],
                model_response=record["raw_output"],
                hint_answer=record["hint_target"],
                client=client,
                model=mon_cfg["model"],
                temperature=mon_cfg["temperature"],
                max_retries=mon_cfg.get("max_retries", 5),
                backoff_factor=mon_cfg.get("backoff_factor", 2.0),
                timeout=mon_cfg.get("timeout", 60),
            )

            output = {
                "instance_id": record["instance_id"],
                "sample_idx": record["sample_idx"],
                "hint_target": record["hint_target"],
                "followed_hint": record.get("followed_hint", False),
                "no_cot": record.get("no_cot", False),
                **{k: v for k, v in result.items() if k != "raw_response"},
                "raw_response": result.get("raw_response", ""),
            }
            out_f.write(json.dumps(output, ensure_ascii=False) + "\n")
            out_f.flush()

            done_keys.add(key)
            total += 1

            if result.get("parse_failure"):
                failure_count += 1
                print(f"  Monitor parse failure: instance {record['instance_id']}, sample {record['sample_idx']}")

            if total % 50 == 0:
                with open(progress_path, "w") as pf:
                    json.dump({"done_keys": [list(k) for k in done_keys]}, pf)
                print(f"  Progress: {total} calls, {failure_count} parse failures.")

    # Final progress save
    with open(progress_path, "w") as pf:
        json.dump({"done_keys": [list(k) for k in done_keys]}, pf)

    rate = failure_count / max(total, 1)
    print(
        f"\nMonitor complete for {model_name}: "
        f"{total} calls, {failure_count} parse failures ({rate:.1%})."
    )


if __name__ == "__main__":
    main()
