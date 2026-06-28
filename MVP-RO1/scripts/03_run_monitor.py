#!/usr/bin/env python3
"""
Step 3: Run the monitor on all hinted AND unhinted outputs for one model.

For the g-mean^2 metric, the monitor must be run on both arms:
  - hinted.jsonl   (X=1, intervention) — arm="hinted"
  - unhinted.jsonl (X=0, control)      — arm="unhinted"

using the same prompt template (_MONITOR_PROMPT_TEMPLATE_2, see src/monitor.py).
The monitor is always told the instance's hint_target, even for unhinted
responses, so that TNR_X=0 / TNR_X=1 are comparable. Supports resume-on-restart.
Requires OPENAI_API_KEY in the environment.

Usage:
    OPENAI_API_KEY=<key> python scripts/03_run_monitor.py --model tulu_sft_no_safety

    Options:
      --config  path to config YAML (default: config/mvp_config.yaml)

Input:
    outputs/{model_name}/hinted.jsonl
    outputs/{model_name}/unhinted.jsonl

Output:
    monitor_outputs/{model_name}/monitor_results.jsonl
    monitor_outputs/{model_name}/monitor_progress.json  (checkpoint state)
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from openai import OpenAI
from tqdm import tqdm
from src.monitor import classify
from src.gmean2_metrics import load_jsonl
from src.dataset import resolve_paths

NUM_WORKERS = 8


def main():
    parser = argparse.ArgumentParser(description="Run monitor (both arms) for one model.")
    parser.add_argument("--model", required=True,
                        help="Model name as defined in config (e.g. tulu_sft_no_safety)")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--dataset", default="mmlu",
                        help="Dataset key (default: mmlu). Namespaces I/O paths.")
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

    paths = resolve_paths(root, config, args.dataset, model_name)
    out_dir = paths["outputs_dir"]
    hinted_path = out_dir / "hinted.jsonl"
    unhinted_path = out_dir / "unhinted.jsonl"
    if not hinted_path.exists() or not unhinted_path.exists():
        print(f"ERROR: {hinted_path} or {unhinted_path} not found. Run 02_run_inference*.py first.")
        sys.exit(1)

    mon_dir = paths["monitor_dir"]
    mon_dir.mkdir(parents=True, exist_ok=True)
    out_path = mon_dir / "monitor_results.jsonl"
    progress_path = mon_dir / "monitor_progress.json"

    # Resume: load already-processed (instance_id, sample_idx, arm) triples
    done_keys: set = set()
    if progress_path.exists():
        with open(progress_path) as f:
            stored = json.load(f).get("done_keys", [])
            done_keys = {tuple(k) for k in stored}
        print(f"Resuming: {len(done_keys)} monitor calls already done.")

    hinted_records = load_jsonl(hinted_path)
    unhinted_records = load_jsonl(unhinted_path)
    all_records = hinted_records + unhinted_records

    pending = [
        r for r in all_records
        if (r["instance_id"], r["sample_idx"], r["arm"]) not in done_keys
        and not r.get("truncated", False)
        and r.get("parsed_answer") is not None
    ]
    print(f"Loaded {len(hinted_records)} hinted + {len(unhinted_records)} unhinted records "
          f"for {model_name} ({len(done_keys)} already done, {len(pending)} remaining).")

    failure_count = 0
    total = 0

    def _classify_record(record):
        result = classify(
            model_response=record["raw_output"],
            hint_answer=record["hint_target"],
            client=client,
            model=mon_cfg["model"],
            temperature=mon_cfg.get("temperature"),
            max_retries=mon_cfg.get("max_retries", 5),
            backoff_factor=mon_cfg.get("backoff_factor", 2.0),
            timeout=mon_cfg.get("timeout", 60),
        )
        return record, result

    with open(out_path, "a", encoding="utf-8") as out_f:
        with tqdm(total=len(pending), unit="call", desc=model_name, dynamic_ncols=True) as pbar:
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                futures = {executor.submit(_classify_record, record): record for record in pending}
                for future in as_completed(futures):
                    record, result = future.result()

                    output = {
                        "instance_id": record["instance_id"],
                        "sample_idx": record["sample_idx"],
                        "arm": record["arm"],
                        "hint_target": record["hint_target"],
                        "followed_hint": record.get("followed_hint", False),
                        "no_cot": record.get("no_cot", False),
                        **{k: v for k, v in result.items() if k != "raw_response"},
                        "raw_response": result.get("raw_response", ""),
                    }
                    out_f.write(json.dumps(output, ensure_ascii=False) + "\n")
                    out_f.flush()

                    done_keys.add((record["instance_id"], record["sample_idx"], record["arm"]))
                    total += 1

                    if result.get("parse_failure"):
                        failure_count += 1
                        tqdm.write(f"  Parse failure: instance {record['instance_id']}, "
                                   f"sample {record['sample_idx']}, arm {record['arm']}")

                    pbar.set_postfix(failures=failure_count)
                    pbar.update(1)

                    if total % 50 == 0:
                        with open(progress_path, "w") as pf:
                            json.dump({"done_keys": [list(k) for k in done_keys]}, pf)

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
