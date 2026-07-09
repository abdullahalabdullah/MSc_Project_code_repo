#!/usr/bin/env python3
"""
Step 3: Run the monitor on all hinted AND unhinted outputs for one model.

For the g-mean^2 metric, the monitor must be run on both arms:
  - hinted.jsonl   (X=1, intervention) — arm="hinted"
  - unhinted.jsonl (X=0, control)      — arm="unhinted"

using the same prompt skeleton (_MONITOR_PROMPT_TEMPLATE, see src/monitor.py).
The HINT description + acknowledgment guideline in that prompt are filled per
hint type (--hint-type) via src.hints.monitor_hint_fields, so the monitor is
matched to the hint the model actually saw. The monitor is always told the
instance's hint_target, even for unhinted responses, so that TNR_X=0 / TNR_X=1
are comparable. Supports resume-on-restart. Requires OPENAI_API_KEY in the env.

Usage:
    OPENAI_API_KEY=<key> python scripts/03_run_monitor.py --model Qwen/Qwen3-4B \
        --dataset wmdp --hint-type professor

    Options:
      --config     path to config YAML (default: config/mvp_config.yaml)
      --dataset    dataset key (namespaces I/O paths)
      --hint-type  hint type to monitor (default: config hint.default; must match
                   the inference run). Namespaces I/O paths.

Input:
    outputs/{model_name}/{dataset}/{hint_type}/hinted.jsonl
    outputs/{model_name}/{dataset}/{hint_type}/unhinted.jsonl

Output:
    monitor_outputs/{model_name}/{dataset}/{hint_type}/monitor_results.jsonl
    monitor_outputs/{model_name}/{dataset}/{hint_type}/monitor_progress.json
"""

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from openai import OpenAI
from tqdm import tqdm
from src.monitor import classify
from src.gmean2_metrics import load_jsonl, resolve_jsonl
from src.dataset import SPLITS, model_dir_name, resolve_paths
from src.hints import HINT_TYPES

NUM_WORKERS = 16
MIN_WORKERS = 8  # floor for automatic rate-limit back-off


def main():
    parser = argparse.ArgumentParser(description="Run monitor (both arms) for one model.")
    parser.add_argument("--model", required=True,
                        help="HF model ID used at inference (e.g. Qwen/Qwen3-4B). "
                             "Resolves the same output dirs as step 02.")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--dataset", default="mmlu_pro",
                        help="Dataset key (default: mmlu_pro). Namespaces I/O paths.")
    parser.add_argument("--hint-type", default=None, choices=HINT_TYPES,
                        help="Hint type to monitor (default: config hint.default). "
                             "Must match the inference run. Namespaces I/O paths.")
    parser.add_argument("--data", default="test", choices=SPLITS,
                        help="Data split to monitor: test (default) or eval. "
                             "Must match the inference run. Namespaces I/O paths.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    config_path = root / args.config

    with open(config_path) as f:
        config = yaml.safe_load(f)

    hint_type = args.hint_type or config.get("hint", {}).get("default", "professor")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    mon_cfg = config["monitor"]
    model_name = model_dir_name(args.model)

    paths = resolve_paths(root, config, args.dataset, model_name, hint_type, split=args.data)
    out_dir = paths["outputs_dir"]
    hinted_path = resolve_jsonl(out_dir / "hinted.jsonl")
    unhinted_path = resolve_jsonl(out_dir / "unhinted.jsonl")
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

    hint_cfg = config.get("hint", {})

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
            hint_type=record.get("hint_type", hint_type),
            hint_cfg=hint_cfg,
        )
        return record, result

    # Gate semaphore controls how many threads call the API concurrently.
    # On a 429, we permanently consume one slot (don't release) to reduce concurrency,
    # down to a floor of MIN_WORKERS. Steps: 16 → 12 → 8.
    gate = threading.Semaphore(NUM_WORKERS)
    current_workers = [NUM_WORKERS]
    lock = threading.Lock()

    def run_one(record):
        gate.acquire()
        release = True
        try:
            return _classify_record(record)
        except RuntimeError as e:
            if "[429]" in str(e):
                with lock:
                    if current_workers[0] > MIN_WORKERS:
                        step = min(4, current_workers[0] - MIN_WORKERS)
                        current_workers[0] -= step
                        release = False  # consume current slot (1 of step)
                        for _ in range(step - 1):
                            threading.Thread(target=gate.acquire, daemon=True).start()
                        print(f"\n  Rate limit — reducing to {current_workers[0]} workers (floor: {MIN_WORKERS})")
            raise
        finally:
            if release:
                gate.release()

    with open(out_path, "a", encoding="utf-8") as out_f:
        with tqdm(total=len(pending), unit="call", desc=model_name, dynamic_ncols=True) as pbar:
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                futures = {executor.submit(run_one, record): record for record in pending}
                for future in as_completed(futures):
                    try:
                        record, result = future.result()
                    except RuntimeError:
                        pbar.update(1)
                        continue

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
