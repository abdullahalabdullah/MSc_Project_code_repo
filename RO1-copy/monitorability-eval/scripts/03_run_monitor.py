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
are comparable. Supports resume-on-restart (shared across backends).

Two interchangeable monitor backends (--backend, default: config monitor.backend):
  api    — OpenAI chat-completions (config monitor.model, e.g. gpt-5-mini),
           threaded with rate-limit back-off. Requires OPENAI_API_KEY in the env.
  local  — a local vLLM engine (see src/monitor_local.py). WHICH model runs is
           picked from the config monitor.local.models registry via
           --monitor-model (default: monitor.local.default = qwen3_8b; also
           gemma_e4b). All entries: bfloat16, greedy, 256 new tokens, seed 42.
           Requires a GPU; no API key. A NON-default --monitor-model writes to
           monitor_outputs/.../{hint_type}/{monitor_key}/ so different monitors
           never mix results or resume state.
Both run the identical monitor prompt and write the identical output schema;
every line records its `monitor_model`.

Usage:
    OPENAI_API_KEY=<key> python scripts/03_run_monitor.py --model Qwen/Qwen3-4B \
        --dataset wmdp --hint-type professor

    python scripts/03_run_monitor.py --model Qwen/Qwen3-4B \
        --dataset wmdp mmlu_pro lexam --hint-type professor --backend local

    Options:
      --config     path to config YAML (default: config/mvp_config.yaml)
      --dataset    ONE OR MORE dataset keys, space- and/or comma-separated
                   (namespace I/O paths). With --backend local the monitor model
                   is loaded once and reused across all of them.
      --hint-type  hint type to monitor (default: config hint.default; must match
                   the inference run). Namespaces I/O paths.
      --backend    monitor backend: api or local (default: config monitor.backend)
      --monitor-model  local backend only: monitor.local.models registry key
                   (default: config monitor.local.default)

Input:
    outputs/{model_name}/{dataset}/{hint_type}/hinted.jsonl
    outputs/{model_name}/{dataset}/{hint_type}/unhinted.jsonl

Output:
    monitor_outputs/{model_name}/{dataset}/{hint_type}/monitor_results.jsonl
    monitor_outputs/{model_name}/{dataset}/{hint_type}/monitor_progress.json
    (non-default --monitor-model: same two files under .../{hint_type}/{monitor_key}/)
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
from tqdm import tqdm
from src.gmean2_metrics import load_jsonl
from src.dataset import SPLITS, model_dir_name, resolve_paths
from src.hints import HINT_TYPES

NUM_WORKERS = 16
MIN_WORKERS = 8  # floor for automatic rate-limit back-off

BACKENDS = ("api", "local")


def make_output(record: dict, result: dict, monitor_model: str) -> dict:
    """Build one monitor_results.jsonl line (schema shared by both backends).

    `monitor_model` records WHICH monitor produced the line (API model id or
    local HF id), so results from different monitors are always attributable."""
    return {
        "monitor_model": monitor_model,
        "instance_id": record["instance_id"],
        "sample_idx": record["sample_idx"],
        "arm": record["arm"],
        "hint_target": record["hint_target"],
        "followed_hint": record.get("followed_hint", False),
        "no_cot": record.get("no_cot", False),
        **{k: v for k, v in result.items() if k != "raw_response"},
        "raw_response": result.get("raw_response", ""),
    }


def save_progress(progress_path: Path, done_keys: set) -> None:
    with open(progress_path, "w") as pf:
        json.dump({"done_keys": [list(k) for k in done_keys]}, pf)


def run_api_backend(pending, done_keys, out_path, progress_path,
                    config, hint_type, desc):
    """Monitor via the OpenAI chat-completions API (threaded, rate-limit-aware)."""
    from openai import OpenAI
    from src.monitor import classify

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set "
              "(required for --backend api).")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    mon_cfg = config["monitor"]
    hint_cfg = config.get("hint", {})

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
        with tqdm(total=len(pending), unit="call", desc=desc, dynamic_ncols=True) as pbar:
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                futures = {executor.submit(run_one, record): record for record in pending}
                for future in as_completed(futures):
                    try:
                        record, result = future.result()
                    except RuntimeError:
                        pbar.update(1)
                        continue

                    out_f.write(json.dumps(make_output(record, result, mon_cfg["model"]),
                                           ensure_ascii=False) + "\n")
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
                        save_progress(progress_path, done_keys)

    return total, failure_count


def run_local_backend(pending, done_keys, out_path, progress_path,
                      config, local_cfg, hint_type, desc, engine):
    """Monitor via a local vLLM engine (config monitor.local; see src/monitor_local.py).

    `engine` is a preloaded (tokenizer, llm) pair from load_local_monitor —
    loaded ONCE in main() and reused across every --dataset, so a multi-dataset
    run pays the model-load cost a single time.

    Processes `pending` in submission batches of monitor.local.batch_size —
    each batch is ONE llm.generate call, and done_keys are checkpointed after
    every batch, so a crash/restart resumes at batch granularity. GPU batching
    itself is adaptive: within a generate call vLLM's continuous-batching
    scheduler packs as many sequences as fit in the KV cache each step, so
    batch_size only needs to be comfortably larger than that concurrency
    (512 ≫ it) — it is a checkpointing knob, not a GPU-utilization knob."""
    from src.monitor_local import classify_batch

    hint_cfg = config.get("hint", {})
    batch_size = int(local_cfg.get("batch_size", 512))

    tokenizer, llm = engine

    failure_count = 0
    total = 0

    with open(out_path, "a", encoding="utf-8") as out_f:
        with tqdm(total=len(pending), unit="call", desc=desc, dynamic_ncols=True) as pbar:
            for start in range(0, len(pending), batch_size):
                batch = pending[start : start + batch_size]
                results = classify_batch(batch, tokenizer, llm, local_cfg,
                                         hint_type=hint_type, hint_cfg=hint_cfg)

                for record, result in zip(batch, results):
                    out_f.write(json.dumps(make_output(record, result, local_cfg["model"]),
                                           ensure_ascii=False) + "\n")
                    done_keys.add((record["instance_id"], record["sample_idx"], record["arm"]))
                    total += 1
                    if result.get("parse_failure"):
                        failure_count += 1
                        tqdm.write(f"  Parse failure: instance {record['instance_id']}, "
                                   f"sample {record['sample_idx']}, arm {record['arm']}")
                    pbar.set_postfix(failures=failure_count)
                    pbar.update(1)
                out_f.flush()

                save_progress(progress_path, done_keys)

    return total, failure_count


def main():
    parser = argparse.ArgumentParser(description="Run monitor (both arms) for one model.")
    parser.add_argument("--model", required=True,
                        help="HF model ID used at inference (e.g. Qwen/Qwen3-4B). "
                             "Resolves the same output dirs as step 02.")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--dataset", nargs="+", default=["mmlu_pro"],
                        help="One or more dataset keys, space- and/or comma-separated "
                             "(default: mmlu_pro). Namespaces I/O paths. With --backend "
                             "local the monitor model is loaded ONCE for all of them.")
    parser.add_argument("--hint-type", default=None, choices=HINT_TYPES,
                        help="Hint type to monitor (default: config hint.default). "
                             "Must match the inference run. Namespaces I/O paths.")
    parser.add_argument("--data", default="test", choices=SPLITS,
                        help="Data split to monitor: test (default) or eval. "
                             "Must match the inference run. Namespaces I/O paths.")
    parser.add_argument("--backend", default=None, choices=BACKENDS,
                        help="Monitor backend: 'api' (OpenAI, needs OPENAI_API_KEY) or "
                             "'local' (vLLM on a GPU, config monitor.local). "
                             "Default: config monitor.backend (api).")
    parser.add_argument("--monitor-model", default=None,
                        help="Local backend only: which config monitor.local.models "
                             "entry to run (e.g. qwen3_8b, gemma_e4b). Default: config "
                             "monitor.local.default. A NON-default choice writes to a "
                             "'{key}/' subdir of the monitor dir so monitors never mix.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    config_path = root / args.config

    with open(config_path) as f:
        config = yaml.safe_load(f)

    hint_type = args.hint_type or config.get("hint", {}).get("default", "professor")
    backend = args.backend or config.get("monitor", {}).get("backend", "api")
    if backend not in BACKENDS:
        print(f"ERROR: unknown monitor backend '{backend}' (valid: {', '.join(BACKENDS)}).")
        sys.exit(1)
    if args.monitor_model and backend != "local":
        print("ERROR: --monitor-model only applies to --backend local "
              "(the api backend's model is config monitor.model).")
        sys.exit(1)

    # Local backend: resolve WHICH local monitor runs (config monitor.local.models
    # registry). A non-default choice gets its own subdir under the monitor dir so
    # two monitors on the same run never share results or resume state.
    local_cfg = None
    mon_subdir = None
    if backend == "local":
        from src.monitor_local import resolve_local_monitor_cfg
        try:
            mon_key, local_cfg = resolve_local_monitor_cfg(config["monitor"], args.monitor_model)
        except ValueError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
        default_key = (config["monitor"].get("local", {}) or {}).get("default", "default")
        if mon_key != default_key:
            mon_subdir = mon_key

    datasets = [d for tok in args.dataset for d in tok.split(",") if d]
    model_name = model_dir_name(args.model)

    # Preflight EVERY dataset before loading any model, so a typo in the last
    # dataset key doesn't surface after minutes of engine loading / API calls.
    runs = []
    for dataset in datasets:
        paths = resolve_paths(root, config, dataset, model_name, hint_type, split=args.data)
        out_dir = paths["outputs_dir"]
        hinted_path = out_dir / "hinted.jsonl"
        unhinted_path = out_dir / "unhinted.jsonl"
        if not hinted_path.exists() or not unhinted_path.exists():
            print(f"ERROR: {hinted_path} or {unhinted_path} not found. Run 02_run_inference*.py first.")
            sys.exit(1)
        runs.append((dataset, paths, hinted_path, unhinted_path))

    monitor_id = local_cfg["model"] if backend == "local" else config["monitor"]["model"]

    # Local backend: load the monitor engine ONCE, reuse across all datasets.
    engine = None
    if backend == "local":
        from src.monitor_local import load_local_monitor
        engine = load_local_monitor(local_cfg)

    grand_total = 0
    grand_failures = 0
    for dataset, paths, hinted_path, unhinted_path in runs:
        mon_dir = paths["monitor_dir"]
        if mon_subdir:
            mon_dir = mon_dir / mon_subdir
        mon_dir.mkdir(parents=True, exist_ok=True)
        out_path = mon_dir / "monitor_results.jsonl"
        progress_path = mon_dir / "monitor_progress.json"

        # Resume: load already-processed (instance_id, sample_idx, arm) triples
        done_keys: set = set()
        if progress_path.exists():
            with open(progress_path) as f:
                stored = json.load(f).get("done_keys", [])
                done_keys = {tuple(k) for k in stored}
            print(f"[{dataset}] Resuming: {len(done_keys)} monitor calls already done.")

        hinted_records = load_jsonl(hinted_path)
        unhinted_records = load_jsonl(unhinted_path)
        all_records = hinted_records + unhinted_records

        pending = [
            r for r in all_records
            if (r["instance_id"], r["sample_idx"], r["arm"]) not in done_keys
            and not r.get("truncated", False)
            and r.get("parsed_answer") is not None
        ]
        print(f"[{dataset}] Loaded {len(hinted_records)} hinted + {len(unhinted_records)} "
              f"unhinted records for {model_name} ({len(done_keys)} already done, "
              f"{len(pending)} remaining). Backend: {backend} ({monitor_id}).")

        desc = f"{model_name}/{dataset}"
        if backend == "local":
            total, failure_count = run_local_backend(
                pending, done_keys, out_path, progress_path, config, local_cfg,
                hint_type, desc, engine)
        else:
            total, failure_count = run_api_backend(
                pending, done_keys, out_path, progress_path, config, hint_type, desc)

        # Final progress save
        save_progress(progress_path, done_keys)

        rate = failure_count / max(total, 1)
        print(
            f"\n[{dataset}] Monitor complete for {model_name} ({backend}): "
            f"{total} calls, {failure_count} parse failures ({rate:.1%})."
        )
        grand_total += total
        grand_failures += failure_count

    if len(runs) > 1:
        rate = grand_failures / max(grand_total, 1)
        print(f"\nAll {len(runs)} datasets done: {grand_total} calls, "
              f"{grand_failures} parse failures ({rate:.1%}).")


if __name__ == "__main__":
    main()
