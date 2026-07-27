"""Shared helpers for the anti-sycophancy curation pipeline.

set_seed / update_diagnostics are verbatim copies of the harmlessness module's utils
(kept self-contained — no cross-module imports, per repo convention). New here:

  - JSONL read/write helpers (this module stores stage outputs as JSONL, per the spec's
    deliverable format, instead of the harmlessness module's joblib).
  - Stage-output caching (spec §2): a stage is skipped when its stage-marker exists, its
    recorded config hash matches the current config, and its outputs exist. A config
    change invalidates the marker (the stage re-runs); --force always re-runs.
  - Persistent API-call caches (spec §6): append-only JSONL key->value stores.
  - run_concurrent(): the gpt-5-mini concurrency pattern from the monitorability
    monitor driver (ThreadPoolExecutor + semaphore gate that steps concurrency down
    64 -> 16 on rate limits), factored once so both grading stages and the
    wrong-answer stage share it.
"""
import hashlib
import json
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

import config


def set_seed(seed):
    """Seed python / numpy / torch (+ HF/accelerate if available)."""
    print(f"Setting seed to {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
    try:
        from transformers import set_seed as set_seed_trf
        set_seed_trf(seed)
    except ImportError:
        pass


def update_diagnostics(path, section, payload):
    """Merge one named section into the run's consolidated diagnostics JSON.

    Each stage owns exactly one section, so re-running a stage overwrites only its slice
    and leaves the others intact. Creates the file on first write; tolerates a
    corrupt/partial file by starting fresh.
    """
    data = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            data = {}
    data[section] = payload
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── JSONL I/O ────────────────────────────────────────────────────────────────

def read_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path, record, fh=None):
    """Append one record; pass an open handle (mode 'a') for hot loops."""
    if fh is not None:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Stage-output caching (spec §2) ───────────────────────────────────────────

def stage_hash(params: dict) -> str:
    """Deterministic hash of the config values a stage consumes."""
    return hashlib.sha256(
        json.dumps(params, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()[:16]


def _marker_path(stage: str) -> str:
    return os.path.join(config.MARKERS_DIR, f"{stage}.done.json")


def upstream_hash(stage: str):
    """The recorded hash of an upstream stage's marker (chains cache invalidation:
    if an upstream stage re-ran under a new config, every downstream hash changes)."""
    path = _marker_path(stage)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f).get("config_hash")
    except (json.JSONDecodeError, OSError):
        return None


def check_stage_cache(stage: str, params: dict, outputs: list, force: bool = False) -> bool:
    """True = the stage's cached outputs are valid — SKIP the stage.

    Valid means: marker exists, its config hash matches `params`, and every path in
    `outputs` exists. On a hash mismatch the stale marker is removed (spec: a config
    change invalidates stale outputs) and the stage re-runs.
    """
    if force:
        return False
    path = _marker_path(stage)
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            marker = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    current = stage_hash(params)
    if marker.get("config_hash") != current:
        print(f"[{stage}] config changed (hash {marker.get('config_hash')} -> {current}) — re-running")
        os.remove(path)
        return False
    missing = [p for p in outputs if not os.path.exists(p)]
    if missing:
        print(f"[{stage}] marker present but outputs missing ({missing}) — re-running")
        return False
    print(f"[{stage}] cached (hash {current}) — skipping. Use --force to re-run.")
    return True


def write_stage_marker(stage: str, params: dict, counts: dict | None = None):
    os.makedirs(config.MARKERS_DIR, exist_ok=True)
    import datetime
    with open(_marker_path(stage), "w") as f:
        json.dump({
            "stage": stage,
            "config_hash": stage_hash(params),
            "params": params,
            "counts": counts or {},
            "completed_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }, f, indent=2, ensure_ascii=False, default=str)


# ── Mid-stage generation checkpoints (fingerprint-guarded, like stages 03/04 of
#    the harmlessness module) ─────────────────────────────────────────────────

def _ckpt_meta_path(ckpt_path):
    return ckpt_path + ".meta.json"


def input_fingerprint(ids, params):
    """Fingerprint of (input id list, stage config) guarding a mid-stage checkpoint."""
    h = hashlib.sha1()
    h.update(str(len(ids)).encode())
    for i in ids:
        h.update(str(i).encode() + b"\x00")
    h.update(stage_hash(params).encode())
    return h.hexdigest()


def load_ckpt(ckpt_path, fingerprint, key_field):
    """Resume map {key: record} from an interrupted generation checkpoint.

    A checkpoint whose fingerprint doesn't match the current inputs/config is stale:
    it is archived to <ckpt>.stale and ignored.
    """
    meta_path = _ckpt_meta_path(ckpt_path)
    if not os.path.exists(ckpt_path):
        return {}
    stale = True
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                stale = json.load(f).get("fingerprint") != fingerprint
        except (json.JSONDecodeError, OSError):
            stale = True
    if stale:
        print(f"Stale checkpoint {ckpt_path} — archiving to .stale and starting fresh")
        os.replace(ckpt_path, ckpt_path + ".stale")
        if os.path.exists(meta_path):
            os.replace(meta_path, meta_path + ".stale")
        return {}
    done = {}
    with open(ckpt_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn final line from an interrupt — that item re-runs
            done[r[key_field]] = r
    print(f"Resuming from checkpoint: {len(done)} items already generated ({ckpt_path})")
    return done


def init_ckpt(ckpt_path, fingerprint):
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    with open(_ckpt_meta_path(ckpt_path), "w") as f:
        json.dump({"fingerprint": fingerprint}, f)
    return open(ckpt_path, "a", encoding="utf-8")


def finalize_ckpt(ckpt_path):
    """Delete the checkpoint after the stage's final output is safely written."""
    for p in (ckpt_path, _ckpt_meta_path(ckpt_path)):
        if os.path.exists(p):
            os.remove(p)


# ── Persistent API-call caches (spec §6) ─────────────────────────────────────

def load_kv_cache(path):
    """{key: value} from an append-only JSONL cache (later lines win)."""
    cache = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    cache[r["key"]] = r["value"]
                except (json.JSONDecodeError, KeyError):
                    continue
    return cache


def append_kv_cache(path, key, value):
    append_jsonl(path, {"key": key, "value": value})


def grader_cache_key(question_id, prompt_type, extracted):
    """Spec §6 cache key: (question_id, prompt-type, sha1(extracted answer))."""
    sha = hashlib.sha1(str(extracted).encode("utf-8")).hexdigest()
    return f"{question_id}|{prompt_type}|{sha}"


# ── Concurrent API driver (monitorability 03_run_monitor.py pattern) ─────────

def run_concurrent(pending, call_fn, on_result, desc="api",
                   num_workers=None, min_workers=None):
    """Run call_fn over `pending` with a self-throttling thread pool.

    call_fn(item) -> result, and must raise RuntimeError with "[429]" in the message
    when it exhausted retries on rate limits (grader.py does this). on_result(item,
    result) runs in the main thread as results complete — do file writes there.

    On a 429-tagged failure one worker slot is permanently consumed (stepping
    concurrency down 4 at a time to the MIN_WORKERS floor), exactly like the
    monitorability monitor driver. Failed items are counted and left un-done, so a
    re-run (resume) retries only them. Returns (n_ok, n_failed).
    """
    num_workers = num_workers or config.NUM_WORKERS
    min_workers = min_workers or config.MIN_WORKERS

    gate = threading.Semaphore(num_workers)
    current_workers = [num_workers]
    lock = threading.Lock()

    def run_one(item):
        gate.acquire()
        release = True
        try:
            return item, call_fn(item)
        except RuntimeError as e:
            if "[429]" in str(e):
                with lock:
                    if current_workers[0] > min_workers:
                        step = min(4, current_workers[0] - min_workers)
                        current_workers[0] -= step
                        release = False  # consume current slot (1 of step)
                        for _ in range(step - 1):
                            threading.Thread(target=gate.acquire, daemon=True).start()
                        print(f"\n  Rate limit — reducing to {current_workers[0]} workers "
                              f"(floor: {min_workers})")
            raise
        finally:
            if release:
                gate.release()

    n_ok = n_failed = 0
    with tqdm(total=len(pending), unit="call", desc=desc, dynamic_ncols=True) as pbar:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(run_one, item): item for item in pending}
            for future in as_completed(futures):
                try:
                    item, result = future.result()
                except RuntimeError as e:
                    n_failed += 1
                    tqdm.write(f"  API call failed permanently: {e}")
                    pbar.update(1)
                    continue
                on_result(item, result)
                n_ok += 1
                pbar.update(1)
    return n_ok, n_failed
