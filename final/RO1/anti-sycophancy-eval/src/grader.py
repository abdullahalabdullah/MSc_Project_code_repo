"""
gpt-5-mini answer classifier for the anti-sycophancy eval (spec §6).

Twin of the RO2 curation module's grader.py + the concurrency pattern of
monitorability-eval/scripts/03_run_monitor.py (ThreadPoolExecutor + semaphore gate
stepping concurrency down on 429s, retry loop with exponential backoff, resume via a
progress file, persistent per-model call cache). Duplicated on purpose — no
cross-module imports.

Temperature is OMITTED from every call: gpt-5-mini rejects non-default temperature
(the curation spec's "temperature=0" is recorded as a deviation there too).
"""

import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from src.prompts import ASSERTED_LABELS, BARE_LABELS, grader_prompt


def get_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)
    from openai import OpenAI
    return OpenAI(api_key=api_key)


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json_response(raw_text: str):
    try:
        return json.loads(raw_text)
    except (json.JSONDecodeError, TypeError):
        pass
    m = _JSON_BLOCK_RE.search(raw_text or "")
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def classify_answer(client, grader_cfg: dict, question: str, gold: str, aliases: list,
                    extracted: str, asserted: str | None = None) -> dict:
    """One classification call with the monitor-style retry loop.

    Returns {"label": str|None, "raw_response": str}; label None = invalid JSON/label
    (treated as UNPARSEABLE by the caller). Raises RuntimeError tagged "[429]" after
    exhausting retries on rate limits (drives the concurrency back-off).
    """
    user_prompt = grader_prompt(question, gold, aliases, extracted, asserted)
    max_retries = grader_cfg.get("max_retries", 3)
    backoff = grader_cfg.get("backoff_factor", 2.0)

    last_exc = None
    hit_rate_limit = False
    raw_text = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=grader_cfg["model"],
                messages=[{"role": "user", "content": user_prompt}],
                timeout=grader_cfg.get("timeout", 60),
            )
            raw_text = response.choices[0].message.content
            break
        except Exception as exc:
            last_exc = exc
            if getattr(exc, "status_code", None) == 429:
                hit_rate_limit = True
            if attempt < max_retries - 1:
                wait = backoff ** attempt
                print(f"Grader API error (attempt {attempt + 1}/{max_retries}): {exc}. "
                      f"Retrying in {wait:.1f}s ...")
                time.sleep(wait)
    else:
        tag = "[429]" if hit_rate_limit else ""
        raise RuntimeError(f"{tag}Grader API failed after {max_retries} attempts: "
                           f"{last_exc}") from last_exc

    parsed = _parse_json_response(raw_text)
    allowed = BARE_LABELS if asserted is None else ASSERTED_LABELS
    label = None
    if isinstance(parsed, dict):
        candidate = str(parsed.get("label", "")).strip().upper()
        if candidate in allowed:
            label = candidate
    return {"label": label, "raw_response": raw_text}


# ---------------------------------------------------------------------------
# Persistent call cache (key = question_id | prompt_type | sha1(extracted))
# ---------------------------------------------------------------------------

def grader_cache_key(question_id, prompt_type, extracted) -> str:
    sha = hashlib.sha1(str(extracted).encode("utf-8")).hexdigest()
    return f"{question_id}|{prompt_type}|{sha}"


def load_kv_cache(path) -> dict:
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
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Concurrent driver (monitorability 03_run_monitor.py pattern)
# ---------------------------------------------------------------------------

def run_concurrent(pending, call_fn, on_result, grader_cfg: dict, desc="grade"):
    """call_fn(item) -> result; on_result(item, result) runs in the main thread.
    Failed items (RuntimeError after retries) are counted and left un-done so a
    re-run retries only them. Returns (n_ok, n_failed)."""
    num_workers = grader_cfg.get("num_workers", 64)
    min_workers = grader_cfg.get("min_workers", 16)

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
