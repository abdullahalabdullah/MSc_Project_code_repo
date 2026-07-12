"""Shared gpt-5-mini grading core for the two API stages (04 Filter-0, 08 Filter-1).

Per spec §6: extract-then-classify, with three layers of idempotence —
  1. samples with no extractable final answer never hit the API (UNPARSEABLE, written
     directly with a truncation-vs-missing-line attribution),
  2. the persistent grader cache (key = question_id | prompt_type | sha1(extracted))
     answers repeat classifications for free across runs/stages,
  3. a done-keys progress file (monitor-driver pattern) makes an interrupted grading
     run resume where it left off.
A call that fails permanently (after retries) stays un-done and is retried on the next
invocation; a call that succeeds but returns un-parseable/invalid JSON is final:
the sample is graded UNPARSEABLE (spec: "a sample whose classification ultimately
fails is UNPARSEABLE") and is not retried.
"""
import json
import os

import config
from utils import (append_kv_cache, grader_cache_key, load_kv_cache, run_concurrent)
from grader import classify_answer


def _reset_if_stale(grades_path, progress_path, stage_hash_now, force):
    """Archive grades+progress when the stage config changed (or on --force, delete).

    The persistent grader cache is deliberately NOT touched: cached classifications
    stay valid across --force re-runs (the spec's cache key covers everything the
    classifier sees except the fixed prompt skeleton, which the stage hash covers).
    """
    stored = None
    if os.path.exists(progress_path):
        try:
            with open(progress_path) as f:
                stored = json.load(f).get("config_hash")
        except (json.JSONDecodeError, OSError):
            stored = None
    if force:
        for p in (grades_path, progress_path):
            if os.path.exists(p):
                os.remove(p)
        return
    if os.path.exists(progress_path) and stored != stage_hash_now:
        print(f"Grading state is stale (config hash {stored} -> {stage_hash_now}) — "
              f"archiving grades/progress to .stale and starting fresh")
        for p in (grades_path, progress_path):
            if os.path.exists(p):
                os.replace(p, p + ".stale")


def grade_samples(items, grades_path, progress_path, client, desc, stage_hash_now,
                  map_label=None, force=False):
    """Grade a list of sample items and append one line per sample to grades_path.

    Each item is a dict with:
      key          — unique (id, sample_idx) tuple for the progress file
      meta         — dict merged into the output line (ids, direction, template, ...)
      question, gold, gold_aliases                — grading context
      extracted    — final answer text, or None (UNPARSEABLE, no API call)
      truncated    — bool (for UNPARSEABLE attribution)
      asserted     — asserted answer, or None for the bare-prompt variant
      prompt_type  — cache-key component ("bare" | "wrong_opinion" | "right_opinion")

    map_label(raw_label, item) -> final label lets the caller apply the direction-
    specific MATCHES_ASSERTED mapping (spec §6). Output line: {**meta, extracted,
    truncated, raw_label, label, unparseable_reason?, cached?}.

    Returns (n_graded_total, n_api_failures).
    """
    map_label = map_label or (lambda raw, item: raw)
    _reset_if_stale(grades_path, progress_path, stage_hash_now, force)

    done_keys = set()
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            done_keys = {tuple(k) for k in json.load(f).get("done_keys", [])}
        print(f"Resuming: {len(done_keys)} samples already graded.")

    cache = load_kv_cache(config.GRADER_CACHE_PATH)
    todo = [it for it in items if it["key"] not in done_keys]
    print(f"{len(items)} samples: {len(done_keys)} done, {len(todo)} to grade "
          f"(grader cache: {len(cache)} entries)")

    os.makedirs(os.path.dirname(grades_path), exist_ok=True)
    out_f = open(grades_path, "a", encoding="utf-8")
    progress_counter = [0]

    def _write_line(item, raw_label, label, extra=None):
        line = {
            **item["meta"],
            "extracted": item["extracted"],
            "truncated": item["truncated"],
            "raw_label": raw_label,
            "label": label,
        }
        if extra:
            line.update(extra)
        out_f.write(json.dumps(line, ensure_ascii=False) + "\n")
        out_f.flush()
        done_keys.add(item["key"])
        progress_counter[0] += 1
        if progress_counter[0] % 50 == 0:
            _save_progress()

    def _save_progress():
        with open(progress_path, "w") as pf:
            json.dump({"config_hash": stage_hash_now,
                       "done_keys": [list(k) for k in done_keys]}, pf)

    # 1) UNPARSEABLE without an API call: no final-answer line (or truncated before it).
    # 2) Cache hits: previously classified identical (qid, prompt_type, answer text).
    pending = []
    for item in todo:
        if item["extracted"] is None:
            reason = "truncated" if item["truncated"] else "no_final_answer_line"
            _write_line(item, None, "UNPARSEABLE", {"unparseable_reason": reason})
            continue
        ckey = grader_cache_key(item["meta"].get("question_id", item["key"][0]),
                                item["prompt_type"], item["extracted"])
        item["cache_key"] = ckey
        if ckey in cache:
            raw = cache[ckey]
            _write_line(item, raw, map_label(raw, item), {"cached": True})
            continue
        pending.append(item)

    print(f"{len(pending)} samples need API calls")
    n_failed = 0
    if pending:
        def call_fn(item):
            return classify_answer(client, item["question"], item["gold"],
                                   item["gold_aliases"], item["extracted"],
                                   asserted=item["asserted"])

        def on_result(item, result):
            raw = result["label"]
            if raw is None:
                # Valid API reply but not a valid JSON label — final per spec §6.
                _write_line(item, None, "UNPARSEABLE",
                            {"unparseable_reason": "classifier_output_invalid",
                             "grader_raw_response": result["raw_response"]})
                return
            append_kv_cache(config.GRADER_CACHE_PATH, item["cache_key"], raw)
            cache[item["cache_key"]] = raw
            _write_line(item, raw, map_label(raw, item))

        n_ok, n_failed = run_concurrent(pending, call_fn, on_result, desc=desc)
        if n_failed:
            print(f"WARNING: {n_failed} grader calls failed permanently — re-run this "
                  f"stage to retry them (they are not marked done).")

    _save_progress()
    out_f.close()
    return len(done_keys), n_failed


def load_grades(grades_path, key_fields):
    """Read a grades JSONL back as {key_tuple: line}, later lines winning (so a
    resumed run's re-grades supersede)."""
    grades = {}
    with open(grades_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            grades[tuple(r[k] for k in key_fields)] = r
    return grades
