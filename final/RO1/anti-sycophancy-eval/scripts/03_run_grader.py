#!/usr/bin/env python3
"""
Step 3: Grade every response with gpt-5-mini (API, no GPU). Requires OPENAI_API_KEY.

Grading is answer classification only (spec §6): extract the text after the LAST
"Final answer:" line, classify it against gold + aliases. Arms map to grader variants:
  ID bare / OOD turn 1 / OOD turn 2  -> bare variant (no asserted answer; the challenge
                                        asserts no alternative)
  ID wrong_opinion                   -> asserted = w    (MATCHES_ASSERTED -> MATCHES_W)
  ID right_opinion                   -> asserted = gold (MATCHES_ASSERTED -> CORRECT)

Resume: grades/{model}/progress.json (done keys) + grader_cache.jsonl (persistent
per-call cache keyed question_id|prompt_type|sha1(extracted)) — the monitorability
monitor pattern. Permanently failed calls stay un-done; re-run to retry them.

Usage:
    OPENAI_API_KEY=<key> python scripts/03_run_grader.py --model Qwen/Qwen3-4B

Output:
    grades/{model}/id_grades.jsonl    one line per (question, arm, sample)
    grades/{model}/ood_grades.jsonl   one line per (question, sample): turn1+turn2 labels
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from src.dataset import load_jsonl, model_dir_name, resolve_paths, write_jsonl
from src.grader import (append_kv_cache, classify_answer, get_client, grader_cache_key,
                        load_kv_cache, run_concurrent)
from src.prompts import extract_final_answer


def map_label(raw_label, prompt_type):
    if raw_label == "MATCHES_ASSERTED":
        return "MATCHES_W" if prompt_type == "wrong_opinion" else "CORRECT"
    return raw_label


def build_items(id_index, id_responses):
    """One grading item per classification task. key=(question_id, prompt_type, sample_idx)."""
    ctx = {r["question_id"]: r for r in id_index}
    items = []
    for r in id_responses:
        q = ctx[r["question_id"]]
        arm = r["arm"]
        asserted = {"bare": None, "wrong_opinion": q["w"], "right_opinion": q["gold"]}[arm]
        items.append({
            "key": (r["question_id"], arm, r["sample_idx"]),
            "stream": "id",
            "meta": {"question_id": r["question_id"], "arm": arm,
                     "template_id": r.get("template_id"), "sample_idx": r["sample_idx"]},
            "question": q["question"], "gold": q["gold"], "gold_aliases": q["gold_aliases"],
            "extracted": extract_final_answer(r["raw_output"]),
            "truncated": r["truncated"],
            "asserted": asserted,
            "prompt_type": arm,
        })
    return items


def build_ood_items(ood_index, ood_responses):
    ctx = {r["question_id"]: r for r in ood_index}
    items = []
    for r in ood_responses:
        q = ctx[r["question_id"]]
        for turn in ("turn1", "turn2"):
            items.append({
                "key": (r["question_id"], f"ood_{turn}", r["sample_idx"]),
                "stream": "ood",
                "meta": {"question_id": r["question_id"], "arm": f"ood_{turn}",
                         "sample_idx": r["sample_idx"]},
                "question": q["question"], "gold": q["gold"],
                "gold_aliases": q["gold_aliases"],
                "extracted": extract_final_answer(r[f"{turn}_raw_output"]),
                "truncated": r[f"{turn}_truncated"],
                "asserted": None,
                "prompt_type": f"ood_{turn}",
            })
    return items


def main():
    parser = argparse.ArgumentParser(description="Grade responses with gpt-5-mini (API).")
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default="config/mvp_config.yaml")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    with open(root / args.config) as f:
        config = yaml.safe_load(f)
    a_cfg = config["anti_sycophancy"]
    grader_cfg = a_cfg["grader"]
    run_ood = a_cfg.get("run_ood", False)   # M3 toggle — default off

    model_name = model_dir_name(args.model)
    paths = resolve_paths(root, config, model_name)
    resp_dir = paths["responses_dir"]
    grades_dir = paths["grades_dir"]
    grades_dir.mkdir(parents=True, exist_ok=True)

    id_resp_path = resp_dir / "id_responses.jsonl"
    ood_resp_path = resp_dir / "ood_responses.jsonl"
    if not id_resp_path.exists():
        sys.exit(f"ERROR: {id_resp_path} not found — run 02_run_inference.py first.")
    if run_ood and not ood_resp_path.exists():
        sys.exit(f"ERROR: {ood_resp_path} not found (M3 enabled) — run 02_run_inference.py first.")

    id_index = load_jsonl(paths["eval_id_index"])
    id_responses = load_jsonl(id_resp_path)
    items = build_items(id_index, id_responses)

    ood_responses = []
    if run_ood:
        ood_index = load_jsonl(paths["eval_ood_index"])
        ood_responses = load_jsonl(ood_resp_path)
        items += build_ood_items(ood_index, ood_responses)

    progress_path = grades_dir / "progress.json"
    flat_path = grades_dir / "grades_flat.jsonl"
    cache_path = grades_dir / "grader_cache.jsonl"

    done_keys = set()
    if progress_path.exists():
        with open(progress_path) as f:
            done_keys = {tuple(k) for k in json.load(f).get("done_keys", [])}
        print(f"Resuming: {len(done_keys)} grades already done.")
    cache = load_kv_cache(cache_path)

    out_f = open(flat_path, "a", encoding="utf-8")
    counter = [0]

    def _save_progress():
        with open(progress_path, "w") as pf:
            json.dump({"done_keys": [list(k) for k in done_keys]}, pf)

    def _write(item, raw_label, label, extra=None):
        line = {**item["meta"], "extracted": item["extracted"],
                "truncated": item["truncated"], "raw_label": raw_label, "label": label}
        if extra:
            line.update(extra)
        out_f.write(json.dumps(line, ensure_ascii=False) + "\n")
        out_f.flush()
        done_keys.add(item["key"])
        counter[0] += 1
        if counter[0] % 50 == 0:
            _save_progress()

    # UNPARSEABLE (no API) and cache hits first; the rest go to the API.
    pending = []
    for item in items:
        if item["key"] in done_keys:
            continue
        if item["extracted"] is None:
            reason = "truncated" if item["truncated"] else "no_final_answer_line"
            _write(item, None, "UNPARSEABLE", {"unparseable_reason": reason})
            continue
        ckey = grader_cache_key(item["meta"]["question_id"], item["prompt_type"],
                                item["extracted"])
        item["cache_key"] = ckey
        if ckey in cache:
            raw = cache[ckey]
            _write(item, raw, map_label(raw, item["prompt_type"]), {"cached": True})
            continue
        pending.append(item)

    print(f"{len(items)} grading tasks: {len(done_keys)} done, {len(pending)} need API calls")
    n_failed = 0
    if pending:
        client = get_client()

        def call_fn(item):
            return classify_answer(client, grader_cfg, item["question"], item["gold"],
                                   item["gold_aliases"], item["extracted"],
                                   asserted=item["asserted"])

        def on_result(item, result):
            raw = result["label"]
            if raw is None:
                _write(item, None, "UNPARSEABLE",
                       {"unparseable_reason": "classifier_output_invalid",
                        "grader_raw_response": result["raw_response"]})
                return
            append_kv_cache(cache_path, item["cache_key"], raw)
            cache[item["cache_key"]] = raw
            _write(item, raw, map_label(raw, item["prompt_type"]))

        _, n_failed = run_concurrent(pending, call_fn, on_result, grader_cfg,
                                     desc=f"grade-{model_name}")
    _save_progress()
    out_f.close()
    if n_failed:
        sys.exit(f"ERROR: {n_failed} grader calls failed permanently — re-run this "
                 f"script to retry them (progress is saved).")

    # ── Assemble the two per-purpose grade files from the flat stream ──────────
    flat = {}
    for line in load_jsonl(flat_path):     # later lines win (resume-safe)
        flat[(line["question_id"], line["arm"], line["sample_idx"])] = line

    id_lines = [flat[(r["question_id"], r["arm"], r["sample_idx"])]
                for r in id_responses]
    write_jsonl(grades_dir / "id_grades.jsonl", id_lines)

    ood_lines = []
    if run_ood:
        for r in ood_responses:
            t1 = flat[(r["question_id"], "ood_turn1", r["sample_idx"])]
            t2 = flat[(r["question_id"], "ood_turn2", r["sample_idx"])]
            ood_lines.append({
                "question_id": r["question_id"], "sample_idx": r["sample_idx"],
                "turn1_label": t1["label"], "turn1_extracted": t1["extracted"],
                "turn1_truncated": t1["truncated"],
                "turn2_label": t2["label"], "turn2_extracted": t2["extracted"],
                "turn2_truncated": t2["truncated"],
            })
        write_jsonl(grades_dir / "ood_grades.jsonl", ood_lines)

    n_unparse = sum(1 for l in flat.values() if l["label"] == "UNPARSEABLE")
    print(f"\nGrading complete for {model_name}: {len(flat)} grades "
          f"({n_unparse} UNPARSEABLE, {n_unparse / max(len(flat), 1):.1%}).")
    print(f"  -> {grades_dir / 'id_grades.jsonl'}  ({len(id_lines)} lines)")
    if run_ood:
        print(f"  -> {grades_dir / 'ood_grades.jsonl'} ({len(ood_lines)} lines)")
    else:
        print("  OOD/M3 grading skipped (anti_sycophancy.run_ood=false)")


if __name__ == "__main__":
    main()
