"""Stage D (05) — wrong-answer generation + eval-slice freeze (API, no GPU).

Generates one plausible-but-certainly-wrong answer `w` per Filter-0 survivor with
gpt-5-mini (spec §5 Stage D prompt, verbatim; no validation step — `w` is used as-is),
AND runs the same generator once over the eval slice to attach its frozen `w`,
finalizing reserved/eval_id_triviaqa.jsonl. After this stage reserved/ is frozen: no
later training stage reads it except the §8 read-only overlap assertion.

Failure policy: a pool question whose generator call ultimately fails is dropped and
logged; an eval-slice failure aborts (the frozen 1,500 must be complete — expected ~0).

Input : filter0/summary.jsonl (+ raw/training_pool.jsonl), reserved/eval_id_triviaqa.partial.jsonl
Output: filter0/wrong_answers.jsonl  {question_id, w}
        reserved/eval_id_triviaqa.jsonl (frozen; spec §4.1 schema incl. w)
Cache : wrong_answer_cache.jsonl keyed on question_id (spec §5 caching rule).
"""
import argparse

import config
import env_setup
import prompts
from grader import generate_wrong_answer, get_client
from utils import (append_kv_cache, check_stage_cache, load_kv_cache, read_jsonl,
                   run_concurrent, update_diagnostics, upstream_hash, write_jsonl,
                   write_stage_marker)

STAGE = "05_wrong_answers"


def _fill_wrong_answers(records, cache, client, desc):
    """Ensure every record has a cached `w`; returns {question_id: w} (missing = failed)."""
    pending = [r for r in records if r["question_id"] not in cache]
    print(f"{desc}: {len(records)} questions, {len(pending)} need API calls "
          f"(cache: {len(cache)} entries)")
    if pending:
        def call_fn(rec):
            return generate_wrong_answer(client, rec["question"], rec["gold"],
                                         rec["gold_aliases"])

        def on_result(rec, result):
            w = result["wrong_answer"]
            if w is None:
                print(f"  WARNING: generator returned invalid JSON for "
                      f"{rec['question_id']} — treated as failed")
                return
            append_kv_cache(config.WRONG_ANSWER_CACHE_PATH, rec["question_id"], w)
            cache[rec["question_id"]] = w

        run_concurrent(pending, call_fn, on_result, desc=desc)
    return {r["question_id"]: cache[r["question_id"]]
            for r in records if r["question_id"] in cache}


def main():
    ap = argparse.ArgumentParser(description="Stage D — wrong answers (API).")
    ap.add_argument("--force", action="store_true", help="Re-run even if cached.")
    args = ap.parse_args()

    env_setup.setup_env()
    config.ensure_output_dirs()

    params = {
        "upstream": upstream_hash("04_filter0_grade"),
        "generator_model": config.GRADER_MODEL,
        "wrong_answer_prompt": prompts.WRONG_ANSWER_PROMPT,
    }
    outputs = [config.WRONG_ANSWERS_PATH, config.EVAL_ID_PATH]
    if check_stage_cache(STAGE, params, outputs, force=args.force):
        return

    pool = {r["question_id"]: r for r in read_jsonl(config.TRAINING_POOL_PATH)}
    survivors = [pool[s["question_id"]]
                 for s in read_jsonl(config.FILTER0_SUMMARY_PATH) if s["pass"]]
    eval_records = read_jsonl(config.EVAL_ID_PARTIAL_PATH)

    # The per-question cache survives --force on purpose (spec §2: cache every
    # generator call) — `w` is frozen once generated.
    cache = load_kv_cache(config.WRONG_ANSWER_CACHE_PATH)
    client = get_client()

    pool_w = _fill_wrong_answers(survivors, cache, client, "wrong-answers-pool")
    eval_w = _fill_wrong_answers(eval_records, cache, client, "wrong-answers-eval")

    dropped = [r["question_id"] for r in survivors if r["question_id"] not in pool_w]
    if dropped:
        print(f"WARNING: {len(dropped)} pool questions dropped (generator failed): "
              f"{dropped[:5]}{' ...' if len(dropped) > 5 else ''}")

    missing_eval = [r["question_id"] for r in eval_records if r["question_id"] not in eval_w]
    if missing_eval:
        raise SystemExit(f"ERROR: {len(missing_eval)} eval-slice questions have no wrong "
                         f"answer (API failures) — the frozen eval file must be complete. "
                         f"Re-run this stage to retry.")

    write_jsonl(config.WRONG_ANSWERS_PATH,
                [{"question_id": qid, "w": w} for qid, w in pool_w.items()])
    write_jsonl(config.EVAL_ID_PATH,
                [{**r, "w": eval_w[r["question_id"]]} for r in eval_records])
    print(f"Pool wrong answers: {len(pool_w)} -> {config.WRONG_ANSWERS_PATH}")
    print(f"Frozen eval file  : {len(eval_records)} -> {config.EVAL_ID_PATH}")

    update_diagnostics(config.DIAGNOSTICS_PATH, "stage_d_wrong_answers", {
        "generator_model": config.GRADER_MODEL,
        "n_pool_survivors": len(survivors),
        "n_pool_with_w": len(pool_w),
        "n_pool_dropped": len(dropped),
        "n_eval": len(eval_records),
    })
    write_stage_marker(STAGE, params, {"pool_w": len(pool_w), "eval_w": len(eval_w)})


if __name__ == "__main__":
    main()
