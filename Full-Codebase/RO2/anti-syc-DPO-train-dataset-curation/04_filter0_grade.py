"""Stage C grading (04) — Filter-0 classification + pass/fail (API, no GPU).

Extracts each sample's final answer (spec §6) and classifies it with gpt-5-mini using
the bare-prompt grader variant. A question passes Filter-0 iff ≥ 9/10 samples are
parseable AND ≥ 70% of parseable samples are CORRECT (spec §2/§5).

Input : filter0/generations.jsonl + raw/training_pool.jsonl
Output: filter0/grades.jsonl    one line per sample
        filter0/summary.jsonl   one line per question: n_parseable, n_correct,
                                consistency, pass, consistency_stratum
Resume: filter0/grade_progress.json + the persistent grader cache.
"""
import argparse

import config
import env_setup
import prompts
from grader import get_client
from grading import grade_samples, load_grades
from utils import (check_stage_cache, read_jsonl, stage_hash, update_diagnostics,
                   upstream_hash, write_jsonl, write_stage_marker)

STAGE = "04_filter0_grade"


def consistency_stratum(consistency):
    """Reporting stratum (spec §5 Stage C); None below the pass band."""
    if consistency is None or consistency < config.FILTER0_MIN_CONSISTENCY:
        return None
    for lo, name in reversed(config.CONSISTENCY_STRATA):
        if consistency >= lo:
            # exact 100% gets its own stratum; [0.9, 1.0) is "90-99"
            if name == "100" and consistency < 1.0:
                continue
            return name
    return None


def main():
    ap = argparse.ArgumentParser(description="Stage C — Filter-0 grading (API).")
    ap.add_argument("--force", action="store_true", help="Re-run even if cached.")
    args = ap.parse_args()

    env_setup.setup_env()
    config.ensure_output_dirs()

    params = {
        "upstream": upstream_hash("03_filter0_generate"),
        "grader_model": config.GRADER_MODEL,
        "grader_prompt_bare": prompts.GRADER_PROMPT_BARE,
        "min_parseable": config.FILTER0_MIN_PARSEABLE,
        "min_consistency": config.FILTER0_MIN_CONSISTENCY,
    }
    outputs = [config.FILTER0_GRADES_PATH, config.FILTER0_SUMMARY_PATH]
    if check_stage_cache(STAGE, params, outputs, force=args.force):
        return

    pool = {r["question_id"]: r for r in read_jsonl(config.TRAINING_POOL_PATH)}
    generations = read_jsonl(config.FILTER0_GENERATIONS_PATH)
    print(f"Loaded {len(generations)} questions x samples")

    items = []
    for rec in generations:
        q = pool[rec["question_id"]]
        for s_idx, sample in enumerate(rec["samples"]):
            items.append({
                "key": (rec["question_id"], s_idx),
                "meta": {"question_id": rec["question_id"], "sample_idx": s_idx,
                         "prompt_type": "bare"},
                "question": q["question"],
                "gold": q["gold"],
                "gold_aliases": q["gold_aliases"],
                "extracted": prompts.extract_final_answer(sample["full_text"]),
                "truncated": sample["truncated"],
                "asserted": None,
                "prompt_type": "bare",
            })

    client = get_client()
    grade_samples(items, config.FILTER0_GRADES_PATH, config.FILTER0_GRADE_PROGRESS_PATH,
                  client, desc="filter0-grade", stage_hash_now=stage_hash(params),
                  force=args.force)

    # Per-question summary + pass/fail (spec pass band).
    grades = load_grades(config.FILTER0_GRADES_PATH, ("question_id", "sample_idx"))
    summaries = []
    n_pass = 0
    incomplete = 0
    for rec in generations:
        qid = rec["question_id"]
        labels = []
        for s_idx in range(len(rec["samples"])):
            g = grades.get((qid, s_idx))
            labels.append(g["label"] if g else None)
        if any(l is None for l in labels):
            incomplete += 1
            continue  # permanently failed API calls — re-run this stage to retry
        parseable = [l for l in labels if l != "UNPARSEABLE"]
        n_correct = sum(1 for l in parseable if l == "CORRECT")
        consistency = (n_correct / len(parseable)) if parseable else None
        passed = (len(parseable) >= config.FILTER0_MIN_PARSEABLE
                  and consistency is not None
                  and consistency >= config.FILTER0_MIN_CONSISTENCY)
        n_pass += passed
        summaries.append({
            "question_id": qid,
            "n_samples": len(labels),
            "n_parseable": len(parseable),
            "n_correct": n_correct,
            "n_not_attempted": sum(1 for l in parseable if l == "NOT_ATTEMPTED"),
            "consistency": round(consistency, 4) if consistency is not None else None,
            "pass": passed,
            "consistency_stratum": consistency_stratum(consistency),
        })
    if incomplete:
        raise SystemExit(f"ERROR: {incomplete} questions have ungraded samples (permanent "
                         f"API failures). Re-run this stage to retry them.")

    write_jsonl(config.FILTER0_SUMMARY_PATH, summaries)
    n_unparseable = sum(s["n_samples"] - s["n_parseable"] for s in summaries)
    n_total = sum(s["n_samples"] for s in summaries)
    print(f"Filter-0: {n_pass}/{len(summaries)} questions pass "
          f"({100.0 * n_pass / max(len(summaries), 1):.1f}%); "
          f"UNPARSEABLE {n_unparseable}/{n_total} samples")

    update_diagnostics(config.DIAGNOSTICS_PATH, "filter0_grading", {
        "grader_model": config.GRADER_MODEL,
        "n_questions": len(summaries),
        "n_pass": n_pass,
        "pass_rate": round(n_pass / max(len(summaries), 1), 4),
        "n_samples": n_total,
        "n_unparseable": n_unparseable,
        "unparseable_rate": round(n_unparseable / max(n_total, 1), 4),
    })
    write_stage_marker(STAGE, params, {"questions": len(summaries), "pass": n_pass})


if __name__ == "__main__":
    main()
