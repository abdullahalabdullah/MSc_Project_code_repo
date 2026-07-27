"""Stage A (01) — TriviaQA acquisition & sampling (CPU + network).

Spec §3: load the full `mandarjoshi/trivia_qa` rc.nocontext train split, draw the
1,500-question eval slice FIRST (seed 7), then the X-question training pool from the
remainder, continuing the same seeded RNG stream. Implemented as one seeded shuffle of
the record indices with the eval slice = first 1500 and the pool = the next X: this is
exactly "eval slice first, pool from the remainder, same stream", and it additionally
makes the eval slice byte-identical across smoke/pilot/full runs and the pilot's pool a
prefix of the full run's.

Outputs:
  raw/training_pool.jsonl                 {question_id, question, gold, gold_aliases, source}
  reserved/eval_id_triviaqa.partial.jsonl same schema (the frozen `w` is attached by stage 05)
"""
import argparse
import random

import config
import env_setup
from utils import check_stage_cache, update_diagnostics, write_jsonl, write_stage_marker

STAGE = "01_sample_pool"


def _record(row):
    answer = row["answer"]
    aliases = list(dict.fromkeys(
        list(answer.get("aliases", [])) + list(answer.get("normalized_aliases", []))
    ))
    return {
        "question_id": row["question_id"],   # native ID — never derived/re-hashed
        "question": row["question"],
        "gold": answer["value"],
        "gold_aliases": aliases,
        "source": "triviaqa",
    }


def main():
    ap = argparse.ArgumentParser(description="Stage A — sample eval slice + training pool.")
    ap.add_argument("--force", action="store_true", help="Re-run even if cached.")
    args = ap.parse_args()

    env_setup.setup_env()
    config.ensure_output_dirs()

    params = {
        "dataset": config.TRIVIAQA_DATASET,
        "dataset_config": config.TRIVIAQA_CONFIG,
        "split": config.TRIVIAQA_SPLIT,
        "seed_sampling": config.SEED_SAMPLING,
        "eval_slice_size": config.EVAL_SLICE_SIZE,
        "x": config.X,
    }
    outputs = [config.TRAINING_POOL_PATH, config.EVAL_ID_PARTIAL_PATH]
    if check_stage_cache(STAGE, params, outputs, force=args.force):
        return

    from datasets import load_dataset
    print(f"Loading {config.TRIVIAQA_DATASET} ({config.TRIVIAQA_CONFIG}, "
          f"{config.TRIVIAQA_SPLIT} split) ...")
    ds = load_dataset(config.TRIVIAQA_DATASET, config.TRIVIAQA_CONFIG,
                      split=config.TRIVIAQA_SPLIT)
    n = len(ds)
    # TriviaQA's rc.nocontext stores ONE ROW PER (question, evidence-document) pair —
    # "nocontext" strips the evidence text but keeps the row-per-document duplication —
    # so `question_id` is NOT unique across rows. Deduplicate to one row per question_id
    # (first occurrence — deterministic in dataset order) BEFORE sampling; otherwise the
    # same question can fall in both the eval slice and the training pool (spec §3
    # requires them disjoint) and duplicate questions would be generated/paired twice.
    seen = set()
    unique_indices = []
    for i, qid in enumerate(ds["question_id"]):
        if qid not in seen:
            seen.add(qid)
            unique_indices.append(i)
    n_unique = len(unique_indices)
    print(f"Loaded {n} rows -> {n_unique} unique questions "
          f"(rc.nocontext repeats each question once per evidence document)")
    if n_unique < config.EVAL_SLICE_SIZE + config.X:
        raise SystemExit(f"ERROR: only {n_unique} unique questions < eval slice "
                         f"({config.EVAL_SLICE_SIZE}) + X ({config.X})")

    # One seeded shuffle over the UNIQUE questions; eval slice = first 1500 (drawn
    # first), pool = next X — same seeded RNG stream (spec §3).
    rng = random.Random(config.SEED_SAMPLING)
    rng.shuffle(unique_indices)
    eval_idx = unique_indices[:config.EVAL_SLICE_SIZE]
    pool_idx = unique_indices[config.EVAL_SLICE_SIZE:config.EVAL_SLICE_SIZE + config.X]

    eval_records = [_record(ds[i]) for i in eval_idx]
    pool_records = [_record(ds[i]) for i in pool_idx]

    # Disjoint by construction; assert on native IDs anyway (catches loader bugs).
    eval_ids = {r["question_id"] for r in eval_records}
    pool_ids = {r["question_id"] for r in pool_records}
    overlap = eval_ids & pool_ids
    assert not overlap, f"eval/pool question_id overlap: {sorted(overlap)[:5]} ..."
    # Native IDs must also be unique within each file.
    assert len(eval_ids) == len(eval_records), "duplicate question_id in eval slice"
    assert len(pool_ids) == len(pool_records), "duplicate question_id in training pool"

    write_jsonl(config.TRAINING_POOL_PATH, pool_records)
    write_jsonl(config.EVAL_ID_PARTIAL_PATH, eval_records)
    print(f"Training pool: {len(pool_records)} -> {config.TRAINING_POOL_PATH}")
    print(f"Eval slice   : {len(eval_records)} -> {config.EVAL_ID_PARTIAL_PATH} "
          f"(w attached by stage 05)")

    update_diagnostics(config.DIAGNOSTICS_PATH, "stage_a_sampling", {
        **params,
        "split_rows": n,
        "unique_questions": n_unique,
        "n_eval_slice": len(eval_records),
        "n_training_pool": len(pool_records),
    })
    write_stage_marker(STAGE, params, {"pool": len(pool_records), "eval": len(eval_records)})


if __name__ == "__main__":
    main()
