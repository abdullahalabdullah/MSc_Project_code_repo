"""Stages G+H (09) — pair selection, balancing, assembly (CPU).

Stage G: for each Filter-1 survivor, select exactly one (chosen, rejected) pair —
uniformly at random within each class (seed 1701, one stream over instances sorted by
instance_id). No ranking by any judge/heuristic/length/style, by design (spec §7).

Stage H: pool pairs, downsample the majority direction to exactly 50/50 (seed 2003),
shuffle (same stream), write pairs/anti_syc_dpo.jsonl. Asserts zero overlap with
reserved/ (question_id vs Eval set 1; exact question text vs Eval set 2). Full runs
(X > 500, not PILOT) abort below 3,000 pairs (spec §8 — report to owner, do not lower
thresholds).

Input : filter1/{summary,generations,grades}.jsonl + instances.jsonl + reserved/
Output: pairs/anti_syc_dpo.jsonl (spec §9 pair schema, verbatim full_text incl. <think>)
"""
import argparse
import random

import config
import env_setup
import prompts
from grading import load_grades
from utils import (check_stage_cache, read_jsonl, update_diagnostics, upstream_hash,
                   write_jsonl, write_stage_marker)

STAGE = "09_pairs"

CHOSEN_CLASS = {"wrong_opinion": ("CORRECT",), "right_opinion": ("CORRECT",)}
REJECTED_CLASS = {"wrong_opinion": ("MATCHES_W",), "right_opinion": ("OTHER_WRONG",)}


def _sample_record(sample, grade_line):
    return {
        "full_text": sample["full_text"],   # verbatim, incl. <think> — never trimmed
        "final_answer": grade_line["extracted"],
        "grade": grade_line["label"],
        "num_tokens": sample["num_tokens"],
    }


def main():
    ap = argparse.ArgumentParser(description="Stages G+H — pair selection + balancing (CPU).")
    ap.add_argument("--force", action="store_true", help="Re-run even if cached.")
    args = ap.parse_args()

    env_setup.setup_env()
    config.ensure_output_dirs()

    params = {
        "upstream": upstream_hash("08_filter1_grade"),
        "seed_pair": config.SEED_PAIR,
        "seed_balance": config.SEED_BALANCE,
        "min_pairs_full_run": config.MIN_PAIRS_FULL_RUN,
    }
    if check_stage_cache(STAGE, params, [config.PAIRS_PATH], force=args.force):
        return

    instances = {i["instance_id"]: i for i in read_jsonl(config.INSTANCES_PATH)}
    summaries = {s["instance_id"]: s for s in read_jsonl(config.FILTER1_SUMMARY_PATH)}
    generations = {g["instance_id"]: g for g in read_jsonl(config.FILTER1_GENERATIONS_PATH)}
    grades = load_grades(config.FILTER1_GRADES_PATH, ("instance_id", "sample_idx"))

    # ── Stage G: one pair per surviving instance (seed 1701, sorted instance ids) ──
    rng = random.Random(config.SEED_PAIR)
    pairs = []
    for iid in sorted(summaries):
        summ = summaries[iid]
        if not summ["pass"]:
            continue
        inst = instances[iid]
        direction = inst["direction"]
        chosen_ix, rejected_ix = [], []
        for s_idx in range(len(generations[iid]["samples"])):
            label = grades[(iid, s_idx)]["label"]
            if label in CHOSEN_CLASS[direction]:
                chosen_ix.append(s_idx)
            elif label in REJECTED_CLASS[direction]:
                rejected_ix.append(s_idx)
        c = rng.choice(chosen_ix)
        r = rng.choice(rejected_ix)
        pairs.append({
            "pair_id": iid,
            "question_id": inst["question_id"],
            "source": "triviaqa",
            "direction": direction,
            "template_id": inst["template_id"],
            "question": inst["question"],
            "gold": inst["gold"],
            "gold_aliases": inst["gold_aliases"],
            "asserted_answer": inst["asserted_answer"],
            "prompt": inst["prompt"],
            "chosen": _sample_record(generations[iid]["samples"][c], grades[(iid, c)]),
            "rejected": _sample_record(generations[iid]["samples"][r], grades[(iid, r)]),
            "filter0_consistency": inst["filter0_consistency"],
            "syc_fraction": summ["syc_fraction"],
            "n_chosen_class": summ["n_chosen_class"],
            "n_rejected_class": summ["n_rejected_class"],
        })

    by_dir = {"wrong_opinion": [p for p in pairs if p["direction"] == "wrong_opinion"],
              "right_opinion": [p for p in pairs if p["direction"] == "right_opinion"]}
    print(f"Stage G pairs: {len(pairs)} "
          f"(wrong={len(by_dir['wrong_opinion'])}, right={len(by_dir['right_opinion'])})")

    # ── Stage H: 50/50 direction balance (downsample majority, seed 2003) + shuffle ──
    rng_bal = random.Random(config.SEED_BALANCE)
    n_keep = min(len(by_dir["wrong_opinion"]), len(by_dir["right_opinion"]))
    balanced = []
    for direction in ("wrong_opinion", "right_opinion"):
        group = by_dir[direction]
        balanced.extend(rng_bal.sample(group, n_keep) if len(group) > n_keep else group)
    rng_bal.shuffle(balanced)

    # ── Spec §8 assertion: zero overlap with reserved/ ─────────────────────────
    eval_id_qids = {r["question_id"] for r in read_jsonl(config.EVAL_ID_PATH)}
    id_overlap = {p["question_id"] for p in balanced} & eval_id_qids
    assert not id_overlap, f"pairs overlap Eval set 1 on question_id: {sorted(id_overlap)[:5]}"
    eval_ood_texts = {r["question"] for r in read_jsonl(config.EVAL_OOD_PATH)}
    text_overlap = {p["question"] for p in balanced} & eval_ood_texts
    assert not text_overlap, f"pairs overlap Eval set 2 on question text: {list(text_overlap)[:2]}"

    # ── Spec §8 minimum-size gate (full run only) ───────────────────────────────
    if len(balanced) < config.MIN_PAIRS_FULL_RUN and not config.PILOT and config.X > 500:
        raise SystemExit(
            f"ERROR: {len(balanced)} pairs post-balance < {config.MIN_PAIRS_FULL_RUN} "
            f"(spec §8 full-run floor). STOP and report to the project owner — the "
            f"documented escalation is raising N_FILTER1 to 16 (owner decision).")

    write_jsonl(config.PAIRS_PATH, balanced)
    n_wrong = sum(1 for p in balanced if p["direction"] == "wrong_opinion")
    print(f"Stage H balanced pairs: {len(balanced)} "
          f"(wrong={n_wrong}, right={len(balanced) - n_wrong}) -> {config.PAIRS_PATH}")

    update_diagnostics(config.DIAGNOSTICS_PATH, "pairs", {
        "n_pairs_selected": len(pairs),
        "n_pairs_balanced": len(balanced),
        "pre_balance_per_direction": {d: len(v) for d, v in by_dir.items()},
        "seed_pair": config.SEED_PAIR,
        "seed_balance": config.SEED_BALANCE,
    })
    write_stage_marker(STAGE, params, {"selected": len(pairs), "balanced": len(balanced)})


if __name__ == "__main__":
    main()
