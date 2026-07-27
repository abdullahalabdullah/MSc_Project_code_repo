"""Stage E (06) — template assignment (CPU).

Each Filter-0 survivor with a wrong answer yields exactly two prompt instances — one
wrong-opinion (asserting `w`), one right-opinion (asserting gold) — each independently
assigned ONE template, uniformly at random with seed 1234 (spec §5 Stage E). Instances
are iterated sorted by (question_id, direction) so the assignment is deterministic
regardless of upstream file ordering. DROPPED_TEMPLATES (pilot gate 3) removes ids from
the pool for the full run; the remaining templates stay uniform.

Input : filter0/summary.jsonl, filter0/wrong_answers.jsonl, raw/training_pool.jsonl
Output: filter1/instances.jsonl  {instance_id, question_id, direction, template_id,
                                  question, gold, gold_aliases, asserted_answer,
                                  prompt, filter0_consistency}
"""
import argparse
import random

import config
import env_setup
import prompts
from utils import (check_stage_cache, read_jsonl, update_diagnostics, upstream_hash,
                   write_jsonl, write_stage_marker)

STAGE = "06_templates"


def main():
    ap = argparse.ArgumentParser(description="Stage E — template assignment (CPU).")
    ap.add_argument("--force", action="store_true", help="Re-run even if cached.")
    args = ap.parse_args()

    env_setup.setup_env()
    config.ensure_output_dirs()

    active_templates = [t for t in prompts.TEMPLATE_IDS
                        if t not in config.DROPPED_TEMPLATES]
    if not active_templates:
        raise SystemExit("ERROR: DROPPED_TEMPLATES removed every template.")
    params = {
        "upstream": upstream_hash("05_wrong_answers"),
        "seed_template": config.SEED_TEMPLATE,
        "templates": {t: prompts.TEMPLATES[t] for t in active_templates},
        "answer_format": prompts.ANSWER_FORMAT_INSTRUCTION,
    }
    if check_stage_cache(STAGE, params, [config.INSTANCES_PATH], force=args.force):
        return

    pool = {r["question_id"]: r for r in read_jsonl(config.TRAINING_POOL_PATH)}
    consistency = {s["question_id"]: s["consistency"]
                   for s in read_jsonl(config.FILTER0_SUMMARY_PATH)}
    wrong = {r["question_id"]: r["w"] for r in read_jsonl(config.WRONG_ANSWERS_PATH)}

    # (question_id, direction) sorted -> deterministic template stream (seed 1234).
    keys = sorted((qid, direction) for qid in wrong
                  for direction in ("right_opinion", "wrong_opinion"))
    rng = random.Random(config.SEED_TEMPLATE)

    instances = []
    for qid, direction in keys:
        q = pool[qid]
        asserted = wrong[qid] if direction == "wrong_opinion" else q["gold"]
        template_id = rng.choice(active_templates)
        instances.append({
            "instance_id": f"{qid}::{direction}",
            "question_id": qid,
            "direction": direction,
            "template_id": template_id,
            "question": q["question"],
            "gold": q["gold"],
            "gold_aliases": q["gold_aliases"],
            "asserted_answer": asserted,
            "prompt": prompts.render_opinion_prompt(template_id, q["question"], asserted),
            "filter0_consistency": consistency.get(qid),
        })

    write_jsonl(config.INSTANCES_PATH, instances)
    per_template = {}
    for inst in instances:
        per_template[inst["template_id"]] = per_template.get(inst["template_id"], 0) + 1
    print(f"{len(instances)} instances ({len(instances) // 2} questions x 2 directions) "
          f"-> {config.INSTANCES_PATH}")
    print(f"  per template: {dict(sorted(per_template.items()))}")
    if config.DROPPED_TEMPLATES:
        print(f"  dropped templates (pilot gate 3): {config.DROPPED_TEMPLATES}")

    update_diagnostics(config.DIAGNOSTICS_PATH, "stage_e_templates", {
        "n_instances": len(instances),
        "active_templates": active_templates,
        "dropped_templates": config.DROPPED_TEMPLATES,
        "per_template": per_template,
        "seed_template": config.SEED_TEMPLATE,
    })
    write_stage_marker(STAGE, params, {"instances": len(instances)})


if __name__ == "__main__":
    main()
