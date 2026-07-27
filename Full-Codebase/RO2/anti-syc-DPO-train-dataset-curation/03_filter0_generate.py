"""Stage C generation (03) — Filter-0 bare-prompt sampling (GPU, vLLM).

Runs every training-pool question through the bare prompt (spec §5 Stage C) N=10 times
with the frozen §2 sampling config, thinking mode ON. Grading happens in stage 04 (API)
so this is the only thing the first GPU session has to do.

Input : raw/training_pool.jsonl
Output: filter0/generations.jsonl   {question_id, samples: [{full_text, finish_reason,
                                     num_tokens, truncated} x N]}
Resume: filter0/generations.ckpt.jsonl (fingerprint-guarded; see generation.py).
"""
import argparse

import config
import env_setup
import prompts
from generation import generate_samples
from utils import check_stage_cache, read_jsonl, upstream_hash, write_stage_marker

STAGE = "03_filter0_generate"


def main():
    ap = argparse.ArgumentParser(description="Stage C — Filter-0 generation (GPU).")
    ap.add_argument("--force", action="store_true", help="Re-run even if cached.")
    args = ap.parse_args()

    env_setup.setup_env()
    config.ensure_output_dirs()

    params = {
        "upstream": upstream_hash("01_sample_pool"),
        "n": config.N,
        "sampling": config.sampling_config(),
        "answer_format": prompts.ANSWER_FORMAT_INSTRUCTION,
    }
    if check_stage_cache(STAGE, params, [config.FILTER0_GENERATIONS_PATH], force=args.force):
        return

    pool = read_jsonl(config.TRAINING_POOL_PATH)
    print(f"Loaded {len(pool)} pool questions")
    items = [{"question_id": r["question_id"],
              "prompt": prompts.bare_prompt(r["question"])} for r in pool]

    records = generate_samples(
        items, key_field="question_id", prompt_field="prompt", n_samples=config.N,
        ckpt_path=config.FILTER0_GEN_CKPT_PATH, out_path=config.FILTER0_GENERATIONS_PATH,
        diag_section="filter0_generation", stage_params=params,
    )
    write_stage_marker(STAGE, params, {"questions": len(records),
                                       "samples": len(records) * config.N})


if __name__ == "__main__":
    main()
