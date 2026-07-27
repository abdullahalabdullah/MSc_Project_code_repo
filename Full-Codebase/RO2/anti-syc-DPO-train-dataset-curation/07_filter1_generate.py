"""Stage F generation (07) — Filter-1 contrast sampling (GPU, vLLM).

Runs every Stage-E prompt instance N_FILTER1 (default 10) times with the frozen §2
sampling config. Grading + class assignment happen in stage 08 (API), so this is the
only thing the second GPU session has to do.

Input : filter1/instances.jsonl
Output: filter1/generations.jsonl   {instance_id, samples: [...] x N_FILTER1}
Resume: filter1/generations.ckpt.jsonl (fingerprint-guarded).
"""
import argparse

import config
import env_setup
import prompts
from generation import generate_samples
from utils import check_stage_cache, read_jsonl, upstream_hash, write_stage_marker

STAGE = "07_filter1_generate"


def main():
    ap = argparse.ArgumentParser(description="Stage F — Filter-1 generation (GPU).")
    ap.add_argument("--force", action="store_true", help="Re-run even if cached.")
    args = ap.parse_args()

    env_setup.setup_env()
    config.ensure_output_dirs()

    params = {
        "upstream": upstream_hash("06_templates"),
        "n": config.N_FILTER1,
        "sampling": config.sampling_config(),
    }
    if check_stage_cache(STAGE, params, [config.FILTER1_GENERATIONS_PATH], force=args.force):
        return

    instances = read_jsonl(config.INSTANCES_PATH)
    print(f"Loaded {len(instances)} prompt instances")

    records = generate_samples(
        instances, key_field="instance_id", prompt_field="prompt",
        n_samples=config.N_FILTER1,
        ckpt_path=config.FILTER1_GEN_CKPT_PATH, out_path=config.FILTER1_GENERATIONS_PATH,
        diag_section="filter1_generation", stage_params=params,
    )
    write_stage_marker(STAGE, params, {"instances": len(records),
                                       "samples": len(records) * config.N_FILTER1})


if __name__ == "__main__":
    main()
