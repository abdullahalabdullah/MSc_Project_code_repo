"""Stage 01 — Load PKU-SafeRLHF-30K, deduplicate on prompt, sample SAMPLE prompts.

Reports the number of unique prompts. Output: config.PROMPTS_PATH (list[str]).
"""
import random

import joblib
from datasets import load_dataset

import config
import env_setup
from utils import set_seed


def main():
    env_setup.setup_env()
    config.ensure_output_dir()
    set_seed(config.SEED)

    print(f"Loading {config.DATASET} [{config.DATASET_SPLIT}] ...")
    ds = load_dataset(config.DATASET, split=config.DATASET_SPLIT)
    print(f"  rows: {len(ds)}")

    # Deduplicate on the prompt column, preserving first-seen order.
    all_prompts = ds[config.PROMPT_COLUMN]
    unique_prompts = list(dict.fromkeys(all_prompts))
    n_unique = len(unique_prompts)
    print(f"\n>> Unique prompts (after dedup on '{config.PROMPT_COLUMN}'): {n_unique}")

    if n_unique < config.SAMPLE:
        raise ValueError(
            f"Requested SAMPLE={config.SAMPLE} but only {n_unique} unique prompts available."
        )

    # Randomly sample SAMPLE unique prompts with the fixed seed.
    sampled = random.sample(unique_prompts, config.SAMPLE)
    print(f">> Sampled {len(sampled)} unique prompts (seed={config.SEED})")

    joblib.dump(sampled, config.PROMPTS_PATH, compress=True)
    print(f"Saved -> {config.PROMPTS_PATH}")


if __name__ == "__main__":
    main()
