"""Stage 01 — Load WildJailbreak harmful prompts, deduplicate, sample SAMPLE prompts.

Uses only the two harmful subsets of allenai/wildjailbreak (config in config.py):
  - vanilla_harmful      -> prompt text from the `vanilla` column
  - adversarial_harmful  -> prompt text from the `adversarial` column

WildJailbreak is GATED (accept terms on the HF page + HF_TOKEN available). Reports the
per-subset counts, the number of unique prompts, and the vanilla/adversarial mix of the
sample. Output: config.PROMPTS_PATH (list[str]).
"""
import os
import random
from collections import Counter

import joblib
from datasets import load_dataset

import config
import env_setup
from utils import set_seed


def main():
    env_setup.setup_env()
    config.ensure_output_dir()
    set_seed(config.SEED)

    token = os.environ.get("HF_TOKEN")
    print(f"Loading {config.DATASET} [{config.DATASET_CONFIG}] (gated — needs HF auth) ...")
    ds = load_dataset(
        config.DATASET,
        config.DATASET_CONFIG,
        delimiter="\t",
        keep_default_na=False,
        split="train",
        token=token,
    )
    print(f"  total rows: {len(ds)}")

    type_to_col = config.WILDJAILBREAK_TYPE_TO_COLUMN

    # Pull columns once (fast) then select each harmful row's prompt from the right column.
    data_types = ds["data_type"]
    columns = {col: ds[col] for col in set(type_to_col.values())}

    pairs = []  # (prompt, data_type)
    per_type_counts = Counter()
    for i, dt in enumerate(data_types):
        if dt not in type_to_col:
            continue
        prompt = columns[type_to_col[dt]][i].strip()
        if not prompt:
            continue
        pairs.append((prompt, dt))
        per_type_counts[dt] += 1

    print("  harmful rows kept per subset:")
    for t in type_to_col:
        print(f"    {t}: {per_type_counts.get(t, 0)}")

    # Deduplicate on prompt text, preserving first-seen order + its subset label.
    seen = {}
    for prompt, dt in pairs:
        if prompt not in seen:
            seen[prompt] = dt
    unique_prompts = list(seen.keys())
    n_unique = len(unique_prompts)
    print(f"\n>> Unique harmful prompts (after dedup): {n_unique}")

    if n_unique < config.SAMPLE:
        raise ValueError(
            f"Requested SAMPLE={config.SAMPLE} but only {n_unique} unique prompts available."
        )

    # Randomly sample SAMPLE unique prompts with the fixed seed.
    sampled = random.sample(unique_prompts, config.SAMPLE)
    mix = Counter(seen[p] for p in sampled)
    print(f">> Sampled {len(sampled)} unique prompts (seed={config.SEED})")
    print("   sample mix: " + ", ".join(f"{t}={mix.get(t, 0)}" for t in type_to_col))

    joblib.dump(sampled, config.PROMPTS_PATH, compress=True)
    print(f"Saved -> {config.PROMPTS_PATH}")


if __name__ == "__main__":
    main()
