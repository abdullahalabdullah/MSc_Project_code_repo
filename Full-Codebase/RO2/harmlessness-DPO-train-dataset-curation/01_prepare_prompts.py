"""Stage 01 — Load WildJailbreak harmful prompts, deduplicate, select the prompt set.

config.SAMPLE controls the selection: None ("ALL", the default) keeps EVERY unique
harmful prompt; an integer draws a seeded random sample of that size (smoke/pilot runs).

config.EXCLUDE_RUNS (comma-separated run dirs) removes prompts already consumed by
previous runs from the pool BEFORE selection — so a pilot and a follow-up full run never
overlap, and their final pairs can later be combined with merge_runs.py.

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
    subset_name = {"vanilla_harmful": "vanilla", "adversarial_harmful": "adversarial"}

    # Pull columns once (fast) then select each harmful row's prompt from the right column.
    data_types = ds["data_type"]
    columns = {col: ds[col] for col in set(type_to_col.values())}
    vanilla_col = ds["vanilla"]  # seed query (== the prompt for the vanilla subset)

    # Each entry: (prompt, subset, vanilla_query)
    rows = []
    per_type_counts = Counter()
    for i, dt in enumerate(data_types):
        if dt not in type_to_col:
            continue
        prompt = columns[type_to_col[dt]][i].strip()
        if not prompt:
            continue
        rows.append((prompt, subset_name[dt], vanilla_col[i].strip()))
        per_type_counts[dt] += 1

    print("  harmful rows kept per subset:")
    for t in type_to_col:
        print(f"    {t}: {per_type_counts.get(t, 0)}")

    # Deduplicate on prompt text, preserving first-seen order + its subset/vanilla seed.
    seen = {}
    for prompt, subset, vanilla_q in rows:
        if prompt not in seen:
            seen[prompt] = (subset, vanilla_q)
    unique_prompts = list(seen.keys())
    n_unique = len(unique_prompts)
    print(f"\n>> Unique harmful prompts (after dedup): {n_unique}")

    # Exclude prompts already consumed by previous runs (EXCLUDE_RUNS env var), so this
    # run samples only NEW prompts and the runs' outputs can later be merged.
    if config.EXCLUDE_RUNS:
        excluded = set()
        for run_dir in config.EXCLUDE_RUNS:
            meta_path = os.path.join(run_dir, os.path.basename(config.PROMPTS_META_PATH))
            if not os.path.exists(meta_path):
                raise FileNotFoundError(
                    f"EXCLUDE_RUNS: no {os.path.basename(config.PROMPTS_META_PATH)} in {run_dir!r}"
                )
            run_prompts = {m["prompt"] for m in joblib.load(meta_path)}
            print(f"   excluding {len(run_prompts)} prompts from previous run: {run_dir}")
            excluded |= run_prompts
        before = len(unique_prompts)
        unique_prompts = [p for p in unique_prompts if p not in excluded]
        n_unique = len(unique_prompts)
        print(f">> After excluding {before - n_unique} previously-used prompts: {n_unique} remain")

    if config.SAMPLE is None:
        # SAMPLE=ALL — keep every unique harmful prompt (full-dataset curation run).
        sampled = unique_prompts
        print(f">> Using ALL {len(sampled)} unique prompts (SAMPLE=ALL)")
    else:
        if n_unique < config.SAMPLE:
            raise ValueError(
                f"Requested SAMPLE={config.SAMPLE} but only {n_unique} unique prompts available."
            )
        # Randomly sample SAMPLE unique prompts with the fixed seed.
        sampled = random.sample(unique_prompts, config.SAMPLE)
        print(f">> Sampled {len(sampled)} unique prompts (seed={config.SEED})")
    mix = Counter(seen[p][0] for p in sampled)
    print("   sample mix: " + ", ".join(f"{s}={mix.get(s, 0)}" for s in ("vanilla", "adversarial")))

    # PROMPTS_PATH stays a plain list[str] (stage 02 input, unchanged). PROMPTS_META_PATH
    # carries the subset + vanilla seed query per prompt (used by the sanity check).
    meta = [{"prompt": p, "subset": seen[p][0], "vanilla": seen[p][1]} for p in sampled]
    joblib.dump(sampled, config.PROMPTS_PATH, compress=True)
    joblib.dump(meta, config.PROMPTS_META_PATH, compress=True)
    print(f"Saved -> {config.PROMPTS_PATH}")
    print(f"Saved -> {config.PROMPTS_META_PATH}")


if __name__ == "__main__":
    main()
