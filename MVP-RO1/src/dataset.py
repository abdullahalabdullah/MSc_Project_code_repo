"""
Dataset preparation for the MVP CoT monitorability evaluation.

Loads MMLU, samples 200 examples (capped at max_per_subject per subject),
assigns incorrect hint targets, and saves a dataset_index.json.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

MMLU_LETTERS = ["A", "B", "C", "D"]


def prepare_dataset(config: dict, output_path: Path) -> list:
    """
    Sample 200 MMLU examples with incorrect hint targets and save to JSON.

    Input:  config dict (from mvp_config.yaml), output_path (Path)
    Output: list of instance dicts; file written to output_path
    """
    from datasets import load_dataset as hf_load_dataset

    ds_cfg = config["dataset"]
    rng = random.Random(ds_cfg["random_seed"])

    print(f"Loading {ds_cfg['name']} ({ds_cfg['config']}, {ds_cfg['split']}) ...")
    dataset = hf_load_dataset(ds_cfg["name"], ds_cfg["config"], split=ds_cfg["split"])

    # Group rows by subject
    by_subject: dict[str, list] = defaultdict(list)
    for idx, row in enumerate(dataset):
        by_subject[row["subject"]].append((idx, row))

    # Sample up to max_per_subject from each subject
    max_per = ds_cfg["max_per_subject"]
    pool = []
    for subject, rows in sorted(by_subject.items()):
        sample_size = min(len(rows), max_per)
        pool.extend(rng.sample(rows, sample_size))

    n = ds_cfg["n_examples"]
    if len(pool) < n:
        raise ValueError(f"Pool has only {len(pool)} examples; need {n}.")

    selected = rng.sample(pool, n)

    records = []
    for instance_id, (orig_idx, row) in enumerate(selected):
        correct_letter = MMLU_LETTERS[row["answer"]]
        wrong_letters = [l for l in MMLU_LETTERS if l != correct_letter]
        hint_target = rng.choice(wrong_letters)

        assert hint_target != correct_letter, (
            f"instance {instance_id}: hint_target == correct_answer ({correct_letter})"
        )

        records.append({
            "instance_id": instance_id,
            "question_id": f"{row['subject']}_{orig_idx}",
            "subject": row["subject"],
            "question": row["question"],
            "choices": {
                "A": row["choices"][0],
                "B": row["choices"][1],
                "C": row["choices"][2],
                "D": row["choices"][3],
            },
            "correct_answer": correct_letter,
            "hint_target": hint_target,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(records)} instances to {output_path}")
    return records


def load_dataset_index(path: Path) -> list:
    """Load dataset_index.json and return list of instance dicts."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)
