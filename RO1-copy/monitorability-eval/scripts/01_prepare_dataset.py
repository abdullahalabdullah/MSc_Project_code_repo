#!/usr/bin/env python3
"""
Step 1: Prepare a dataset index.

Loads a dataset from the `datasets:` registry in config/mvp_config.yaml, applies
its filter, samples a balanced subset (uniform; two-level where applicable),
assigns incorrect hint targets, and writes data/{dataset}/dataset_index.json.

Usage:
    python scripts/01_prepare_dataset.py --dataset <key> [--config config/mvp_config.yaml]
    python scripts/01_prepare_dataset.py --dataset <key> --n-examples 50 --force
    python scripts/01_prepare_dataset.py --dataset <key> --data eval

Options:
    --data         which split to prepare: test (default) or eval. The eval
                   split is sampled from the pool MINUS the test split's
                   questions (disjoint by construction) and defaults to half
                   the test split's size; the test index must exist (and be
                   FINAL — all top-ups done) before preparing eval.
    --n-examples   override the sample-size budget for this run (e.g. 50).
                   Default: config n_examples for test; len(test)//2 for eval.
    --force        regenerate even if the split's dataset index exists

Output:
    data/{dataset}/dataset_index.json        — test split (see src/dataset.py)
    data/{dataset}/eval/dataset_index.json   — eval split
"""

import argparse
import sys
from pathlib import Path

# Allow imports from MVP/ root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from src.dataset import SPLITS, load_dataset_index, prepare_dataset, resolve_paths


def main():
    parser = argparse.ArgumentParser(description="Prepare a dataset index.")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--dataset", default="mmlu_pro",
                        help="Dataset key from the config `datasets:` registry: "
                             "mmlu_pro, gpqa_diamond, wmdp, codemmlu, lsat_ar, "
                             "lsat_lr, medxpertqa, lexam. Default: mmlu_pro.")
    parser.add_argument("--data", default="test", choices=SPLITS,
                        help="Split to prepare: test (default) or eval (disjoint "
                             "from test; requires the FINAL test index).")
    parser.add_argument("--n-examples", type=int, default=None,
                        help="Override the sample-size budget (e.g. 50). Default: "
                             "config n_examples for test; half the test size for eval.")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate even if the dataset index already exists.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parents[2] / config_path

    with open(config_path) as f:
        config = yaml.safe_load(f)

    root = Path(__file__).resolve().parents[2]
    output_path = resolve_paths(root, config, args.dataset, split=args.data)["data_index"]

    if output_path.exists() and not args.force:
        print(f"dataset_index.json already exists at {output_path}.")
        print("Pass --force (or delete the file) to regenerate, e.g. with --n-examples.")
        return

    exclude = None
    n_examples = args.n_examples
    if args.data == "eval":
        eval_excluded = config.get("eval_exclude_datasets", []) or []
        if args.dataset in eval_excluded:
            print(f"ERROR: dataset '{args.dataset}' has no eval split "
                  f"(config eval_exclude_datasets: {eval_excluded} — pool too "
                  f"small for a disjoint eval set on top of the test split).")
            sys.exit(1)
        test_path = resolve_paths(root, config, args.dataset, split="test")["data_index"]
        if not test_path.exists():
            print(f"ERROR: test index not found at {test_path}.")
            print("Prepare (and FINALIZE — all top-ups done) the test split first: "
                  f"01_prepare_dataset.py --dataset {args.dataset}")
            sys.exit(1)
        exclude = load_dataset_index(test_path)
        if n_examples is None:
            n_examples = len(exclude) // 2
        print(f"Preparing EVAL split: {n_examples} examples, disjoint from the "
              f"{len(exclude)}-example test split at {test_path}.")

    prepare_dataset(config, output_path, dataset_key=args.dataset,
                    n_examples=n_examples, exclude=exclude)


if __name__ == "__main__":
    main()
