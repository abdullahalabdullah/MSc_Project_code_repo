#!/usr/bin/env python3
"""
Step 1: Prepare a dataset index.

Loads a dataset from the `datasets:` registry in config/mvp_config.yaml, applies
its filter, samples a balanced subset (uniform; two-level where applicable),
assigns incorrect hint targets, and writes data/{dataset}/dataset_index.json.

Usage:
    python scripts/01_prepare_dataset.py --dataset <key> [--config config/mvp_config.yaml]
    python scripts/01_prepare_dataset.py --dataset <key> --n-examples 50 --force

Options:
    --n-examples   override the config n_examples budget for this run (e.g. 50)
    --force        regenerate even if data/{dataset}/dataset_index.json exists

Output:
    data/{dataset}/dataset_index.json   — up to n_examples instance dicts (see src/dataset.py)
"""

import argparse
import sys
from pathlib import Path

# Allow imports from MVP/ root
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.dataset import prepare_dataset, resolve_paths


def main():
    parser = argparse.ArgumentParser(description="Prepare a dataset index.")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--dataset", default="mmlu_redux",
                        help="Dataset key from the config `datasets:` registry: "
                             "mmlu_redux, gpqa_diamond, wmdp, codemmlu, lsat_ar, "
                             "lsat_lr. Default: mmlu_redux.")
    parser.add_argument("--n-examples", type=int, default=None,
                        help="Override the config n_examples budget (e.g. 50).")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate even if the dataset index already exists.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent.parent / config_path

    with open(config_path) as f:
        config = yaml.safe_load(f)

    root = Path(__file__).parent.parent
    output_path = resolve_paths(root, config, args.dataset)["data_index"]

    if output_path.exists() and not args.force:
        print(f"dataset_index.json already exists at {output_path}.")
        print("Pass --force (or delete the file) to regenerate, e.g. with --n-examples.")
        return

    prepare_dataset(config, output_path, dataset_key=args.dataset, n_examples=args.n_examples)


if __name__ == "__main__":
    main()
