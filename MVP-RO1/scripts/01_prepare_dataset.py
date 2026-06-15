#!/usr/bin/env python3
"""
Step 1: Prepare the MMLU dataset index.

Loads cais/mmlu (all subjects, test split), samples 200 examples with a
cap of 5 per subject, assigns incorrect hint targets, and writes
data/dataset_index.json.

Usage:
    python scripts/01_prepare_dataset.py [--config config/mvp_config.yaml]

Output:
    data/dataset_index.json   — 200 instance dicts (see src/dataset.py)
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
    parser.add_argument("--dataset", default="mmlu",
                        help="Dataset key (mmlu [legacy paths], or a key from the "
                             "config `datasets:` registry, e.g. gpqa, gsm_mc, math_mc, "
                             "agieval, wmdp). Default: mmlu.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent.parent / config_path

    with open(config_path) as f:
        config = yaml.safe_load(f)

    root = Path(__file__).parent.parent
    output_path = resolve_paths(root, config, args.dataset)["data_index"]

    if output_path.exists():
        print(f"dataset_index.json already exists at {output_path}.")
        print("Delete the file to regenerate with a different seed.")
        return

    prepare_dataset(config, output_path, dataset_key=args.dataset)


if __name__ == "__main__":
    main()
