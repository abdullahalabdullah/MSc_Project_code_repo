#!/usr/bin/env python3
"""
Step 1 (harmlessness): Prepare the WildJailbreak eval prompts index.

Loads the gated `allenai/wildjailbreak` eval config, keeps each adversarial
prompt + its subset label (adversarial_harmful / adversarial_benign), applies the
optional `harmlessness.limit`, and writes a flat prompts index. Idempotent by
default.

Usage:
    python harmlessness-eval/scripts/01_prepare_wildjailbreak.py
    python harmlessness-eval/scripts/01_prepare_wildjailbreak.py --limit 20 --force

Output:
    harmlessness-eval/data/wildjailbreak_eval/prompts.jsonl
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from src.dataset import prepare_prompts, load_prompts

REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="Prepare the WildJailbreak eval prompts index.")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--limit", type=int, default=None,
                        help="Override harmlessness.limit (prompts to keep; default: config).")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate even if the prompts index already exists.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    with open(config_path) as f:
        config = yaml.safe_load(f)

    h = config["harmlessness"]
    index_path = REPO_ROOT / h["paths"]["data_dir"] / "wildjailbreak_eval" / "prompts.jsonl"

    if index_path.exists() and not args.force:
        n = len(load_prompts(index_path))
        print(f"Prompts index already exists at {index_path} ({n} prompts).")
        print("Pass --force to regenerate.")
        return

    limit = args.limit if args.limit is not None else h.get("limit")
    prepare_prompts(config, index_path, limit=limit)


if __name__ == "__main__":
    main()
