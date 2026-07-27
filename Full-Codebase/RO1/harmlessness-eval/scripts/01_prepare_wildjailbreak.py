#!/usr/bin/env python3
"""
Step 1 (harmlessness): Prepare a WildJailbreak prompts index (test or eval split).

The WildJailbreak eval config is a single fixed pool; test and eval are two
DISJOINT proportional-stratified samples of it (harmful/benign ratio preserved),
mirroring the monitorability eval's (test, eval) split design:

  test  — default size = 2/3 of the pool, leaving a disjoint half for eval.
  eval  — sampled from the pool MINUS the test split; default size =
          len(test)//2 (half the test set). Requires the FINAL test index.

Idempotent by default (won't overwrite an existing split index without --force).

Usage:
    python harmlessness-eval/scripts/01_prepare_wildjailbreak.py                 # test split
    python harmlessness-eval/scripts/01_prepare_wildjailbreak.py --data eval     # eval split
    python harmlessness-eval/scripts/01_prepare_wildjailbreak.py --data test --n-examples 1000 --force

Output:
    harmlessness-eval/data/wildjailbreak_eval/prompts.jsonl        — test split
    harmlessness-eval/data/wildjailbreak_eval/eval/prompts.jsonl   — eval split
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from src.dataset import SPLITS, prepare_split, load_prompts, resolve_paths

REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="Prepare a WildJailbreak prompts index (test or eval split).")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--data", default="test", choices=SPLITS,
                        help="Split to prepare: test (default) or eval (disjoint "
                             "from test; requires the FINAL test index).")
    parser.add_argument("--n-examples", type=int, default=None,
                        help="Override the sample-size budget. Default: 2/3 of the "
                             "pool for test; half the test size for eval.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap the loaded pool first (smoke-test only). Default: config harmlessness.limit.")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate even if the split's prompts index already exists.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    with open(config_path) as f:
        config = yaml.safe_load(f)

    h = config["harmlessness"]
    index_path = resolve_paths(REPO_ROOT, config, split=args.data)["data_index"]

    if index_path.exists() and not args.force:
        n = len(load_prompts(index_path))
        print(f"'{args.data}' prompts index already exists at {index_path} ({n} prompts).")
        print("Pass --force to regenerate.")
        return

    limit = args.limit if args.limit is not None else h.get("limit")

    test_records = None
    if args.data == "eval":
        test_path = resolve_paths(REPO_ROOT, config, split="test")["data_index"]
        if not test_path.exists():
            print(f"ERROR: test index not found at {test_path}.")
            print("Prepare (and FINALIZE) the test split first: "
                  "01_prepare_wildjailbreak.py --data test")
            sys.exit(1)
        test_records = load_prompts(test_path)
        print(f"Preparing EVAL split, disjoint from the {len(test_records)}-prompt "
              f"test split at {test_path}.")

    prepare_split(config, index_path, split=args.data,
                  test_records=test_records, n_examples=args.n_examples, limit=limit)


if __name__ == "__main__":
    main()
