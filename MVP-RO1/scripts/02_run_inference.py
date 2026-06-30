#!/usr/bin/env python3
"""
Step 2: Run hinted and unhinted inference for one model checkpoint.

Loads the model via vLLM, generates N=8 samples per arm per instance,
and writes JSONL output files. Supports resume-on-restart.

Usage:
    python scripts/02_run_inference.py --model tulu_sft_no_safety
    python scripts/02_run_inference.py --model tulu_sft_full
    python scripts/02_run_inference.py --model tulu_dpo
    python scripts/02_run_inference.py --model tulu_final

    Options:
      --config  path to config YAML (default: config/mvp_config.yaml)
      --seed    base random seed (default: 88)

Input:
    data/dataset_index.json

Output:
    outputs/{model_name}/hinted.jsonl    — 200*8 = 1600 records
    outputs/{model_name}/unhinted.jsonl  — 1600 records
    outputs/{model_name}/progress.json   — checkpoint state
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.dataset import ensure_hf_auth, load_dataset_index, resolve_paths
from src.inference import run_inference


def main():
    parser = argparse.ArgumentParser(description="Run inference for one model.")
    parser.add_argument("--model", required=True,
                        help="Model name as defined in config (e.g. tulu_sft_no_safety)")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--dataset", default="mmlu",
                        help="Dataset key (default: mmlu). Namespaces I/O paths.")
    parser.add_argument("--seed", type=int, default=88)
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    config_path = root / args.config

    with open(config_path) as f:
        config = yaml.safe_load(f)

    ensure_hf_auth()  # log in for gated model downloads (env var or .hf_token)

    # Resolve model entry
    model_cfg = next((m for m in config["models"] if m["name"] == args.model), None)
    if model_cfg is None:
        available = [m["name"] for m in config["models"]]
        print(f"ERROR: model '{args.model}' not found in config. Available: {available}")
        sys.exit(1)

    paths = resolve_paths(root, config, args.dataset)
    dataset_path = paths["data_index"]
    if not dataset_path.exists():
        print(f"ERROR: dataset_index.json not found at {dataset_path}.")
        print(f"Run 01_prepare_dataset.py --dataset {args.dataset} first.")
        sys.exit(1)

    dataset = load_dataset_index(dataset_path)
    print(f"Loaded {len(dataset)} instances from {dataset_path}.")

    # Resolve relative output paths to be relative to MVP/ root
    config["paths"]["outputs_dir"] = str(paths["outputs_dir"])

    run_inference(config, model_cfg, dataset, base_seed=args.seed)


if __name__ == "__main__":
    main()
