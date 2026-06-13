#!/usr/bin/env python3
"""
Step 2 (Qwen3 variant): Run hinted and unhinted inference for one Qwen3 checkpoint.

Identical role to scripts/02_run_inference.py, but uses src/inference_qwen.py so
the model's `<think>...</think>` reasoning is generated (enable_thinking) and
handled correctly, with Qwen3-recommended sampling read from the `qwen_inference:`
block of the config.

Use this script for Qwen3 (and other thinking models); use 02_run_inference.py
for the Llama/Tulu checkpoints. Steps 03–06 (monitor, metrics, analysis, excel)
are model-agnostic and the same for both.

Usage:
    python scripts/02_run_inference_qwen.py --model qwen3_4b

Options:
    --config  path to config YAML (default: config/mvp_config.yaml)
    --seed    base random seed (default: 42)

Input:
    data/dataset_index.json

Output:
    outputs/{model_name}/hinted.jsonl
    outputs/{model_name}/unhinted.jsonl
    outputs/{model_name}/progress.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.dataset import load_dataset_index
from src.inference_qwen import run_inference_qwen


def main():
    parser = argparse.ArgumentParser(description="Run Qwen3 (thinking) inference for one model.")
    parser.add_argument("--model", required=True,
                        help="Model name as defined in config (e.g. qwen3_4b)")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    config_path = root / args.config

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if "qwen_inference" not in config:
        print("ERROR: config is missing the 'qwen_inference:' block required by this script.")
        sys.exit(1)

    model_cfg = next((m for m in config["models"] if m["name"] == args.model), None)
    if model_cfg is None:
        available = [m["name"] for m in config["models"]]
        print(f"ERROR: model '{args.model}' not found in config. Available: {available}")
        sys.exit(1)

    dataset_path = root / config["paths"]["data_dir"] / "dataset_index.json"
    if not dataset_path.exists():
        print(f"ERROR: dataset_index.json not found at {dataset_path}.")
        print("Run 01_prepare_dataset.py first.")
        sys.exit(1)

    dataset = load_dataset_index(dataset_path)
    print(f"Loaded {len(dataset)} instances from {dataset_path}.")

    config["paths"]["outputs_dir"] = str(root / config["paths"]["outputs_dir"])

    run_inference_qwen(config, model_cfg, dataset, base_seed=args.seed)


if __name__ == "__main__":
    main()
