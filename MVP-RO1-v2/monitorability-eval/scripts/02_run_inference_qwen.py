#!/usr/bin/env python3
"""
Step 2: Run hinted and unhinted inference for one Qwen3 (thinking-mode) checkpoint.

Uses src/inference_qwen.py so the model's `<think>...</think>` reasoning is
generated (thinking mode) and handled correctly, with Qwen3-recommended sampling
read from the `qwen_inference:` block of the config.

Usage:
    # single dataset
    python scripts/02_run_inference_qwen.py --model Qwen/Qwen3-4B --dataset mmlu_redux

    # multiple datasets — model loaded ONCE, weights shared across all datasets
    python scripts/02_run_inference_qwen.py --model Qwen/Qwen3-4B --datasets mmlu_redux gpqa_diamond wmdp

Options:
    --config        path to config YAML (default: config/mvp_config.yaml)
    --seed          base random seed (default: 88)
    --dataset       single dataset key (default: mmlu_redux)
    --datasets      one or more dataset keys; overrides --dataset when given
    --hint-type     hint shown in the hinted arm (default: config hint.default).
                    One of: professor, visual_pattern, metadata, wrong_few_shot,
                    grader_hacking, unethical. Output is namespaced by hint type.
    --n-samples     samples (N) per arm per instance (default: config
                    qwen_inference.n_samples). No fixed cap.

Input:
    data/{dataset}/dataset_index.json

Output:
    outputs/{model_name}/{dataset}/{hint_type}/hinted.jsonl
    outputs/{model_name}/{dataset}/{hint_type}/unhinted.jsonl
    outputs/{model_name}/{dataset}/{hint_type}/progress.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from src.dataset import ensure_hf_auth, load_dataset_index, model_dir_name, resolve_paths
from src.hints import HINT_TYPES
from src.inference_qwen import load_qwen_model, run_inference_qwen


def main():
    parser = argparse.ArgumentParser(description="Run Qwen3 (thinking) inference for one model.")
    parser.add_argument("--model", required=True,
                        help="HF model ID (e.g. Qwen/Qwen3-4B) or a local model dir. "
                             "Output dirs are namespaced by its folder-safe name ('/' -> '_').")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--dataset", default="mmlu_redux",
                        help="Single dataset key (default: mmlu_redux). Ignored when --datasets is given.")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="One or more dataset keys. Model is loaded once and reused across all.")
    parser.add_argument("--hint-type", default=None, choices=HINT_TYPES,
                        help="Hint shown in the hinted arm (default: config hint.default). "
                             "Output is namespaced by hint type. Choices: " + ", ".join(HINT_TYPES))
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Samples (N) per arm per instance (default: config "
                             "qwen_inference.n_samples). Overrides the config value for this run.")
    parser.add_argument("--seed", type=int, default=88)
    args = parser.parse_args()

    datasets = args.datasets if args.datasets else [args.dataset]

    root = Path(__file__).resolve().parents[2]
    config_path = root / args.config

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if "qwen_inference" not in config:
        print("ERROR: config is missing the 'qwen_inference:' block required by this script.")
        sys.exit(1)

    hint_type = args.hint_type or config.get("hint", {}).get("default", "professor")
    print(f"Hint type: {hint_type}")

    if args.n_samples is not None:
        if args.n_samples < 1:
            print(f"ERROR: --n-samples must be >= 1 (got {args.n_samples}).")
            sys.exit(1)
        config["qwen_inference"]["n_samples"] = args.n_samples
    print(f"N samples/arm: {config['qwen_inference']['n_samples']}")

    ensure_hf_auth()  # log in for gated model downloads (env var or config hf_token)

    model_cfg = {"name": model_dir_name(args.model), "hf_id": args.model}

    # Validate all dataset indexes exist before loading the model.
    for ds in datasets:
        paths = resolve_paths(root, config, ds)
        if not paths["data_index"].exists():
            print(f"ERROR: dataset_index.json not found at {paths['data_index']}.")
            print(f"Run 01_prepare_dataset.py --dataset {ds} first.")
            sys.exit(1)

    # Load model once — reused across all datasets.
    tokenizer, llm = load_qwen_model(model_cfg)

    for ds in datasets:
        paths = resolve_paths(root, config, ds)
        dataset = load_dataset_index(paths["data_index"])
        print(f"\n{'='*60}")
        print(f"Dataset: {ds}  ({len(dataset)} instances from {paths['data_index']})")
        print(f"{'='*60}")
        ds_config = dict(config, paths=dict(config["paths"], outputs_dir=str(paths["outputs_dir"])))
        run_inference_qwen(ds_config, model_cfg, dataset, base_seed=args.seed,
                           tokenizer=tokenizer, llm=llm, hint_type=hint_type)


if __name__ == "__main__":
    main()
