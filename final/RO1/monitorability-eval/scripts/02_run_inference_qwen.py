#!/usr/bin/env python3
"""
Step 2: Run hinted and unhinted inference for one Qwen3 (thinking-mode) checkpoint.

Uses src/inference_qwen.py so the model's `<think>...</think>` reasoning is
generated (thinking mode) and handled correctly, with Qwen3-recommended sampling
read from the `qwen_inference:` block of the config.

Usage:
    # single dataset
    python scripts/02_run_inference_qwen.py --model Qwen/Qwen3-4B --dataset mmlu_pro

    # multiple datasets — model loaded ONCE, weights shared across all datasets
    python scripts/02_run_inference_qwen.py --model Qwen/Qwen3-4B --datasets mmlu_pro gpqa_diamond wmdp

Options:
    --config        path to config YAML (default: config/mvp_config.yaml)
    --seed          base random seed (default: 88)
    --dataset       single dataset key (default: mmlu_pro)
    --datasets      one or more dataset keys; overrides --dataset when given
    --hint-type     hint shown in the hinted arm (default: config hint.default).
                    One of: professor, visual_pattern, metadata, wrong_few_shot,
                    grader_hacking, unethical. Output is namespaced by hint type.
    --n-samples     samples (N) per arm per instance (default: config
                    qwen_inference.n_samples). No fixed cap.
    --merge         default False. Treat --model as a LoRA ADAPTER repo
                    (adapter-only push, no config.json): merge it into the fixed
                    base from the config `merge:` block (always the same base),
                    save the merged model locally under merge.out_root, and run
                    inference on that. Output dirs stay namespaced by the
                    ADAPTER id, so downstream steps use the same --model. The
                    base model's HF cache and the local merged dir are both
                    freed automatically once vLLM has loaded them (see
                    src/merge.py) — a deliberate disk-over-time tradeoff for
                    small container disks (e.g. RunPod's default 20-30GB).

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
from src.dataset import (SPLITS, assert_splits_disjoint, ensure_hf_auth,
                         load_dataset_index, model_dir_name, resolve_paths)
from src.hints import HINT_TYPES
from src.inference_qwen import load_qwen_model, run_inference_qwen, unhinted_arm_complete


def main():
    parser = argparse.ArgumentParser(description="Run Qwen3 (thinking) inference for one model.")
    parser.add_argument("--model", required=True,
                        help="HF model ID (e.g. Qwen/Qwen3-4B) or a local model dir. "
                             "Output dirs are namespaced by its folder-safe name ('/' -> '_').")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--dataset", default="mmlu_pro",
                        help="Single dataset key (default: mmlu_pro). Ignored when --datasets is given.")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="One or more dataset keys. Model is loaded once and reused across all.")
    parser.add_argument("--hint-type", default=None, choices=HINT_TYPES,
                        help="Hint shown in the hinted arm (default: config hint.default). "
                             "Output is namespaced by hint type. Choices: " + ", ".join(HINT_TYPES))
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Samples (N) per arm per instance (default: config "
                             "qwen_inference.n_samples). Overrides the config value for this run.")
    parser.add_argument("--merge", action="store_true", default=False,
                        help="Treat --model as a LoRA adapter repo and merge it into the "
                             "fixed base from the config 'merge:' block before inference. "
                             "Outputs stay namespaced by the adapter id.")
    parser.add_argument("--data", default="test", choices=SPLITS,
                        help="Data split to run on: test (default) or eval. Selects "
                             "data/{ds}[/eval]/dataset_index.json and namespaces all "
                             "outputs by the split.")
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

    # Validate all dataset indexes exist before any model work. For the eval
    # split, also hard-fail if it overlaps the test split (guards against a
    # test index regenerated after the eval index was sampled).
    for ds in datasets:
        paths = resolve_paths(root, config, ds, split=args.data)
        if not paths["data_index"].exists():
            print(f"ERROR: dataset_index.json not found at {paths['data_index']}.")
            print(f"Run 01_prepare_dataset.py --dataset {ds} --data {args.data} first.")
            sys.exit(1)
        if args.data == "eval":
            test_path = resolve_paths(root, config, ds, split="test")["data_index"]
            if test_path.exists():
                assert_splits_disjoint(load_dataset_index(test_path),
                                       load_dataset_index(paths["data_index"]), ds)
                print(f"{ds}: eval split verified disjoint from the test split.")

    # With --merge, args.model is an adapter-only repo: merge it into the fixed
    # base (config merge.base — the same base for every adapter) and point vLLM
    # at the local merged dir. The output namespace stays the ADAPTER id so all
    # downstream steps (03–09) are run with the same --model.
    model_id = args.model
    if args.merge:
        from src.merge import merge_adapter
        merge_cfg = config.get("merge", {})
        if not merge_cfg.get("base"):
            print("ERROR: --merge requires a 'merge:' block with 'base:' in the config.")
            sys.exit(1)
        merged_dir = merge_adapter(
            base=merge_cfg["base"],
            adapter=args.model,
            out_dir=root / merge_cfg.get("out_root", "merged-models") / model_dir_name(args.model),
        )
        model_id = str(merged_dir)

    model_cfg = {"name": model_dir_name(args.model), "hf_id": model_id}

    # Load model once — reused across all datasets.
    tokenizer, llm = load_qwen_model(model_cfg)

    # vLLM has now loaded the merged weights into GPU memory — the on-disk
    # merged dir is redundant. Free it so it doesn't sit around filling a
    # small container disk. A re-run of --merge for this adapter just
    # re-merges (slower, not broken).
    if args.merge:
        import shutil
        shutil.rmtree(merged_dir, ignore_errors=True)
        print(f"Freed local merged model dir {merged_dir} (already loaded into vLLM).")

    for ds in datasets:
        # Namespace outputs model-first: outputs/{model}/{ds}/{hint}/ — the
        # canonical layout the monitor/results phase reads (resolve_paths).
        paths = resolve_paths(root, config, ds, model_cfg["name"], hint_type, split=args.data)
        dataset = load_dataset_index(paths["data_index"])
        print(f"\n{'='*60}")
        print(f"Dataset: {ds}  ({len(dataset)} instances from {paths['data_index']})")
        print(f"{'='*60}")

        # The unhinted arm is hint-type-invariant (identical prompt + seeds), so
        # reuse a COMPLETE unhinted.jsonl from any sibling hint's run of this
        # (model, dataset, split) instead of regenerating it — halves the cost
        # of every hint run after the first. Also accept a gzipped sibling
        # (unhinted.jsonl.gz — large runs are stored gzipped in git for the
        # GitHub 100MB limit, so a fresh clone may only have the .gz).
        n_samp = config["qwen_inference"]["n_samples"]
        shared_unhinted = None
        for other_hint in HINT_TYPES:
            if other_hint == hint_type:
                continue
            base = resolve_paths(root, config, ds, model_cfg["name"], other_hint,
                                 split=args.data)["outputs_dir"] / "unhinted.jsonl"
            for cand in (base, base.with_name("unhinted.jsonl.gz")):
                if unhinted_arm_complete(cand, dataset, n_samp):
                    shared_unhinted = cand
                    break
            if shared_unhinted is not None:
                break

        ds_config = dict(config, paths=dict(config["paths"], outputs_dir=str(paths["outputs_dir"])))
        run_inference_qwen(ds_config, model_cfg, dataset, base_seed=args.seed,
                           tokenizer=tokenizer, llm=llm, hint_type=hint_type,
                           shared_unhinted=shared_unhinted)


if __name__ == "__main__":
    main()
