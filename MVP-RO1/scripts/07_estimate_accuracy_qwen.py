#!/usr/bin/env python3
"""
Step 7 (Qwen3 variant): Estimate unhinted accuracy for one Qwen3 checkpoint.

Same diagnostic as scripts/07_estimate_accuracy.py, but uses the Qwen3 thinking
prompt/sampling (src/inference_qwen.py) so the accuracy figure is measured under
the same generation regime as the main Qwen inference run. Reads sampling from
the `qwen_inference:` config block.

Usage:
    python scripts/07_estimate_accuracy_qwen.py --model qwen3_4b
    python scripts/07_estimate_accuracy_qwen.py --model qwen3_4b --n 32 --seed 2000

Output:
    analysis/{model}/accuracy_runs.jsonl   — one record per instance
    analysis/{model}/accuracy_summary.json — dataset-level summary
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.dataset import MMLU_LETTERS, load_dataset_index
from src.inference_qwen import format_prompt_qwen, parse_answer_qwen


def answer_entropy(answers: list, truncated: list) -> float | None:
    """Normalized Shannon entropy (bits, [0,1]) of the answer distribution over
    {A,B,C,D}, computed from valid (non-truncated, parseable) runs only."""
    valid = [a for a, t in zip(answers, truncated) if a is not None and not t]
    if not valid:
        return None

    n = len(valid)
    h = 0.0
    for letter in MMLU_LETTERS:
        count = valid.count(letter)
        if count == 0:
            continue
        p = count / n
        h -= p * math.log2(p)

    return h / math.log2(len(MMLU_LETTERS))


def main():
    parser = argparse.ArgumentParser(description="Estimate Qwen3 unhinted accuracy with N runs.")
    parser.add_argument("--model",  required=True, help="Model name from config")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--n",      type=int, default=32, help="Samples per instance")
    parser.add_argument("--seed",   type=int, default=2000, help="Base random seed")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    with open(root / args.config) as f:
        config = yaml.safe_load(f)

    if "qwen_inference" not in config:
        print("ERROR: config is missing the 'qwen_inference:' block required by this script.")
        sys.exit(1)

    models = {m["name"]: m for m in config["models"]}
    if args.model not in models:
        print(f"ERROR: model '{args.model}' not in config. Available: {list(models)}")
        sys.exit(1)
    model_cfg = models[args.model]

    dataset_path = root / config["paths"]["data_dir"] / "dataset_index.json"
    dataset = load_dataset_index(dataset_path)

    analysis_dir = root / config["paths"]["analysis_dir"] / args.model
    analysis_dir.mkdir(parents=True, exist_ok=True)

    runs_path     = analysis_dir / "accuracy_runs.jsonl"
    summary_path  = analysis_dir / "accuracy_summary.json"
    progress_path = analysis_dir / "accuracy_progress.json"

    completed: set = set()
    if progress_path.exists():
        with open(progress_path) as f:
            completed = set(json.load(f).get("completed_instance_ids", []))
        print(f"Resuming: {len(completed)} instances already done.")

    todo = [item for item in dataset if item["instance_id"] not in completed]
    print(f"{len(todo)} instances to process for {args.model} (N={args.n} per instance).")

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    hf_id = model_cfg["hf_id"]
    inf   = config["qwen_inference"]
    enable_thinking = inf.get("enable_thinking", True)

    print(f"Loading tokenizer: {hf_id}")
    tokenizer = AutoTokenizer.from_pretrained(hf_id)

    print(f"Loading model via vLLM: {hf_id}")
    llm = LLM(model=hf_id, dtype="bfloat16", trust_remote_code=True)

    n_samples        = args.n
    base_seed        = args.seed
    checkpoint_every = inf["checkpoint_every"]

    def _seed(item, s):
        return base_seed + item["instance_id"] * n_samples + s

    null_count      = 0
    truncated_count = 0

    for batch_start in range(0, len(todo), checkpoint_every):
        batch = todo[batch_start : batch_start + checkpoint_every]

        prompts = [
            format_prompt_qwen(item, include_hint=False, tokenizer=tokenizer, enable_thinking=enable_thinking)
            for item in batch
        ]

        per_item_outputs = [[] for _ in batch]

        for s in range(n_samples):
            sp_s = [
                SamplingParams(
                    n=1,
                    temperature=inf["temperature"],
                    top_p=inf["top_p"],
                    max_tokens=inf["max_new_tokens"],
                    seed=_seed(item, s),
                )
                for item in batch
            ]
            print(
                f"  Batch [{batch_start}:{batch_start+len(batch)}] "
                f"— sample {s+1}/{n_samples} ..."
            )
            outputs = llm.generate(prompts, sp_s)

            for i, out in enumerate(outputs):
                completion = out.outputs[0]
                trunc = completion.finish_reason == "length"
                per_item_outputs[i].append((completion.text, trunc))

        with open(runs_path, "a", encoding="utf-8") as f:
            for item, item_outputs in zip(batch, per_item_outputs):
                answers   = []
                truncated = []
                for raw, trunc in item_outputs:
                    letter = parse_answer_qwen(raw) if not trunc else None
                    answers.append(letter)
                    truncated.append(trunc)
                    if letter is None:
                        null_count += 1
                    if trunc:
                        truncated_count += 1

                correct = item["correct_answer"]
                n_valid   = sum(1 for a, t in zip(answers, truncated) if a is not None and not t)
                n_correct = sum(1 for a, t in zip(answers, truncated) if a == correct and not t)
                accuracy  = n_correct / n_valid if n_valid > 0 else None
                entropy   = answer_entropy(answers, truncated)

                record = {
                    "instance_id":    item["instance_id"],
                    "question_id":    item["question_id"],
                    "correct_answer": correct,
                    "answers":        answers,
                    "n_total":        n_samples,
                    "n_valid":        n_valid,
                    "n_correct":      n_correct,
                    "accuracy":       accuracy,
                    "answer_entropy": entropy,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        completed.update(item["instance_id"] for item in batch)
        with open(progress_path, "w") as f:
            json.dump({"completed_instance_ids": sorted(completed)}, f)
        print(f"  Checkpoint: {len(completed)}/{len(dataset)} instances done.")

    records = []
    with open(runs_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    acc_vals = [r["accuracy"] for r in records if r["accuracy"] is not None]
    mean_acc  = sum(acc_vals) / len(acc_vals) if acc_vals else None

    total_valid   = sum(r["n_valid"]   for r in records)
    total_correct = sum(r["n_correct"] for r in records)
    pooled_acc    = total_correct / total_valid if total_valid > 0 else None

    entropy_vals = [r["answer_entropy"] for r in records if r["answer_entropy"] is not None]
    mean_entropy = sum(entropy_vals) / len(entropy_vals) if entropy_vals else None

    summary = {
        "model_name":         args.model,
        "hf_id":              model_cfg["hf_id"],
        "n_instances":        len(records),
        "n_samples_per_instance": n_samples,
        "base_seed":          base_seed,
        "mean_accuracy":      mean_acc,
        "pooled_accuracy":    pooled_acc,
        "mean_answer_entropy": mean_entropy,
        "total_valid_runs":   total_valid,
        "total_correct_runs": total_correct,
        "total_null":         null_count,
        "total_truncated":    truncated_count,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone.")
    print(f"  Mean accuracy (macro):  {mean_acc:.4f}" if mean_acc is not None else "  Mean accuracy: N/A")
    print(f"  Pooled accuracy (micro): {pooled_acc:.4f}" if pooled_acc is not None else "  Pooled accuracy: N/A")
    print(f"  Mean answer entropy:    {mean_entropy:.4f}" if mean_entropy is not None else "  Mean answer entropy: N/A")
    print(f"  Written: {runs_path}")
    print(f"  Written: {summary_path}")


if __name__ == "__main__":
    main()
