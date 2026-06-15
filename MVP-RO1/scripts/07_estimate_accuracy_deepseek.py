#!/usr/bin/env python3
"""
Step 7 (DeepSeek-R1-Distill variant): Estimate per-instance accuracy and
hint-following for one DeepSeek-R1-Distill checkpoint, using N=32 runs in
BOTH the unhinted and hinted conditions, then correlate the two.

Same analysis as scripts/07_estimate_accuracy.py, but uses the DeepSeek
thinking prompt/sampling (src/inference_deepseek.py) — no system prompt,
forced opening `<think>`, sampling from the `deepseek_inference:` config
block — so the figures are measured under the same generation regime as the
main DeepSeek run.

For each instance:
  - the unhinted prompt is run --n times -> acc_u (fraction correct) and
    p0 (fraction choosing hint_target, the always-incorrect option the hint
    points to; see src/dataset.py)
  - the hinted prompt is run --n times -> p1 (fraction choosing hint_target)
  - te = p1 - p0 (hint effect, net of the base rate)

A final analysis step correlates acc_u with p1 and te (Spearman + Pearson),
splits instances into Wrong/Right/Mixed buckets by acc_u, and produces a
figure. See documentation.md, section "Hint susceptibility vs. accuracy".

Usage:
    python scripts/07_estimate_accuracy_deepseek.py --model deepseek_r1_distill_qwen_1_5b
    python scripts/07_estimate_accuracy_deepseek.py --model deepseek_r1_distill_qwen_1_5b --n 32 --seed 2000

Output:
    analysis/{model}/accuracy_runs.jsonl          — unhinted, one record per instance
    analysis/{model}/accuracy_summary.json        — unhinted dataset-level summary
    analysis/{model}/hint_accuracy_runs.jsonl     — hinted, one record per instance
    analysis/{model}/hint_accuracy_summary.json   — hinted dataset-level summary
    analysis/{model}/hint_susceptibility_summary.json  — correlations, buckets, MWU
    analysis/{model}/hint_susceptibility_summary.md    — human-readable report
    analysis/{model}/hint_susceptibility_plot.png      — scatter + binned bar charts
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import yaml
from src.dataset import MMLU_LETTERS, load_dataset_index
from src.gmean2_metrics import load_jsonl
from src.hint_susceptibility import (
    compute_hint_susceptibility,
    plot_hint_susceptibility,
    write_hint_susceptibility_md,
)
from src.inference_deepseek import format_prompt_deepseek
from src.inference_qwen import parse_answer_qwen


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


def build_record(item: dict, answers: list, truncated: list, n_samples: int,
                  hint_rate_field: str) -> dict:
    """Build one per-instance record from N raw (answer, truncated) runs.

    `hint_rate_field` names the field holding the fraction of valid runs that
    chose `hint_target` — "p0" for the unhinted arm, "p1" for the hinted arm.
    """
    correct = item["correct_answer"]
    hint_target = item["hint_target"]

    n_valid = sum(1 for a, t in zip(answers, truncated) if a is not None and not t)
    n_correct = sum(1 for a, t in zip(answers, truncated) if a == correct and not t)
    n_hint = sum(1 for a, t in zip(answers, truncated) if a == hint_target and not t)

    accuracy = n_correct / n_valid if n_valid > 0 else None
    hint_rate = n_hint / n_valid if n_valid > 0 else None
    entropy = answer_entropy(answers, truncated)

    record = {
        "instance_id": item["instance_id"],
        "question_id": item["question_id"],
        "correct_answer": correct,
        "hint_target": hint_target,
        "answers": answers,        # list of N letters (None if parse failed)
        "n_total": n_samples,
        "n_valid": n_valid,         # parseable, non-truncated runs
        "n_correct": n_correct,
        "n_hint_target": n_hint,    # of the valid runs, how many == hint_target
        "accuracy": accuracy,
        "answer_entropy": entropy,  # normalized Shannon entropy over A-D, [0,1]
    }
    record[hint_rate_field] = hint_rate
    return record


def run_sampling_pass(llm, SamplingParams, tokenizer, items: list, include_hint: bool,
                       n_samples: int, seed_fn, inf: dict, checkpoint_every: int,
                       runs_path: Path, progress_path: Path, hint_rate_field: str,
                       label: str) -> tuple[int, int]:
    """Run --n samples per item for one arm (hinted or unhinted), writing one
    JSONL record per instance to `runs_path`. Supports resume via
    `progress_path`. Returns (null_count, truncated_count)."""

    completed: set = set()
    if progress_path.exists():
        with open(progress_path) as f:
            completed = set(json.load(f).get("completed_instance_ids", []))
        print(f"  [{label}] Resuming: {len(completed)} instances already done.")

    todo = [item for item in items if item["instance_id"] not in completed]
    print(f"  [{label}] {len(todo)} instances to process (N={n_samples} per instance).")

    null_count = 0
    truncated_count = 0

    for batch_start in range(0, len(todo), checkpoint_every):
        batch = todo[batch_start: batch_start + checkpoint_every]

        prompts = [
            format_prompt_deepseek(item, include_hint=include_hint, tokenizer=tokenizer)
            for item in batch
        ]

        # per_item_outputs[i][s] = (raw_text, truncated)
        per_item_outputs = [[] for _ in batch]

        for s in range(n_samples):
            sp_s = [
                SamplingParams(
                    n=1,
                    temperature=inf["temperature"],
                    top_p=inf["top_p"],
                    max_tokens=inf["max_new_tokens"],
                    seed=seed_fn(item, s),
                )
                for item in batch
            ]
            print(f"  [{label}] Batch [{batch_start}:{batch_start+len(batch)}] "
                  f"— sample {s+1}/{n_samples} ...")
            outputs = llm.generate(prompts, sp_s)

            for i, out in enumerate(outputs):
                completion = out.outputs[0]
                trunc = completion.finish_reason == "length"
                per_item_outputs[i].append((completion.text, trunc))

        with open(runs_path, "a", encoding="utf-8") as f:
            for item, item_outputs in zip(batch, per_item_outputs):
                answers, truncated = [], []
                for raw, trunc in item_outputs:
                    # parse_answer_qwen handles a completion that starts inside the
                    # <think> block (the forced opening tag is in the prompt).
                    letter = parse_answer_qwen(raw) if not trunc else None
                    answers.append(letter)
                    truncated.append(trunc)
                    if letter is None:
                        null_count += 1
                    if trunc:
                        truncated_count += 1

                record = build_record(item, answers, truncated, n_samples, hint_rate_field)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        completed.update(item["instance_id"] for item in batch)
        with open(progress_path, "w") as f:
            json.dump({"completed_instance_ids": sorted(completed)}, f)
        print(f"  [{label}] Checkpoint: {len(completed)}/{len(items)} instances done.")

    return null_count, truncated_count


def write_arm_summary(records: list, model_name: str, hf_id: str, n_samples: int,
                       base_seed: int, hint_rate_field: str, null_count: int,
                       truncated_count: int, path: Path) -> dict:
    acc_vals = [r["accuracy"] for r in records if r["accuracy"] is not None]
    mean_acc = float(np.mean(acc_vals)) if acc_vals else None

    total_valid = sum(r["n_valid"] for r in records)
    total_correct = sum(r["n_correct"] for r in records)
    total_hint = sum(r["n_hint_target"] for r in records)
    pooled_acc = total_correct / total_valid if total_valid > 0 else None
    pooled_hint_rate = total_hint / total_valid if total_valid > 0 else None

    hint_vals = [r[hint_rate_field] for r in records if r.get(hint_rate_field) is not None]
    mean_hint_rate = float(np.mean(hint_vals)) if hint_vals else None

    entropy_vals = [r["answer_entropy"] for r in records if r["answer_entropy"] is not None]
    mean_entropy = float(np.mean(entropy_vals)) if entropy_vals else None

    summary = {
        "model_name": model_name,
        "hf_id": hf_id,
        "n_instances": len(records),
        "n_samples_per_instance": n_samples,
        "base_seed": base_seed,
        "mean_accuracy": mean_acc,        # macro: per-instance accuracy averaged equally
        "pooled_accuracy": pooled_acc,    # micro: all valid runs pooled
        f"mean_{hint_rate_field}": mean_hint_rate,
        f"pooled_{hint_rate_field}": pooled_hint_rate,
        "mean_answer_entropy": mean_entropy,
        "total_valid_runs": total_valid,
        "total_correct_runs": total_correct,
        "total_hint_target_runs": total_hint,
        "total_null": null_count,
        "total_truncated": truncated_count,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Estimate DeepSeek-R1-Distill unhinted/hinted accuracy and hint-following with N=32 runs."
    )
    parser.add_argument("--model", required=True, help="Model name from config")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--n", type=int, default=32, help="Samples per instance, per arm")
    parser.add_argument("--seed", type=int, default=2000,
                        help="Base random seed for the unhinted arm")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    with open(root / args.config) as f:
        config = yaml.safe_load(f)

    if "deepseek_inference" not in config:
        print("ERROR: config is missing the 'deepseek_inference:' block required by this script.")
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

    unhinted_runs_path = analysis_dir / "accuracy_runs.jsonl"
    unhinted_summary_path = analysis_dir / "accuracy_summary.json"
    unhinted_progress_path = analysis_dir / "accuracy_progress.json"

    hinted_runs_path = analysis_dir / "hint_accuracy_runs.jsonl"
    hinted_summary_path = analysis_dir / "hint_accuracy_summary.json"
    hinted_progress_path = analysis_dir / "hint_accuracy_progress.json"

    susceptibility_json_path = analysis_dir / "hint_susceptibility_summary.json"
    susceptibility_md_path = analysis_dir / "hint_susceptibility_summary.md"
    susceptibility_plot_path = analysis_dir / "hint_susceptibility_plot.png"

    # Lazy vLLM import — only runs on GPU
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    hf_id = model_cfg["hf_id"]
    inf = config["deepseek_inference"]

    print(f"Loading tokenizer: {hf_id}")
    tokenizer = AutoTokenizer.from_pretrained(hf_id)

    print(f"Loading model via vLLM: {hf_id}")
    llm = LLM(model=hf_id, dtype="bfloat16", trust_remote_code=True)

    n_samples = args.n
    base_seed = args.seed
    checkpoint_every = inf["checkpoint_every"]

    def seed_unhinted(item, s):
        return base_seed + item["instance_id"] * n_samples + s

    def seed_hinted(item, s):
        # Disjoint seed space from the unhinted arm so the two arms are
        # independent samples.
        return base_seed + 1_000_000 + item["instance_id"] * n_samples + s

    print(f"\n=== Pass 1/2: unhinted (N={n_samples}) ===")
    u_null, u_trunc = run_sampling_pass(
        llm, SamplingParams, tokenizer, dataset, include_hint=False,
        n_samples=n_samples, seed_fn=seed_unhinted, inf=inf,
        checkpoint_every=checkpoint_every,
        runs_path=unhinted_runs_path, progress_path=unhinted_progress_path,
        hint_rate_field="p0", label="unhinted",
    )

    print(f"\n=== Pass 2/2: hinted (N={n_samples}) ===")
    h_null, h_trunc = run_sampling_pass(
        llm, SamplingParams, tokenizer, dataset, include_hint=True,
        n_samples=n_samples, seed_fn=seed_hinted, inf=inf,
        checkpoint_every=checkpoint_every,
        runs_path=hinted_runs_path, progress_path=hinted_progress_path,
        hint_rate_field="p1", label="hinted",
    )

    # --- Per-arm summaries ----------------------------------------------------
    unhinted_records = load_jsonl(unhinted_runs_path)
    hinted_records = load_jsonl(hinted_runs_path)

    unhinted_summary = write_arm_summary(
        unhinted_records, args.model, hf_id, n_samples, base_seed,
        hint_rate_field="p0", null_count=u_null, truncated_count=u_trunc,
        path=unhinted_summary_path,
    )
    hinted_summary = write_arm_summary(
        hinted_records, args.model, hf_id, n_samples, base_seed,
        hint_rate_field="p1", null_count=h_null, truncated_count=h_trunc,
        path=hinted_summary_path,
    )

    # --- Hint susceptibility analysis ------------------------------------------
    result = compute_hint_susceptibility(unhinted_records, hinted_records)
    result["model_name"] = args.model
    result["n_samples"] = n_samples
    result["plot_filename"] = susceptibility_plot_path.name

    with open(susceptibility_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    write_hint_susceptibility_md(result, susceptibility_md_path)
    if result["n_total"] > 0:
        plot_hint_susceptibility(result, susceptibility_plot_path)

    # --- Print results --------------------------------------------------------
    print("\nDone.")
    print(f"  Unhinted: mean accuracy (macro)={unhinted_summary['mean_accuracy']:.4f}  "
          f"mean p0={unhinted_summary['mean_p0']:.4f}"
          if unhinted_summary['mean_accuracy'] is not None else "  Unhinted: N/A")
    print(f"  Hinted:   mean accuracy (macro)={hinted_summary['mean_accuracy']:.4f}  "
          f"mean p1={hinted_summary['mean_p1']:.4f}"
          if hinted_summary['mean_accuracy'] is not None else "  Hinted: N/A")

    c_p1 = result["correlations"]["acc_u_vs_p1"]
    c_te = result["correlations"]["acc_u_vs_te"]
    print(f"  acc_u vs p1: Spearman rho={c_p1['spearman_r']}  p={c_p1['spearman_p']}  n={c_p1['n']}"
          if c_p1["spearman_r"] is not None else "  acc_u vs p1: N/A (insufficient variance)")
    print(f"  acc_u vs te: Spearman rho={c_te['spearman_r']}  p={c_te['spearman_p']}  n={c_te['n']}"
          if c_te["spearman_r"] is not None else "  acc_u vs te: N/A (insufficient variance)")

    print(f"\n  Written: {unhinted_runs_path}")
    print(f"  Written: {unhinted_summary_path}")
    print(f"  Written: {hinted_runs_path}")
    print(f"  Written: {hinted_summary_path}")
    print(f"  Written: {susceptibility_json_path}")
    print(f"  Written: {susceptibility_md_path}")
    if result["n_total"] > 0:
        print(f"  Written: {susceptibility_plot_path}")


if __name__ == "__main__":
    main()
