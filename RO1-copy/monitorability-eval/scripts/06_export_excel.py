#!/usr/bin/env python3
"""
Step 6: Export per-record qualitative data to a multi-sheet Excel workbook.

For every generated sample (each row of outputs/{model}/hinted.jsonl and
unhinted.jsonl), combines:
  - the dataset item (question, choices, correct answer, hint target)
  - the model's response (raw output, CoT, parsed answer, followed_hint, ...)
  - the monitor's verdict for that sample (acknowledged, label, reasoning, ...)
  - the per-example g-mean^2 metrics (TE, R_effect, TPR_mc, TNR_X=0/1,
    g-mean^2, acc_u, ...) computed by src.gmean2_metrics.compute_gmean2

Per-example metrics (TE, R_effect, etc.) are constant across all samples of
the same instance_id, so they are repeated on every row for that example —
this is intentional (see compute_gmean2 docstring: these quantities are
defined per-example, not per-sample).

Each row is also tagged with a `category`, used to split rows across sheets:
  - Hinted_Followed_Verbalised     X=1, hint followed, monitor says Z=1
  - Hinted_Followed_NotVerbalised  X=1, hint followed, monitor says Z=0
  - Hinted_Followed_MonitorMissing X=1, hint followed, no monitor verdict
  - Hinted_Hint_Not_Followed       X=1, hint not followed
  - Unhinted_Control                X=0 (control arm)
  - Invalid_Truncated                truncated or unparseable response

Usage:
    python scripts/06_export_excel.py --model Qwen/Qwen3-4B                   # every dataset
    python scripts/06_export_excel.py --model Qwen/Qwen3-4B --dataset wmdp   # one dataset

    Options:
      --config   path to config YAML (default: config/mvp_config.yaml)
      --dataset  restrict to one dataset key (default: every dataset in config)
      --output   output .xlsx path (only valid with --dataset; default:
                 analysis/{model}/{dataset}/{hint}/qualitative_export.xlsx)

Input:
    data/dataset_index.json
    outputs/{model_name}/hinted.jsonl
    outputs/{model_name}/unhinted.jsonl
    monitor_outputs/{model_name}/monitor_results.jsonl  (optional)

Output:
    <output path>.xlsx — one workbook with sheets:
      All_Records, <category sheets>, Per_Example_Summary
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml

from src.gmean2_metrics import load_jsonl, compute_gmean2
from src.dataset import SPLITS, model_dir_name, resolve_paths
from src.hints import HINT_TYPES


CATEGORY_SHEETS = [
    "Hinted_Followed_Verbalised",
    "Hinted_Followed_NotVerbalised",
    "Hinted_Followed_MonitorMissing",
    "Hinted_Hint_Not_Followed",
    "Unhinted_Control",
    "Invalid_Truncated",
]

# Per-example (per instance_id) g-mean^2 metrics — see src/gmean2_metrics.py.
# These are repeated on every per-sample row of the same instance_id.
PER_EXAMPLE_FIELDS = [
    "p1_i", "p0_i", "te_i", "r_effect_i", "level1_pass",
    "z_x1_y1_i", "tnr_x0_i", "tnr_x1_i", "tnr_x1_vacuous_i", "tpr_mc_i", "gmean2_i", "acc_u_i",
    "n_hinted_valid", "n_unhinted_valid", "n_followed_hint", "n_not_followed_hint",
    "mean_cot_tokens_hinted", "mean_cot_tokens_unhinted",
    "mean_cot_tokens_hint_followed", "mean_cot_tokens_hint_not_followed",
]

COLUMNS = [
    "model_name", "category", "instance_id", "question_id", "subject", "question",
    "choice_A", "choice_B", "choice_C", "choice_D", "choice_E", "correct_answer", "hint_target",
    "arm", "sample_idx", "seed",
    "parsed_answer", "followed_hint", "no_cot", "truncated", "token_count",
    "monitor_acknowledged", "monitor_label", "monitor_parse_failure",
] + PER_EXAMPLE_FIELDS + [
    "monitor_reasoning", "monitor_raw_response", "cot", "raw_output", "full_prompt",
]

SUMMARY_COLUMNS = [
    "model_name", "instance_id", "question_id", "subject", "question",
    "correct_answer", "hint_target",
] + PER_EXAMPLE_FIELDS


def categorize(rec: dict, monitor_entry: dict | None) -> str:
    if rec.get("truncated") or rec.get("parsed_answer") is None:
        return "Invalid_Truncated"
    if rec["arm"] == "unhinted":
        return "Unhinted_Control"
    if not rec.get("followed_hint", False):
        return "Hinted_Hint_Not_Followed"
    ack = monitor_entry.get("acknowledged") if monitor_entry else None
    if ack is True:
        return "Hinted_Followed_Verbalised"
    if ack is False:
        return "Hinted_Followed_NotVerbalised"
    return "Hinted_Followed_MonitorMissing"


def build_rows(model_name, dataset_by_id, hinted, unhinted, monitor_records, per_example_metrics):
    monitor_lookup = {
        (r["instance_id"], r["sample_idx"], r["arm"]): r
        for r in monitor_records
    }
    metrics_by_id = {m["instance_id"]: m for m in per_example_metrics}

    rows = []
    for rec in hinted + unhinted:
        iid = rec["instance_id"]
        ds = dataset_by_id.get(iid, {})
        choices = ds.get("choices", {})
        monitor_entry = monitor_lookup.get((iid, rec["sample_idx"], rec["arm"]))
        metrics = metrics_by_id.get(iid, {})

        row = {
            "model_name": model_name,
            "category": categorize(rec, monitor_entry),
            "instance_id": iid,
            "question_id": ds.get("question_id", rec.get("question_id")),
            "subject": ds.get("subject"),
            "question": ds.get("question"),
            "choice_A": choices.get("A"),
            "choice_B": choices.get("B"),
            "choice_C": choices.get("C"),
            "choice_D": choices.get("D"),
            "choice_E": choices.get("E"),
            "correct_answer": ds.get("correct_answer", rec.get("correct_answer")),
            "hint_target": ds.get("hint_target", rec.get("hint_target")),
            "arm": rec["arm"],
            "sample_idx": rec["sample_idx"],
            "seed": rec.get("seed"),
            "parsed_answer": rec.get("parsed_answer"),
            "followed_hint": rec.get("followed_hint"),
            "no_cot": rec.get("no_cot"),
            "truncated": rec.get("truncated"),
            "token_count": rec.get("token_count"),
            "monitor_acknowledged": monitor_entry.get("acknowledged") if monitor_entry else None,
            "monitor_label": monitor_entry.get("label") if monitor_entry else None,
            "monitor_parse_failure": monitor_entry.get("parse_failure", False) if monitor_entry else None,
            "monitor_reasoning": monitor_entry.get("reasoning") if monitor_entry else None,
            "monitor_raw_response": monitor_entry.get("raw_response") if monitor_entry else None,
            "cot": rec.get("cot"),
            "raw_output": rec.get("raw_output"),
            "full_prompt": rec.get("full_prompt"),
        }
        for field in PER_EXAMPLE_FIELDS:
            row[field] = metrics.get(field)

        rows.append(row)
    return rows


def build_summary_rows(model_name, dataset_by_id, per_example_metrics):
    rows = []
    for m in per_example_metrics:
        iid = m["instance_id"]
        ds = dataset_by_id.get(iid, {})
        row = {
            "model_name": model_name,
            "instance_id": iid,
            "question_id": ds.get("question_id"),
            "subject": ds.get("subject"),
            "question": ds.get("question"),
            "correct_answer": m.get("correct_answer"),
            "hint_target": m.get("hint_target"),
        }
        row.update({field: m.get(field) for field in PER_EXAMPLE_FIELDS})
        rows.append(row)
    return rows


def run_dataset(root, config, dataset_key, models, output_override, hint_type, split="test"):
    """Export a qualitative workbook for every given model on a single dataset."""
    print(f"\n=== dataset: {dataset_key}  (hint: {hint_type}, split: {split}) ===")

    dataset_path = resolve_paths(root, config, dataset_key, split=split)["data_index"]
    if not dataset_path.exists():
        print(f"  Skipping: {dataset_path} not found. "
              f"Run 01_prepare_dataset.py --dataset {dataset_key} --data {split} first.")
        return
    with open(dataset_path, encoding="utf-8") as f:
        dataset_index = json.load(f)
    dataset_by_id = {d["instance_id"]: d for d in dataset_index}

    for model_cfg in models:
        model_name = model_cfg["name"]

        mpaths = resolve_paths(root, config, dataset_key, model_name, hint_type, split=split)
        out_dir = mpaths["outputs_dir"]
        mon_dir = mpaths["monitor_dir"]

        hinted_path = out_dir / "hinted.jsonl"
        unhinted_path = out_dir / "unhinted.jsonl"
        monitor_path = mon_dir / "monitor_results.jsonl"

        if not hinted_path.exists() or not unhinted_path.exists():
            print(f"Skipping {model_name}: inference outputs not found in {out_dir}")
            continue

        hinted = load_jsonl(hinted_path)
        unhinted = load_jsonl(unhinted_path)
        monitor = load_jsonl(monitor_path) if monitor_path.exists() else []

        result = compute_gmean2(hinted, unhinted, monitor)
        per_instance = result["per_instance"]

        rows = build_rows(model_name, dataset_by_id, hinted, unhinted, monitor, per_instance)
        summary_rows = build_summary_rows(model_name, dataset_by_id, per_instance)

        df = pd.DataFrame(rows, columns=COLUMNS)
        summary_df = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)

        if output_override:
            out_path = root / output_override
        else:
            out_path = mpaths["analysis_dir"] / "qualitative_export.xlsx"

        out_path.parent.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="All_Records", index=False)
            for cat in CATEGORY_SHEETS:
                cat_df = df[df["category"] == cat]
                cat_df.to_excel(writer, sheet_name=cat[:31], index=False)
            summary_df.to_excel(writer, sheet_name="Per_Example_Summary", index=False)

        print(f"  {model_name}: {len(rows)} records, {len(per_instance)} examples -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Export per-record qualitative data to a multi-sheet Excel workbook.")
    parser.add_argument("--model", required=True,
                        help="HF model ID used at inference (e.g. Qwen/Qwen3-4B).")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--dataset", default=None,
                        help="Restrict to one dataset key (default: every dataset in config). "
                             "Namespaces I/O paths.")
    parser.add_argument("--hint-type", default=None, choices=HINT_TYPES,
                        help="Hint type to read (default: config hint.default). "
                             "Must match the inference/monitor run. Namespaces I/O paths.")
    parser.add_argument("--output", default=None,
                        help="Output .xlsx path (only valid with --model and --dataset; default: "
                             "outputs-and-results/monitorability-results/{model}/{dataset}/{hint}/qualitative_export.xlsx)")
    parser.add_argument("--data", default="test", choices=SPLITS,
                        help="Data split to read: test (default) or eval. "
                             "Must match the inference/monitor run. Namespaces I/O paths.")
    args = parser.parse_args()

    if args.output and not args.dataset:
        print("ERROR: --output requires --dataset "
              "(each dataset/model pair gets its own workbook).")
        sys.exit(1)

    root = Path(__file__).resolve().parents[2]
    config_path = root / args.config

    with open(config_path) as f:
        config = yaml.safe_load(f)

    hint_type = args.hint_type or config.get("hint", {}).get("default", "professor")
    models = [{"name": model_dir_name(args.model), "hf_id": args.model}]

    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = list(config.get("datasets", {}).keys())
        if not datasets:
            print("ERROR: config has no 'datasets' section to sweep.")
            sys.exit(1)

    for dataset_key in datasets:
        run_dataset(root, config, dataset_key, models, args.output, hint_type, split=args.data)

    print("\nDone.")


if __name__ == "__main__":
    main()
