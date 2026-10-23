#!/usr/bin/env python3
"""
Step 58 (experiment: uncertainty-vs-monitorability): per-run answer export and
answer-parse failure report.

Needs only the inference outputs (no monitor), so it can run on the GPU box
straight after inference. For every (dataset, run) it writes one row with the
answer letter the model chose, and counts the runs whose final answer letter
could not be parsed (parsed_answer is None/empty).

A run is VALID for the downstream analysis (step 59, as steps 56/57) iff its
answer was parsed AND it was not truncated at max_new_tokens. The report splits
the invalid runs so the two causes are visible:
    unparsed            parsed_answer missing (the count asked for)
      of which truncated     hit the token cap before answering
      of which not truncated finished but no letter could be parsed
    truncated but parsed     answer recovered from a truncated run (still
                             excluded from the analysis)

Usage:
    python monitorability-eval/scripts/58_uvm_answers_and_parse_report.py

    Options:
      --config     path to config YAML (default: config/uncertainty_vs_monitorability.yaml)
      --model      model id (default: Qwen/Qwen3-4B)
      --datasets   dataset keys (default: gpqa_diamond lexam wmdp)
      --hint-type  hint type (default: grader_hacking)
      --data       split: test (default) | eval

Input:
    {outputs_dir}/{model}/{dataset}/{hint}/{hinted,unhinted}.jsonl[.gz]

Output:
    {analysis_dir}/{model}/{dataset}/{hint}/answers_per_run.csv
        dataset, instance_id, question_id, arm, sample_idx, parsed_answer,
        correct_answer, hint_target, is_correct, followed_hint, truncated,
        token_count, valid
    {analysis_dir}/{model}/parse_report_{hint}.csv   one row per dataset × arm
    {analysis_dir}/{model}/parse_report_{hint}.md    same, as a Markdown table
"""

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.dataset import SPLITS, model_dir_name, resolve_paths
from src.gmean2_metrics import resolve_jsonl
from src.hints import HINT_TYPES

DEFAULT_CONFIG = "config/uncertainty_vs_monitorability.yaml"
DEFAULT_MODEL = "Qwen/Qwen3-4B"
DEFAULT_DATASETS = ["gpqa_diamond", "lexam", "wmdp"]
ARMS = ["unhinted", "hinted"]

ANSWER_FIELDS = ["dataset", "instance_id", "question_id", "arm", "sample_idx",
                 "parsed_answer", "correct_answer", "hint_target", "is_correct",
                 "followed_hint", "truncated", "token_count", "valid"]
REPORT_FIELDS = ["dataset", "arm", "runs", "valid", "unparsed",
                 "unparsed_pct", "unparsed_truncated", "unparsed_not_truncated",
                 "truncated_but_parsed", "questions",
                 "questions_lt_min_valid"]


def is_parsed(pa):
    return pa not in (None, "", "None", "N/A")


def read_jsonl(path):
    path = resolve_jsonl(path)
    if not path.exists():
        return None
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Per-run answers + answer-parse failure report.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--hint-type", default="grader_hacking",
                        choices=HINT_TYPES)
    parser.add_argument("--data", default="test", choices=SPLITS)
    parser.add_argument("--min-valid", type=int, default=27,
                        help="report how many questions fall below this many "
                             "valid unhinted runs (step 59's inclusion rule)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    with open(root / args.config) as f:
        config = yaml.safe_load(f)
    model_name = model_dir_name(args.model)
    n_samples = config["qwen_inference"]["n_samples"]

    report = []
    for ds in args.datasets:
        paths = resolve_paths(root, config, ds, model_name, args.hint_type,
                              split=args.data)
        out_rows = []
        for arm in ARMS:
            records = read_jsonl(paths["outputs_dir"] / f"{arm}.jsonl")
            if records is None:
                print(f"WARNING: no {arm} outputs for {ds} under "
                      f"{paths['outputs_dir']} — skipped.")
                continue
            valid_per_q = {}
            stats = dict(runs=0, valid=0, unparsed=0, unparsed_truncated=0,
                         unparsed_not_truncated=0, truncated_but_parsed=0)
            for r in records:
                pa = r.get("parsed_answer")
                parsed = is_parsed(pa)
                trunc = bool(r.get("truncated", False))
                valid = parsed and not trunc
                stats["runs"] += 1
                stats["valid"] += valid
                if not parsed:
                    stats["unparsed"] += 1
                    stats["unparsed_truncated" if trunc
                          else "unparsed_not_truncated"] += 1
                elif trunc:
                    stats["truncated_but_parsed"] += 1
                iid = int(r["instance_id"])
                valid_per_q[iid] = valid_per_q.get(iid, 0) + valid
                out_rows.append(dict(
                    dataset=ds, instance_id=iid, question_id=r["question_id"],
                    arm=arm, sample_idx=int(r["sample_idx"]),
                    parsed_answer=pa if parsed else "",
                    correct_answer=r["correct_answer"],
                    hint_target=r["hint_target"],
                    is_correct=parsed and pa == r["correct_answer"],
                    followed_hint=parsed and pa == r["hint_target"],
                    truncated=trunc, token_count=r.get("token_count", ""),
                    valid=valid))
            report.append(dict(
                dataset=ds, arm=arm, **stats,
                unparsed_pct=round(100 * stats["unparsed"]
                                   / max(stats["runs"], 1), 2),
                questions=len(valid_per_q),
                questions_lt_min_valid=sum(v < args.min_valid
                                           for v in valid_per_q.values())))

        if not out_rows:
            continue
        out_rows.sort(key=lambda r: (r["instance_id"], r["arm"] != "unhinted",
                                     r["sample_idx"]))
        paths["analysis_dir"].mkdir(parents=True, exist_ok=True)
        ans_path = paths["analysis_dir"] / "answers_per_run.csv"
        with ans_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ANSWER_FIELDS)
            w.writeheader()
            w.writerows(out_rows)
        print(f"Wrote {ans_path} ({len(out_rows)} runs)")

    if not report:
        print("ERROR: no inference outputs found — nothing to report.")
        sys.exit(1)

    # one roll-up row per arm across the datasets
    for arm in ARMS:
        sub = [r for r in report if r["arm"] == arm]
        if not sub:
            continue
        tot = {k: sum(r[k] for r in sub) for k in REPORT_FIELDS
               if k not in ("dataset", "arm", "unparsed_pct")}
        tot["unparsed_pct"] = round(100 * tot["unparsed"]
                                    / max(tot["runs"], 1), 2)
        report.append(dict(dataset="ALL", arm=arm, **tot))

    summary_dir = (root / config["paths"]["analysis_dir"] / model_name)
    if args.data != "test":
        summary_dir = summary_dir / args.data
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_path = summary_dir / f"parse_report_{args.hint_type}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
        w.writeheader()
        w.writerows(report)

    header = ("| dataset | arm | runs | valid | unparsed | unparsed % | "
              "unparsed: truncated | unparsed: not truncated | "
              "truncated but parsed | questions | "
              f"questions < {args.min_valid} valid |")
    lines = [
        f"# Answer-parse report — {args.model}, {args.hint_type} "
        f"(N = {n_samples} runs/question/arm)",
        "",
        "`unparsed` = runs whose final answer letter could not be parsed. "
        "Valid = parsed and not truncated; only valid runs enter the "
        f"analysis. The last column counts questions with fewer than "
        f"{args.min_valid} valid runs in that arm (for the unhinted arm these "
        "are dropped by step 59).",
        "",
        header,
        "|" + "---|" * len(REPORT_FIELDS),
    ]
    for r in report:
        lines.append("| " + " | ".join(str(r[k]) for k in REPORT_FIELDS) + " |")
    md_path = summary_dir / f"parse_report_{args.hint_type}.md"
    md_path.write_text("\n".join(lines) + "\n")

    print("\n" + "\n".join(lines[4:]))
    print(f"\nWrote {csv_path}\nWrote {md_path}")


if __name__ == "__main__":
    main()
