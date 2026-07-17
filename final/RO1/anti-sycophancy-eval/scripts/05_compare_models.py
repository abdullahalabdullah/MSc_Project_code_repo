#!/usr/bin/env python3
"""
Step 5: Paired base-vs-trained comparison (CPU) — spec §4.3 rule 1.

Loads both models' per-question outcomes (04_compute_metrics.py output), asserts they
evaluated the exact same frozen indexes (checksums), and computes per-question paired
deltas with bootstrap CIs + exact McNemar for each metric. New to this eval — the
older evals compare models by eye across per-model dirs.

Usage:
    python scripts/05_compare_models.py --base Qwen/Qwen3-4B \
        --trained er-abd/anti-syc-dpo-qwen3-4B-step-N \
        [--capability-json path/to/capability.json]

Output:
    analysis/compare/{base}__vs__{trained}/comparison.json
    analysis/compare/{base}__vs__{trained}/per_question_deltas.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from src.compare import compare_metric, paired_questions
from src.dataset import load_jsonl, model_dir_name, resolve_paths

METRIC_FIELDS_ID = [
    ("wrong_w_rate", True),           # M1 — conditioned on bare-correct in BOTH models
    ("right_incorrect_rate", True),   # M2 — same conditioning
]
METRIC_FIELDS_OOD = [
    ("regressive_rate", False),       # M3 regressive
    ("progressive_rate", False),      # M3 progressive
]


def load_outcomes(analysis_dir: Path, model_name: str):
    pq_path = analysis_dir / "per_question.jsonl"
    m_path = analysis_dir / "metrics.json"
    if not pq_path.exists() or not m_path.exists():
        sys.exit(f"ERROR: {model_name} has no metrics under {analysis_dir} — run "
                 f"03_run_grader.py + 04_compute_metrics.py for it first.")
    with open(m_path) as f:
        metrics = json.load(f)
    id_out, ood_out = {}, {}
    for r in load_jsonl(pq_path):
        (id_out if r["set"] == "id" else ood_out)[r["question_id"]] = r
    return metrics, id_out, ood_out


def main():
    parser = argparse.ArgumentParser(description="Paired base-vs-trained comparison.")
    parser.add_argument("--base", required=True, help="Base model HF id.")
    parser.add_argument("--trained", required=True, help="Trained model HF id.")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--capability-json", default=None,
                        help="Optional capability-retention numbers to co-report "
                             "(spec §4.3 rule 2).")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    with open(root / args.config) as f:
        config = yaml.safe_load(f)
    a_cfg = config["anti_sycophancy"]

    base_name = model_dir_name(args.base)
    trained_name = model_dir_name(args.trained)
    base_paths = resolve_paths(root, config, base_name)
    trained_paths = resolve_paths(root, config, trained_name)

    base_metrics, base_id, base_ood = load_outcomes(base_paths["analysis_dir"], base_name)
    trained_metrics, trained_id, trained_ood = load_outcomes(
        trained_paths["analysis_dir"], trained_name)

    # Paired requirement: identical frozen indexes (hence identical prompts+seeds).
    if base_metrics["index_sha256"] != trained_metrics["index_sha256"]:
        sys.exit("ERROR: the two runs used different frozen indexes "
                 f"({base_metrics['index_sha256']} vs {trained_metrics['index_sha256']}) "
                 "— the comparison would not be paired.")

    comparison = {
        "base_model": base_name,
        "trained_model": trained_name,
        "index_sha256": base_metrics["index_sha256"],
        "config_snapshot": base_metrics["config_snapshot"],
        "per_model_metrics": {base_name: base_metrics, trained_name: trained_metrics},
        "metrics": {},
    }
    # M1/M2 always; M3 (OOD) only when enabled AND both models actually have OOD data.
    metric_fields = list(METRIC_FIELDS_ID)
    run_ood = a_cfg.get("run_ood", False)
    if run_ood and base_ood and trained_ood:
        metric_fields += METRIC_FIELDS_OOD
    for field, require_cond in metric_fields:
        b_out = base_id if field in ("wrong_w_rate", "right_incorrect_rate") else base_ood
        t_out = (trained_id if field in ("wrong_w_rate", "right_incorrect_rate")
                 else trained_ood)
        comparison["metrics"][field] = compare_metric(
            b_out, t_out, field, a_cfg["metrics"], require_conditioned=require_cond)

    if args.capability_json:
        with open(args.capability_json) as f:
            comparison["capability"] = json.load(f)

    audit_path = root / a_cfg["paths"]["data_dir"] / "grader_audit_results.json"
    if audit_path.exists():
        with open(audit_path) as f:
            comparison["grader_audit"] = json.load(f)

    out_dir = base_paths["compare_dir"] / f"{base_name}__vs__{trained_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    # Flat per-question delta table for inspection.
    with open(out_dir / "per_question_deltas.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "question_id", "base_rate", "trained_rate", "delta"])
        for field, require_cond in metric_fields:
            b_out = base_id if field in ("wrong_w_rate", "right_incorrect_rate") else base_ood
            t_out = (trained_id if field in ("wrong_w_rate", "right_incorrect_rate")
                     else trained_ood)
            for qid, b, t in paired_questions(b_out, t_out, field, require_cond):
                writer.writerow([field, qid, b, t, t - b])

    print(f"Comparison ({trained_name} vs {base_name}):")
    for field, m in comparison["metrics"].items():
        boot = m["bootstrap"]
        if boot["mean_delta"] is None:
            print(f"  {field}: no paired questions")
            continue
        print(f"  {field}: {m['base_mean']:.3f} -> {m['trained_mean']:.3f}  "
              f"delta {boot['mean_delta']:+.3f} "
              f"[{boot['ci_low']:+.3f}, {boot['ci_high']:+.3f}]  "
              f"McNemar p={m['mcnemar'].get('p_value'):.4f}  (n={m['n_paired_questions']})")
    print(f"-> {out_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()
