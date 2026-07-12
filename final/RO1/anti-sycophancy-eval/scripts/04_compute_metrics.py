#!/usr/bin/env python3
"""
Step 4: Compute M1 / M2 / M3 for one model (CPU; machine-readable outputs only —
all .md rendering is isolated in 06_generate_reports.py).

Usage:
    python scripts/04_compute_metrics.py --model Qwen/Qwen3-4B

Output (namespaced by model):
    analysis/{model}/metrics.json        headline metrics + config snapshot + checksums
    analysis/{model}/per_question.jsonl  per-question outcomes (consumed by 05_compare_models.py)
    analysis/{model}/per_question.csv    same, flat (human/pandas-friendly)
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from src.dataset import load_jsonl, model_dir_name, resolve_paths, sha256_file
from src.metrics import aggregate, per_question_id_outcomes, per_question_ood_outcomes


def main():
    parser = argparse.ArgumentParser(description="Compute anti-sycophancy metrics for one model.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default="config/mvp_config.yaml")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    with open(root / args.config) as f:
        config = yaml.safe_load(f)
    a_cfg = config["anti_sycophancy"]

    model_name = model_dir_name(args.model)
    paths = resolve_paths(root, config, model_name)
    grades_dir = paths["grades_dir"]
    analysis_dir = paths["analysis_dir"]
    analysis_dir.mkdir(parents=True, exist_ok=True)

    id_grades_path = grades_dir / "id_grades.jsonl"
    ood_grades_path = grades_dir / "ood_grades.jsonl"
    if not id_grades_path.exists() or not ood_grades_path.exists():
        sys.exit(f"ERROR: grades not found under {grades_dir} — run 03_run_grader.py first.")

    id_out = per_question_id_outcomes(load_jsonl(id_grades_path))
    ood_out = per_question_ood_outcomes(load_jsonl(ood_grades_path))
    agg = aggregate(id_out, ood_out,
                    min_conditioned=a_cfg["metrics"].get("min_conditioned_questions", 500))

    metrics = {
        "model_name": model_name,
        "hf_id": args.model,
        "grader_model": a_cfg["grader"]["model"],
        "config_snapshot": a_cfg["inference"],
        "index_sha256": {
            "eval_id_index": sha256_file(paths["eval_id_index"]),
            "eval_ood_index": sha256_file(paths["eval_ood_index"]),
        },
        **agg,
    }
    with open(analysis_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Per-question outcomes: one JSONL for 05_compare_models.py, one flat CSV.
    per_q = ([{"set": "id", "question_id": qid, **o} for qid, o in sorted(id_out.items())]
             + [{"set": "ood", "question_id": qid, **o} for qid, o in sorted(ood_out.items())])
    with open(analysis_dir / "per_question.jsonl", "w", encoding="utf-8") as f:
        for r in per_q:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    fields = sorted({k for r in per_q for k in r})
    with open(analysis_dir / "per_question.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_q)

    print(f"Metrics for {model_name}:")
    for k in ("M1_syc_rate", "M2_overcorrection_rate", "M3_regressive_flip_rate",
              "M3_progressive_flip_rate", "id_bare_accuracy"):
        v = metrics[k]
        print(f"  {k}: {v if v is None else f'{v:.3f}'}")
    print(f"  conditioned questions: {metrics['n_conditioned_questions']} "
          f"(floor {metrics['min_conditioned_required']})")
    print(f"-> {analysis_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
