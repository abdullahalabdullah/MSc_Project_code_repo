"""Grader audit tooling (spec §6, blocking pilot gate 1).

Two subcommands, run against a completed pilot run dir:

  --make   Sample 100 classified grades stratified across labels (from filter0 +
           filter1 grades) into audit/grader_audit_sample.csv. The owner fills the
           `human_label` column by hand (same label vocabulary as the grader).
  --score  Compute agreement % + a confusion table over the hand-labeled rows and
           write audit/grader_audit_results.json (read by stage 11 / pilot gate 1;
           required agreement >= 95%).

Usage:
    python audit_grader_sample.py --make  outputs/pilot
    # ... hand-label the CSV ...
    python audit_grader_sample.py --score outputs/pilot
"""
import argparse
import csv
import json
import os
import random
import sys

AUDIT_SAMPLE_SIZE = 100
AUDIT_SEED = 7  # sampling-only; any fixed value works — recorded in the results file

CSV_FIELDS = ["source_stage", "question_id", "instance_id", "sample_idx", "prompt_type",
              "question", "gold", "gold_aliases", "asserted_answer", "extracted",
              "grader_label", "human_label", "notes"]


def _read_jsonl(path):
    records = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def _load_graded(run_dir):
    """All classified (non-UNPARSEABLE) grade lines, with grading context re-joined."""
    pool = {r["question_id"]: r for r in _read_jsonl(os.path.join(run_dir, "raw", "training_pool.jsonl"))}
    instances = {i["instance_id"]: i for i in _read_jsonl(os.path.join(run_dir, "filter1", "instances.jsonl"))}
    rows = []
    for r in _read_jsonl(os.path.join(run_dir, "filter0", "grades.jsonl")):
        if r["label"] == "UNPARSEABLE":
            continue
        q = pool.get(r["question_id"], {})
        rows.append({
            "source_stage": "filter0", "question_id": r["question_id"],
            "instance_id": "", "sample_idx": r["sample_idx"], "prompt_type": "bare",
            "question": q.get("question", ""), "gold": q.get("gold", ""),
            "gold_aliases": "; ".join(q.get("gold_aliases", [])),
            "asserted_answer": "", "extracted": r["extracted"],
            "grader_label": r["label"], "human_label": "", "notes": "",
        })
    for r in _read_jsonl(os.path.join(run_dir, "filter1", "grades.jsonl")):
        if r["label"] == "UNPARSEABLE":
            continue
        inst = instances.get(r["instance_id"], {})
        rows.append({
            "source_stage": "filter1", "question_id": r["question_id"],
            "instance_id": r["instance_id"], "sample_idx": r["sample_idx"],
            "prompt_type": r["direction"],
            "question": inst.get("question", ""), "gold": inst.get("gold", ""),
            "gold_aliases": "; ".join(inst.get("gold_aliases", [])),
            "asserted_answer": inst.get("asserted_answer", ""),
            "extracted": r["extracted"],
            "grader_label": r["label"], "human_label": "", "notes": "",
        })
    return rows


def make_sample(run_dir):
    rows = _load_graded(run_dir)
    if not rows:
        sys.exit("ERROR: no classified grades found — run the pipeline through stage 08 first.")
    by_label = {}
    for r in rows:
        by_label.setdefault(r["grader_label"], []).append(r)
    rng = random.Random(AUDIT_SEED)
    labels = sorted(by_label)
    # Stratified: equal share per label, remainder (and shortfalls) filled by frequency.
    per_label = AUDIT_SAMPLE_SIZE // len(labels)
    sample = []
    picked = set()
    for lab in labels:
        pool_l = by_label[lab]
        rng.shuffle(pool_l)
        sample.extend(pool_l[:per_label])
        picked.update(id(r) for r in pool_l[:per_label])
    remaining = [r for lab in labels for r in by_label[lab] if id(r) not in picked]
    rng.shuffle(remaining)
    sample.extend(remaining[:AUDIT_SAMPLE_SIZE - len(sample)])
    rng.shuffle(sample)

    out_path = os.path.join(run_dir, "audit", "grader_audit_sample.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(sample)
    dist = {lab: sum(1 for r in sample if r["grader_label"] == lab) for lab in labels}
    print(f"Wrote {len(sample)} grades -> {out_path}")
    print(f"  label distribution: {dist}")
    print("Hand-label the `human_label` column (same vocabulary as grader_label), "
          "then run:  python audit_grader_sample.py --score " + run_dir)


def score_sample(run_dir):
    csv_path = os.path.join(run_dir, "audit", "grader_audit_sample.csv")
    if not os.path.exists(csv_path):
        sys.exit(f"ERROR: {csv_path} not found — run --make first.")
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    labeled = [r for r in rows if r["human_label"].strip()]
    if not labeled:
        sys.exit("ERROR: no rows have human_label filled in yet.")
    agree = sum(1 for r in labeled
                if r["human_label"].strip().upper() == r["grader_label"].strip().upper())
    confusion = {}
    for r in labeled:
        key = f"{r['grader_label'].strip().upper()} -> {r['human_label'].strip().upper()}"
        confusion[key] = confusion.get(key, 0) + 1
    agreement = agree / len(labeled)
    results = {
        "n_labeled": len(labeled),
        "n_agree": agree,
        "agreement": round(agreement, 4),
        "required": 0.95,
        "pass": agreement >= 0.95,
        "confusion": dict(sorted(confusion.items())),
        "audit_seed": AUDIT_SEED,
    }
    out_path = os.path.join(run_dir, "audit", "grader_audit_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Agreement: {agreement:.1%} on {len(labeled)} grades "
          f"({'PASS' if results['pass'] else 'FAIL — refine the grader prompt and re-validate'})")
    print(f"Results -> {out_path}  (re-run 11_qa_report.py to fold into the QA report/gates)")


def main():
    ap = argparse.ArgumentParser(description="Grader audit (spec §6 / pilot gate 1).")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--make", metavar="RUN_DIR", help="Write the 100-grade audit CSV.")
    group.add_argument("--score", metavar="RUN_DIR", help="Score the hand-labeled CSV.")
    args = ap.parse_args()
    if args.make:
        make_sample(args.make)
    else:
        score_sample(args.score)


if __name__ == "__main__":
    main()
