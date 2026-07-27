#!/usr/bin/env python3
"""
Step 1: Import the frozen eval files (from the RO2 curation run) and build the
frozen prompt indexes.

The two reserved files are PRODUCED by RO2/anti-syc-DPO-train-dataset-curation
(Stage B; complete after that pipeline's stage 05). Import them ONCE with
--source-dir; they are then committed under anti-sycophancy-eval/data/reserved/
(small files) so this eval stays self-contained. Re-runs without --source-dir just
verify checksums and (re)build the indexes idempotently.

The prompt indexes freeze everything both models will see:
  data/eval_id_index.jsonl   per question: bare prompt + one wrong-opinion and one
                             right-opinion prompt (T1-T8 assigned with seed 1234 over
                             sorted (question_id, direction) — the curation Stage E
                             procedure, so the eval measures the training distribution)
  data/eval_ood_index.jsonl  per question: the turn-1 prompt (question + the shared
                             answer-format instruction); the challenge text lives in
                             the config block.

Usage:
    python scripts/01_prepare_eval_data.py --source-dir <RO2 run>/reserved   # first time
    python scripts/01_prepare_eval_data.py                                   # verify/build
"""

import argparse
import datetime
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from src.dataset import load_jsonl, resolve_paths, sha256_file, write_jsonl
from src.prompts import TEMPLATE_IDS, bare_prompt, render_opinion_prompt

RESERVED_FILES = ("eval_id_triviaqa.jsonl", "eval_ood_sycophancyeval.jsonl")


def import_reserved(source_dir: Path, paths: dict, force: bool):
    paths["reserved_dir"].mkdir(parents=True, exist_ok=True)
    manifest = {"imported_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "source_dir": str(source_dir), "sha256": {}}
    for fname in RESERVED_FILES:
        src = source_dir / fname
        dst = paths["reserved_dir"] / fname
        if not src.exists():
            sys.exit(f"ERROR: {src} not found — run the RO2 curation pipeline through "
                     f"stage 05 first (05_wrong_answers.py finalizes the eval slice).")
        if dst.exists() and not force:
            if sha256_file(src) != sha256_file(dst):
                sys.exit(f"ERROR: {dst} already exists with DIFFERENT content than "
                         f"{src}. The reserved files are frozen — pass --force only "
                         f"if you are certain the new source is the canonical one.")
            print(f"  {fname}: already imported (checksums match)")
        else:
            shutil.copy2(src, dst)
            print(f"  {fname}: imported from {src}")
        manifest["sha256"][fname] = sha256_file(dst)
    with open(paths["reserved_manifest"], "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  manifest -> {paths['reserved_manifest']}")


def verify_reserved(paths: dict):
    missing = [f for f in RESERVED_FILES if not (paths["reserved_dir"] / f).exists()]
    if missing:
        sys.exit(f"ERROR: reserved eval files missing: {missing}. Import them once with:\n"
                 f"  python scripts/01_prepare_eval_data.py --source-dir "
                 f"<RO2 anti-syc curation run>/reserved")
    if paths["reserved_manifest"].exists():
        with open(paths["reserved_manifest"]) as f:
            manifest = json.load(f)
        for fname, expected in manifest.get("sha256", {}).items():
            actual = sha256_file(paths["reserved_dir"] / fname)
            if actual != expected:
                sys.exit(f"ERROR: {fname} checksum drifted from the imported manifest "
                         f"({actual[:12]} != {expected[:12]}) — the frozen file was "
                         f"modified. Restore it from git or re-import.")
        print("  reserved files verified against manifest.")


def build_id_index(paths: dict, template_seed: int, force: bool):
    out = paths["eval_id_index"]
    if out.exists() and not force:
        print(f"  {out.name}: exists ({sum(1 for _ in open(out))} lines) — skipping "
              f"(--force to rebuild)")
        return
    records = load_jsonl(paths["eval_id_reserved"])
    # Curation Stage E procedure: one template per (question, direction), assigned by
    # a single seeded stream over the SORTED keys ("right_opinion" < "wrong_opinion").
    keys = sorted((r["question_id"], d) for r in records
                  for d in ("right_opinion", "wrong_opinion"))
    rng = random.Random(template_seed)
    assignment = {k: rng.choice(TEMPLATE_IDS) for k in keys}

    lines = []
    for r in sorted(records, key=lambda r: r["question_id"]):
        qid = r["question_id"]
        right_t = assignment[(qid, "right_opinion")]
        wrong_t = assignment[(qid, "wrong_opinion")]
        lines.append({
            "question_id": qid,
            "question": r["question"],
            "gold": r["gold"],
            "gold_aliases": r["gold_aliases"],
            "w": r["w"],
            "bare_prompt": bare_prompt(r["question"]),
            "right_template_id": right_t,
            "right_prompt": render_opinion_prompt(right_t, r["question"], r["gold"]),
            "wrong_template_id": wrong_t,
            "wrong_prompt": render_opinion_prompt(wrong_t, r["question"], r["w"]),
        })
    write_jsonl(out, lines)
    print(f"  {out.name}: built ({len(lines)} questions, sha256 {sha256_file(out)[:12]})")


def build_ood_index(paths: dict, force: bool):
    out = paths["eval_ood_index"]
    if out.exists() and not force:
        print(f"  {out.name}: exists ({sum(1 for _ in open(out))} lines) — skipping "
              f"(--force to rebuild)")
        return
    records = load_jsonl(paths["eval_ood_reserved"])
    lines = [{
        "question_id": r["question_id"],
        "question": r["question"],
        "gold": r["gold"],
        "gold_aliases": r["gold_aliases"],
        "base_dataset": r.get("base_dataset", "unknown"),
        "turn1_prompt": bare_prompt(r["question"]),
    } for r in records]
    write_jsonl(out, lines)
    print(f"  {out.name}: built ({len(lines)} questions, sha256 {sha256_file(out)[:12]})")


def main():
    parser = argparse.ArgumentParser(description="Import frozen eval files + build prompt indexes.")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--source-dir", default=None,
                        help="reserved/ dir of an RO2 anti-syc curation run (first import).")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite reserved files / rebuild indexes.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    with open(root / args.config) as f:
        config = yaml.safe_load(f)
    paths = resolve_paths(root, config)

    if args.source_dir:
        print(f"Importing reserved files from {args.source_dir}:")
        import_reserved(Path(args.source_dir), paths, args.force)
    verify_reserved(paths)

    template_seed = config["anti_sycophancy"]["template_seed"]
    print("Building frozen prompt indexes:")
    build_id_index(paths, template_seed, args.force)
    build_ood_index(paths, args.force)


if __name__ == "__main__":
    main()
