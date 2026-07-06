"""Merge the final pairs of multiple curation runs into ONE training-ready DPO dataset.

Use case: a pilot run consumed some prompts (e.g. SAMPLE=4500 -> ~1k pairs), then a full
run curated MORE prompts with the pilot's excluded (EXCLUDE_RUNS=... -> ~4k pairs). This
script concatenates their final_pairs.joblib files (deduplicating on prompt, first run
wins), writes the merged final_pairs.joblib into a new output dir, and re-runs the stage-06
export there — producing a fresh seeded train/valid split + dpo_dataset over the merged
pool. The export respects the same env knobs as a normal run (SEED, VALID_FRACTION,
STRATIFY_VALID, EXPORT_FIELD).

NOTE: the merged split is a NEW split over all pairs — it does not preserve the source
runs' train/valid boundaries. That is fine when the merged dataset trains a model from
scratch (the standard flow), but do not evaluate a model on the merged valid split if it
was already trained on one of the source runs' train splits.

Usage:
    python merge_runs.py outputs/run_A outputs/run_B [--out outputs/merged_<timestamp>]

Output: <out>/final_pairs.joblib, <out>/dpo_dataset/, <out>/diagnostics.json ("merge" +
"export" sections).
"""
import argparse
import importlib.util
import os
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", help="previous run output dirs (each with final_pairs.joblib)")
    parser.add_argument("--out", default=None,
                        help="output dir for the merged run (default: outputs/merged_<timestamp>)")
    args = parser.parse_args()

    out_dir = args.out or os.path.join(HERE, "outputs", "merged_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    # config reads RUN_OUTPUT_DIR at import time — set it BEFORE importing config so every
    # path (FINAL_PAIRS_PATH, DPO_DATASET_DIR, DIAGNOSTICS_PATH) points at the merged dir.
    os.environ["RUN_OUTPUT_DIR"] = out_dir

    import joblib

    import config
    from utils import update_diagnostics

    if os.path.exists(config.FINAL_PAIRS_PATH):
        raise SystemExit(f"Refusing to overwrite existing {config.FINAL_PAIRS_PATH} — pick a fresh --out dir.")

    merged, seen_prompts = [], set()
    per_run, n_dupes = {}, 0
    for run_dir in args.run_dirs:
        fp = os.path.join(run_dir, os.path.basename(config.FINAL_PAIRS_PATH))
        if not os.path.exists(fp):
            raise FileNotFoundError(f"No {os.path.basename(config.FINAL_PAIRS_PATH)} in {run_dir!r}")
        pairs = joblib.load(fp)
        kept = 0
        for p in pairs:
            if p["prompt"] in seen_prompts:
                n_dupes += 1
                continue
            seen_prompts.add(p["prompt"])
            merged.append(p)
            kept += 1
        per_run[run_dir] = {"pairs": len(pairs), "kept": kept}
        print(f"  {run_dir}: {len(pairs)} pairs, {kept} kept")

    if not merged:
        raise SystemExit("No pairs to merge.")
    print(f"Merged: {len(merged)} pairs ({n_dupes} duplicate prompts dropped, first run wins)")

    config.ensure_output_dir()
    joblib.dump(merged, config.FINAL_PAIRS_PATH, compress=True)
    update_diagnostics(config.DIAGNOSTICS_PATH, "merge", {
        "source_runs": per_run,
        "duplicate_prompts_dropped": n_dupes,
        "merged_pairs": len(merged),
    })

    # Re-run the stage-06 export on the merged final_pairs (fresh seeded split + save).
    spec = importlib.util.spec_from_file_location(
        "export_dpo", os.path.join(HERE, "06_export_dpo_dataset.py"))
    export_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export_mod)
    export_mod.main()

    print(f"\nMerged run dir: {out_dir}")
    print(f"Train with:  bash train.sh {os.path.abspath(config.DPO_DATASET_DIR)}")


if __name__ == "__main__":
    main()
