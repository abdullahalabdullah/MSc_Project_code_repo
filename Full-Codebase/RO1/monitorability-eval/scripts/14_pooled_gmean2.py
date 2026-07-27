#!/usr/bin/env python3
"""
Step 14: Pool g-mean^2 across hint types for one model, with a paired
(cluster) bootstrap CI over instances.

WHY A NEW BOOTSTRAP. The per-hint-type CIs written by 04_compute_gmean2.py
(results_summary.json -> ci95.gmean2) describe ONE hint type each. The pooled
point estimate is a NEW statistic (the unweighted mean of the per-hint-type
g-mean^2 values) and needs its own interval. A CI cannot be obtained by
arithmetic on other CIs: averaging the three intervals' endpoints assumes the
hint types move in perfect lockstep and is ~1.5x too wide in practice (measured
correlations across replicates are ~0.0-0.3, so the estimation errors partially
cancel). The covariance information needed is only present in the per-instance
files, which is why this script reads results_per_instance.csv rather than the
summary JSONs.

METHOD (paired cluster bootstrap; questions are the resampling unit):
  All hint types for a dataset run on the IDENTICAL instance pool (only the
  prompt differs), so the three estimates are correlated. Each replicate:
    1. draw N instance_ids WITH replacement from the shared pool
       (N = size of the pool, i.e. n_total)
    2. for each hint type, average gmean2_i over the drawn ids that have a
       defined value for that hint type
    3. average those per-hint-type means -> one replicate of the pooled statistic
  The 2.5th/97.5th percentiles over `bootstrap_resamples` replicates are the CI.
  Reusing the SAME drawn ids across hint types within a replicate is what makes
  it paired and what propagates the shared-instance correlation.

  Note the per-hint-type defined sets differ (g-mean^2 is defined only for
  level-1-passing instances, and level-1 pass rates vary by hint type). Each
  hint type is therefore averaged over its own defined subset and |V| varies
  across replicates, which also propagates uncertainty in the level-1 filtering.
  Restricting to the intersection instead would drop instances and condition on
  a biased (most-hint-susceptible) subset.

WHAT THE CI COVERS. Question-sampling uncertainty only. Hint types are treated
as FIXED (they are the population -- the three from the GDM hint-taking paper --
not a sample), so systematic between-hint-type differences do not widen the
interval; the reported hint-type min-max range carries that heterogeneity
instead. Per-instance gmean2_i is treated as a fixed attribute, so Monte-Carlo
noise from the N generations per arm is not propagated (same convention as
bootstrap_cis in src/gmean2_metrics.py).

Usage:
    python scripts/14_pooled_gmean2.py --model er-abd/harmless-dpo-qwen3-4B-step-735
    python scripts/14_pooled_gmean2.py --model <id> --dataset gpqa_diamond

    Options:
      --config      path to config YAML (default: config/mvp_config.yaml)
      --dataset     restrict to one dataset key (default: every dataset in config)
      --data        split: test (default) | eval
      --resamples   bootstrap replicates (default: gmean2.bootstrap_resamples or 10000)
      --seed        bootstrap seed (default: 88, matching src/gmean2_metrics.py)

Input (per hint type, produced by 04_compute_gmean2.py):
    {analysis_dir}/{model}/{dataset}/{hint}/results_per_instance.csv
    {analysis_dir}/{model}/{dataset}/{hint}/results_summary.json   (per-hint CIs)

Output (under {analysis_dir}/{model}/):
    pooled_gmean2.json  — per dataset: pooled estimate, paired-bootstrap CI,
                          per-hint-type values + their own CIs, hint-type range
    pooled_gmean2.csv   — same, one row per dataset (main-table source)
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from src.dataset import SPLITS, model_dir_name, resolve_paths
from src.hints import HINT_TYPES

DEFAULT_RESAMPLES = 10000
DEFAULT_SEED = 88


def _fmt(v, decimals=4):
    return f"{v:.{decimals}f}" if v is not None else "N/A"


def load_hint_type(analysis_dir: Path):
    """Return (id -> gmean2_i for defined instances, all instance_ids, per-hint CI).

    Returns None when the hint type has not been computed for this dataset.
    """
    per_instance = analysis_dir / "results_per_instance.csv"
    if not per_instance.exists():
        return None

    defined, pool = {}, []
    with per_instance.open() as f:
        for row in csv.DictReader(f):
            iid = int(row["instance_id"])
            pool.append(iid)
            raw = row.get("gmean2_i", "")
            if raw not in ("", "None", "N/A"):
                defined[iid] = float(raw)

    ci = None
    summary = analysis_dir / "results_summary.json"
    if summary.exists():
        with summary.open() as f:
            ci = (json.load(f).get("ci95") or {}).get("gmean2")

    return {"defined": defined, "pool": pool, "ci": ci}


def pooled_gmean2(loaded: dict, n_resamples: int, seed: int) -> dict:
    """Pool across hint types and attach a paired-bootstrap CI.

    `loaded` maps hint_type -> the dict returned by load_hint_type().
    """
    hint_types = sorted(loaded)
    per_hint = {h: float(np.mean(list(loaded[h]["defined"].values())))
                for h in hint_types if loaded[h]["defined"]}
    if not per_hint:
        return None

    point = float(np.mean(list(per_hint.values())))

    # The prepared instance pool is shared by construction (same dataset_index;
    # only the prompt changes), but 04_compute_gmean2.py drops instances whose
    # hinted or unhinted arm had no valid run, and those losses differ slightly
    # by hint type. Resample from the UNION — the best observable proxy for the
    # prepared pool — so every replicate spans the same question universe. A
    # large divergence would mean something other than sporadic invalid runs, so
    # flag it rather than silently pooling.
    pools = {h: set(loaded[h]["pool"]) for h in hint_types}
    union = sorted(set().union(*pools.values()))
    pool_mismatch = {h: len(union) - len(p) for h, p in pools.items() if len(p) != len(union)}
    if pool_mismatch:
        worst = max(pool_mismatch.values()) / len(union)
        msg = ", ".join(f"{h} missing {n}" for h, n in pool_mismatch.items())
        if worst > 0.05:
            raise ValueError(
                f"instance pools diverge by >5% of the union ({len(union)}): {msg}; "
                f"the paired bootstrap assumes a shared question pool"
            )
        print(f"    note: pool sizes differ slightly (union {len(union)}; {msg}) — "
              f"resampling from the union")

    pool = np.array(union)
    maps = [loaded[h]["defined"] for h in per_hint]
    rng = np.random.default_rng(seed)

    replicates = []
    for _ in range(n_resamples):
        drawn = rng.choice(pool, size=len(pool), replace=True)
        means = []
        for m in maps:
            vals = [m[i] for i in drawn if i in m]
            if vals:
                means.append(np.mean(vals))
        if means:
            replicates.append(float(np.mean(means)))

    lo, hi = (float(v) for v in np.percentile(replicates, [2.5, 97.5]))
    values = list(per_hint.values())

    return {
        "gmean2_pooled": point,
        "ci95": [lo, hi],
        "ci95_width": hi - lo,
        "bootstrap_se": float(np.std(replicates)),
        "n_resamples": len(replicates),
        "seed": seed,
        "n_pool": len(pool),
        "hint_types": list(per_hint),
        "per_hint_gmean2": per_hint,
        "per_hint_ci95": {h: loaded[h]["ci"] for h in per_hint},
        "per_hint_n_defined": {h: len(loaded[h]["defined"]) for h in per_hint},
        "hint_range": [min(values), max(values)],
        "n_hint_types": len(per_hint),
    }


def run_dataset(root, config, dataset_key, model_name, split, n_resamples, seed):
    loaded = {}
    for hint_type in HINT_TYPES:
        paths = resolve_paths(root, config, dataset_key, model_name, hint_type, split=split)
        entry = load_hint_type(paths["analysis_dir"])
        if entry is not None:
            loaded[hint_type] = entry

    if not loaded:
        print(f"  {dataset_key:<14} SKIP — no results_per_instance.csv found")
        return None

    result = pooled_gmean2(loaded, n_resamples, seed)
    if result is None:
        print(f"  {dataset_key:<14} SKIP — no defined g-mean^2 instances")
        return None

    missing = [h for h in HINT_TYPES if h not in result["per_hint_gmean2"]]
    if missing:
        # Pooling over fewer hint types is not comparable across datasets.
        print(f"  WARNING: {dataset_key} pools only {result['n_hint_types']} hint "
              f"type(s); missing {', '.join(missing)}")
    result["missing_hint_types"] = missing

    lo, hi = result["ci95"]
    rlo, rhi = result["hint_range"]
    print(f"  {dataset_key:<14} {result['gmean2_pooled']:.3f}  "
          f"CI [{lo:.3f}, {hi:.3f}]  width {hi - lo:.3f}  "
          f"hint range {rlo:.3f}-{rhi:.3f}  ({result['n_hint_types']} hint types)")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Pool g-mean^2 across hint types with a paired-bootstrap CI.")
    parser.add_argument("--model", required=True,
                        help="Model/adapter HF id (namespaces the result dirs).")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--dataset", default=None,
                        help="Restrict to one dataset key (default: every dataset in config).")
    parser.add_argument("--data", default="test", choices=SPLITS,
                        help="Split to read (default: test).")
    parser.add_argument("--resamples", type=int, default=None,
                        help=f"Bootstrap replicates (default: config gmean2.bootstrap_resamples "
                             f"or {DEFAULT_RESAMPLES}).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Bootstrap seed (default: {DEFAULT_SEED}).")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    with config_path.open() as f:
        config = yaml.safe_load(f)

    n_resamples = args.resamples or config.get("gmean2", {}).get(
        "bootstrap_resamples", DEFAULT_RESAMPLES)

    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = list(config.get("datasets", {}).keys())
        if not datasets:
            print("ERROR: config has no 'datasets' section to sweep.")
            sys.exit(1)

    model_name = model_dir_name(args.model)
    print(f"\n=== pooled g-mean^2 — {model_name} (split: {args.data}, "
          f"{n_resamples} resamples, seed {args.seed}) ===")

    results = {}
    for dataset_key in datasets:
        out = run_dataset(root, config, dataset_key, model_name, args.data,
                          n_resamples, args.seed)
        if out is not None:
            results[dataset_key] = out

    if not results:
        print("\nNo datasets produced a pooled estimate — nothing written.")
        sys.exit(1)

    # Written model-first, alongside the per-dataset analysis dirs.
    out_dir = (root / config["paths"]["analysis_dir"] / model_name)
    if args.data != "test":
        out_dir = out_dir / args.data
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": args.model,
        "split": args.data,
        "n_resamples": n_resamples,
        "seed": args.seed,
        "method": "paired cluster bootstrap over instances; hint types fixed",
        "datasets": results,
    }
    json_path = out_dir / "pooled_gmean2.json"
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)

    csv_path = out_dir / "pooled_gmean2.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "gmean2_pooled", "ci95_lo", "ci95_hi", "ci95_width",
                    "hint_range_lo", "hint_range_hi", "n_hint_types",
                    "missing_hint_types", "n_pool"])
        for ds, r in results.items():
            w.writerow([ds, _fmt(r["gmean2_pooled"]), _fmt(r["ci95"][0]),
                        _fmt(r["ci95"][1]), _fmt(r["ci95_width"]),
                        _fmt(r["hint_range"][0]), _fmt(r["hint_range"][1]),
                        r["n_hint_types"], ";".join(r["missing_hint_types"]),
                        r["n_pool"]])

    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
