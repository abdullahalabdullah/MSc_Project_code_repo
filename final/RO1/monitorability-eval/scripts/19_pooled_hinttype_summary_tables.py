#!/usr/bin/env python3
"""
Step 19: Main-results summary tables at HINT-TYPE level — every headline
metric pooled over the 8 datasets, per model, with paired-bootstrap CIs,
rendered as markdown tables (one table per model, one column per hint type).

Transpose companion of 15_pooled_summary_tables.py (whose columns are datasets
and whose cells pool the 3 hint types): here each column is one hint type and
each cell pools the 8 datasets — the unweighted mean of the 8 per-dataset
estimates (never a mean over pooled instances; datasets weighted equally, the
same pooling rule with the roles of hint type and dataset swapped). Rendered
rows = step 15's four MONITORABILITY metrics only (g-mean^2, TE on V,
R_effect on V, % TE>0): the accuracy rows are computed and stored in the JSON
but omitted from the tables — dataset-pooled accuracy averages disjoint
question populations of very different difficulty, and the unhinted arm
cannot vary by hint type.

CI METHOD (matches 18_gmean2_by_hinttype_barchart.py):
  - Pooled rows: paired cluster bootstrap over instances. Each replicate
    draws, for every dataset, n_d instance ids with replacement from that
    dataset's shared pool, and reuses the SAME dataset-wise draws across all
    hint types and all metrics; per replicate: per-dataset mean over the drawn
    ids defined for that (dataset, hint), then the mean over datasets.
    10,000 resamples, seed 88. Datasets and hint types are treated as fixed.
  - Unhinted accuracy: the unhinted arm is a single condition shared verbatim
    by all hint types, so its dataset-pooled value is BY CONSTRUCTION the same
    in every column; its CI is a plain percentile bootstrap over instances
    (within-dataset resamples, mean over datasets).

Usage:
    python scripts/19_pooled_hinttype_summary_tables.py

    Options: --config, --data test|eval, --resamples, --seed,
             --base / --harmless / --antisyc   (as in step 15)

Input (per model / dataset / hint type, produced by 04_compute_gmean2.py):
    {analysis_dir}/{model}/{dataset}/{hint}/results_per_instance.csv

Output:
    {analysis_dir}/{model}/pooled_summary_by_hint.json      (machine-readable)
    {analysis_dir}/main_results_hint_type_level_tables.md   (presentation tables)
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from src.dataset import SPLITS, model_dir_name, resolve_paths
from src.hints import HINT_TYPES

# Reuse step 15's loader, metric definitions and row labels so the two files
# can never drift apart.
sys.path.insert(0, str(Path(__file__).resolve().parent))
importer = __import__("15_pooled_summary_tables")
load_hint_type = importer.load_hint_type
METRICS = importer.METRICS
POOLED_KEYS = importer.POOLED_KEYS
_nan_vector = importer._nan_vector

# Rendered rows: monitorability metrics only. The accuracy rows are computed
# and stored in pooled_summary_by_hint.json but NOT rendered here —
# dataset-pooled accuracy averages disjoint question populations of very
# different difficulty (blunt to interpret), and the unhinted arm cannot vary
# by hint type, so its row would repeat one number across all columns. See
# the dataset-level report for accuracies.
TABLE_METRICS = [m for m in METRICS if m[0] not in ("acc_h", "acc_u")]

DEFAULT_RESAMPLES = 10000
DEFAULT_SEED = 88

DEFAULT_BASE = "Qwen/Qwen3-4B"
DEFAULT_HARMLESS = "er-abd/harmless-dpo-qwen3-4B-step-735"
DEFAULT_ANTISYC = "er-abd/anti-syc-dpo-qwen3-4B-step-520"

HINT_LABELS = {
    "professor": "Professor hint",
    "grader_hacking": "Grader-hacking hint",
    "unethical": "Unethical hint",
}


def summarize_model(loaded: dict, datasets: list, n_resamples: int,
                    seed: int) -> dict:
    """Pool every metric over datasets, per hint type, with bootstrap CIs.

    `loaded` maps dataset -> hint_type -> load_hint_type() dict. Returns
    hint_type -> metric -> {estimate, ci95, per_dataset, ...}, plus "acc_u".
    """
    # Per dataset: shared union pool across hint types + NaN-padded vectors.
    pools, vecs, acc_u = {}, {}, {}
    for ds in datasets:
        by_hint = loaded[ds]
        union = sorted(set().union(*(set(e["pool"]) for e in by_hint.values())))
        pools[ds] = np.array(union)
        vecs[ds] = {k: {h: _nan_vector(by_hint[h]["maps"][k], pools[ds])
                        for h in by_hint} for k in POOLED_KEYS}
        # Merge the shared unhinted arm across hint dirs (checking agreement
        # where both define it), as in step 15.
        merged = np.full(len(union), np.nan)
        for h in by_hint:
            v = _nan_vector(by_hint[h]["maps"]["acc_u"], pools[ds])
            both = ~np.isnan(v) & ~np.isnan(merged)
            if not np.allclose(v[both], merged[both], atol=1e-9):
                raise ValueError(f"acc_u_i disagrees between hint dirs for "
                                 f"{ds}; the unhinted arm should be "
                                 f"hint-type-invariant")
            merged = np.where(np.isnan(merged), v, merged)
        acc_u[ds] = merged

    # Point estimates: per-dataset mean over defined instances, then the
    # unweighted mean over datasets.
    point = {h: {} for h in HINT_TYPES}
    per_ds_point = {h: {} for h in HINT_TYPES}
    for h in HINT_TYPES:
        for k in POOLED_KEYS:
            per_ds = {ds: float(np.nanmean(vecs[ds][k][h]))
                      for ds in datasets if h in vecs[ds][k]
                      and not np.all(np.isnan(vecs[ds][k][h]))}
            per_ds_point[h][k] = per_ds
            point[h][k] = float(np.mean(list(per_ds.values()))) if per_ds else None
    acc_u_per_ds = {ds: float(np.nanmean(acc_u[ds])) for ds in datasets}
    acc_u_point = float(np.mean(list(acc_u_per_ds.values())))

    # Paired bootstrap: one index matrix per dataset, reused across every hint
    # type and metric (and for acc_u).
    rng = np.random.default_rng(seed)
    reps = {h: {k: np.zeros((n_resamples, len(datasets)))
                for k in POOLED_KEYS} for h in HINT_TYPES}
    reps_acc_u = np.zeros((n_resamples, len(datasets)))
    with np.errstate(invalid="ignore"):
        for d_idx, ds in enumerate(datasets):
            n_d = len(pools[ds])
            idx = rng.integers(0, n_d, size=(n_resamples, n_d))
            for h in HINT_TYPES:
                for k in POOLED_KEYS:
                    v = vecs[ds][k].get(h)
                    reps[h][k][:, d_idx] = (np.nanmean(v[idx], axis=1)
                                            if v is not None else np.nan)
            reps_acc_u[:, d_idx] = np.nanmean(acc_u[ds][idx], axis=1)

    out = {}
    for h in HINT_TYPES:
        out[h] = {}
        for k in POOLED_KEYS:
            per_rep = np.nanmean(reps[h][k], axis=1)
            lo, hi = (float(v) for v in np.percentile(per_rep, [2.5, 97.5]))
            out[h][k] = {"estimate": point[h][k], "ci95": [lo, hi],
                         "per_dataset": per_ds_point[h][k],
                         "n_datasets": len(per_ds_point[h][k])}
    per_rep = np.nanmean(reps_acc_u, axis=1)
    lo, hi = (float(v) for v in np.percentile(per_rep, [2.5, 97.5]))
    out["acc_u"] = {"estimate": acc_u_point, "ci95": [lo, hi],
                    "per_dataset": acc_u_per_ds, "n_datasets": len(datasets)}
    return out


def delta_by_hint(base_loaded: dict, trained_loaded: dict,
                  base_res: dict, trained_res: dict, datasets: list,
                  n_resamples: int, seed: int) -> dict:
    """Paired-bootstrap 95% CI on (trained − base), per hint type and metric.

    Each replicate draws, for every dataset, one set of instance ids from the
    cross-model shared pool and reuses it for BOTH models and all hint types;
    per replicate: per-dataset mean over each model's defined subset, mean
    over datasets, then the difference. Point estimate = difference of the
    two reported pooled estimates.
    """
    keys = [k for k, _, _ in TABLE_METRICS]
    pools, vecs = {}, {}
    for ds in datasets:
        union = sorted(set().union(
            *(set(e["pool"]) for e in base_loaded[ds].values()),
            *(set(e["pool"]) for e in trained_loaded[ds].values())))
        pools[ds] = np.array(union)
        for tag, loaded in (("b", base_loaded[ds]), ("t", trained_loaded[ds])):
            for h in HINT_TYPES:
                for k in keys:
                    vecs[(ds, tag, h, k)] = _nan_vector(
                        loaded[h]["maps"][k], pools[ds])

    rng = np.random.default_rng(seed)
    reps = {(tag, h, k): np.zeros((n_resamples, len(datasets)))
            for tag in ("b", "t") for h in HINT_TYPES for k in keys}
    with np.errstate(invalid="ignore"):
        for d_idx, ds in enumerate(datasets):
            idx = rng.integers(0, len(pools[ds]),
                               size=(n_resamples, len(pools[ds])))
            for tag in ("b", "t"):
                for h in HINT_TYPES:
                    for k in keys:
                        reps[(tag, h, k)][:, d_idx] = np.nanmean(
                            vecs[(ds, tag, h, k)][idx], axis=1)

    out = {}
    for h in HINT_TYPES:
        out[h] = {}
        for k in keys:
            delta_reps = (np.nanmean(reps[("t", h, k)], axis=1)
                          - np.nanmean(reps[("b", h, k)], axis=1))
            lo, hi = (float(v) for v in
                      np.nanpercentile(delta_reps, [2.5, 97.5]))
            b = base_res[h][k]["estimate"]
            t = trained_res[h][k]["estimate"]
            out[h][k] = {"delta": (t - b) if b is not None and t is not None
                         else None, "ci95": [lo, hi]}
    return out


# --- markdown rendering ------------------------------------------------------

def _row_label(label: str) -> str:
    return label.replace("(mean of 3 hints)", "(mean of 8 datasets)")


def _fmt_cell(entry: dict, key: str, n_ds_expected: int) -> str:
    est, (lo, hi) = entry["estimate"], entry["ci95"]
    if est is None:
        return "N/A"
    if key == "pct_te_pos":
        cell = f"{100 * est:.0f}%"
    else:
        cell = f"{est:.3f} [{lo:.3f}, {hi:.3f}]"
    if entry.get("n_datasets", n_ds_expected) < n_ds_expected:
        cell += f" †({entry['n_datasets']} datasets)"
    return cell


def render_model_table(results: dict, n_ds: int) -> str:
    header = "| Metric | " + " | ".join(
        HINT_LABELS[h] for h in HINT_TYPES) + " |"
    lines = [header, "|---" * (len(HINT_TYPES) + 1) + "|"]
    for key, label, _ in TABLE_METRICS:
        cells = [_fmt_cell(results[h][key], key, n_ds) for h in HINT_TYPES]
        lines.append(f"| {_row_label(label)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def delta_cell_str(entry: dict, key: str) -> str:
    if entry is None or entry["delta"] is None:
        return "N/A"
    d, (lo, hi) = entry["delta"], entry["ci95"]
    if key == "pct_te_pos":
        dp = round(100 * d)
        return (f"{dp:+.0f}" if dp else "0") + \
            f" pp [{100 * lo:+.0f}, {100 * hi:+.0f}]"
    dp = round(d, 3)
    return (f"{dp:+.3f}" if dp else "0.000") + f" [{lo:+.3f}, {hi:+.3f}]"


def render_delta_table(deltas: dict) -> str:
    header = ("| Metric (Δ = trained − base) [paired-boot 95% CI] | "
              + " | ".join(HINT_LABELS[h] for h in HINT_TYPES) + " |")
    lines = [header, "|---" * (len(HINT_TYPES) + 1) + "|"]
    for key, label, _ in TABLE_METRICS:
        label = _row_label(label.split(" [")[0])
        cells = [delta_cell_str(deltas[h].get(key), key) for h in HINT_TYPES]
        lines.append(f"| Δ {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


FOOTNOTES = """\
**Delta tables**: each cell is the trained model's pooled point estimate minus
the base model's, with its own **paired-bootstrap 95% CI**: each replicate
draws, for every dataset, one set of instance ids and reuses it for BOTH
models and all hint types (10,000 resamples, seed 88), so the shared-question
covariance is propagated — these intervals are much tighter than comparing
the two models' separate CIs, which is the correct behavior for a paired
design. A delta CI excluding 0 is a paired test at the 5% level. The
`% with TE > 0` row is in percentage points (pp).

**Paired-boot 95% CI** (every interval in this report): paired cluster
bootstrap over instances — each replicate draws, for every dataset, n instance
ids with replacement from that dataset's shared pool and reuses the same
dataset-wise draws across the three hint types (10,000 resamples, seed 88).
The pooled point estimate is the unweighted mean of the 8 per-dataset
estimates, never a mean over pooled instances. Datasets and hint types are
treated as fixed, so the interval covers question-sampling uncertainty only.

**Accuracy rows are intentionally omitted** from this report: dataset-pooled
accuracy averages disjoint question populations of very different difficulty,
and the unhinted arm cannot vary by hint type. Accuracies are reported at
dataset level and per (dataset, hint type); both are still stored in
`pooled_summary_by_hint.json`.

*V* = level-1-passing instances (TE > 0 and p1 > 0). Each dataset: n = 200
instances, 8 samples per arm. † marks a cell pooled over fewer than 8 datasets.\
"""


def render_report(models: list, all_results: dict, deltas_all: dict,
                  n_ds: int, split: str) -> str:
    out = ["# Main results — monitorability per hint type, pooled over "
           "datasets", ""]
    if split != "test":
        out += [f"*Split: {split}*", ""]
    out += ["## Per-model results", ""]
    for label, model_id in models:
        out += [f"### {label} — `{model_id}`", "",
                render_model_table(all_results[model_id], n_ds), ""]
    for i, (label, model_id) in enumerate(models[1:], 1):
        title = ("Base vs harmlessness-trained DPO" if i == 1
                 else "Base vs anti-sycophancy-trained DPO")
        out += [f"## Comparison {i}: {title}", "",
                f"Δ = `{model_id}` − `{models[0][1]}`", "",
                render_delta_table(deltas_all[model_id]), ""]
    out += ["---", "", FOOTNOTES, ""]
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(
        description="Hint-type-level summary tables (pooled over datasets).")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--data", default="test", choices=SPLITS)
    parser.add_argument("--resamples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--harmless", default=DEFAULT_HARMLESS)
    parser.add_argument("--antisyc", default=DEFAULT_ANTISYC)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    with config_path.open() as f:
        config = yaml.safe_load(f)
    n_resamples = args.resamples or config.get("gmean2", {}).get(
        "bootstrap_resamples", DEFAULT_RESAMPLES)
    datasets = list(config.get("datasets", {}).keys())

    models = [("Base", args.base),
              ("Harmlessness DPO", args.harmless),
              ("Anti-sycophancy DPO", args.antisyc)]

    all_results, loaded_all, used_all, out_dirs = {}, {}, {}, {}
    for label, model_id in models:
        model_name = model_dir_name(model_id)
        print(f"\n=== {label}: {model_name} (split: {args.data}, "
              f"{n_resamples} resamples, seed {args.seed}) ===")
        loaded = {}
        for ds in datasets:
            by_hint = {}
            for hint in HINT_TYPES:
                paths = resolve_paths(root, config, ds, model_name, hint,
                                      split=args.data)
                entry = load_hint_type(paths["analysis_dir"])
                if entry is not None:
                    by_hint[hint] = entry
            if len(by_hint) < len(HINT_TYPES):
                missing = [h for h in HINT_TYPES if h not in by_hint]
                print(f"  WARNING: {ds} missing hint type(s): "
                      f"{', '.join(missing)} — dataset excluded")
                continue
            loaded[ds] = by_hint
        if not loaded:
            print(f"ERROR: no complete datasets for {model_name}.")
            sys.exit(1)
        used = [ds for ds in datasets if ds in loaded]
        loaded_all[model_id], used_all[model_id] = loaded, used
        results = summarize_model(loaded, used, n_resamples, args.seed)
        for h in HINT_TYPES:
            g = results[h]["gmean2"]
            print(f"  {h:<16} g-mean²={g['estimate']:.3f} "
                  f"CI [{g['ci95'][0]:.3f}, {g['ci95'][1]:.3f}]")
        all_results[model_id] = results

        out_dir = root / config["paths"]["analysis_dir"] / model_name
        if args.data != "test":
            out_dir = out_dir / args.data
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": model_id, "split": args.data,
            "n_resamples": n_resamples, "seed": args.seed,
            "datasets_pooled": used,
            "pooling": "unweighted mean of per-dataset estimates",
            "ci_method": {
                "pooled_rows": "paired cluster bootstrap over instances; "
                               "per-dataset resamples reused across hint "
                               "types; datasets and hint types fixed",
                "acc_u": "plain percentile bootstrap over instances "
                         "(single shared arm; identical across hint types)",
            },
            "hint_types": {h: results[h] for h in HINT_TYPES},
            "acc_u": results["acc_u"],
        }
        json_path = out_dir / "pooled_summary_by_hint.json"
        with json_path.open("w") as f:
            json.dump(payload, f, indent=2)
        print(f"  Wrote {json_path}")
        out_dirs[model_id] = out_dir

    # Paired-bootstrap CIs on (trained − base), per trained model.
    base_id = models[0][1]
    deltas_all = {}
    for label, model_id in models[1:]:
        shared = [ds for ds in used_all[base_id] if ds in used_all[model_id]]
        print(f"\n=== deltas: {label} − Base ({len(shared)} datasets) ===")
        deltas = delta_by_hint(loaded_all[base_id], loaded_all[model_id],
                               all_results[base_id], all_results[model_id],
                               shared, n_resamples, args.seed)
        for h in HINT_TYPES:
            g = deltas[h]["gmean2"]
            print(f"  {h:<16} Δg-mean²={g['delta']:+.3f} "
                  f"CI [{g['ci95'][0]:+.3f}, {g['ci95'][1]:+.3f}]")
        deltas_all[model_id] = deltas
        payload = {
            "trained_model": model_id, "base_model": base_id,
            "split": args.data, "n_resamples": n_resamples,
            "seed": args.seed, "datasets_pooled": shared,
            "method": "paired cluster bootstrap on the difference; "
                      "per-dataset instance resamples shared across both "
                      "models and all hint types; point estimate = "
                      "difference of the reported pooled estimates",
            "hint_types": deltas,
        }
        json_path = out_dirs[model_id] / "delta_vs_base_by_hint.json"
        with json_path.open("w") as f:
            json.dump(payload, f, indent=2)
        print(f"  Wrote {json_path}")

    report_dir = root / config["paths"]["analysis_dir"]
    if args.data != "test":
        report_dir = report_dir / args.data
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "main_results_hint_type_level_tables.md"
    report_path.write_text(render_report(models, all_results, deltas_all,
                                         len(datasets), args.data))
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
