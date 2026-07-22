#!/usr/bin/env python3
"""
Step 15: Main-results summary tables — every headline metric pooled over hint
types, per model, with paired-bootstrap CIs, rendered as presentation-ready
markdown tables (one table per model, one column per dataset, datasets grouped
by domain).

POOLING RULE (same as 14_pooled_gmean2.py, extended to all metrics): the pooled
point estimate is the unweighted mean of the three per-hint-type estimates
(professor, grader_hacking, unethical) — NOT a mean over the pooled instances.
Each hint type is its own complete n=200 run; instance-pooling would let
whichever hint type produced more level-1-passing instances dominate the cell,
and hint types differ substantially in how often they induce hint-following.

CI METHOD:
  - Pooled rows (g-mean^2, TE on V, R_effect on V, %TE>0, hinted accuracy):
    paired cluster bootstrap over instances. Each replicate draws N instance
    ids with replacement from the shared pool ONCE and reuses that same draw
    across all hint types (and all metrics), which propagates the correlation
    induced by the shared question pool. Hint types are treated as fixed (they
    are the population, not a sample). Same seed/stream as 14_pooled_gmean2.py,
    so the g-mean^2 row reproduces pooled_gmean2.json exactly.
  - Unhinted accuracy: the unhinted arm is a SINGLE condition shared verbatim
    by all hint types (03_run_monitor.py reuses the verdicts; acc_u_i is
    byte-identical across hint dirs — asserted here). No pooling applies; its
    CI is a plain percentile bootstrap over instances.

Usage:
    python scripts/15_pooled_summary_tables.py
    python scripts/15_pooled_summary_tables.py --resamples 10000 --seed 88

    Options:
      --config      path to config YAML (default: config/mvp_config.yaml)
      --data        split: test (default) | eval
      --resamples   bootstrap replicates (default: gmean2.bootstrap_resamples or 10000)
      --seed        bootstrap seed (default: 88, matching 14_pooled_gmean2.py)
      --base / --harmless / --antisyc   override the three model ids

Input (per model / dataset / hint type, produced by 04_compute_gmean2.py):
    {analysis_dir}/{model}/{dataset}/{hint}/results_per_instance.csv

Output:
    {analysis_dir}/{model}/pooled_summary.json            (machine-readable, per model)
    {analysis_dir}/main_results_dataset_level_tables.md   (presentation tables)
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

DEFAULT_BASE = "Qwen/Qwen3-4B"
DEFAULT_HARMLESS = "er-abd/harmless-dpo-qwen3-4B-step-735"
DEFAULT_ANTISYC = "er-abd/anti-syc-dpo-qwen3-4B-step-520"

# Presentation grouping/order of the dataset columns. Purely visual: every
# dataset keeps its own column; the domain is a header prefix.
DOMAINS = [
    ("Academic QA", ["mmlu_pro", "gpqa_diamond"]),
    ("Applied Reasoning", ["lsat_ar", "lsat_lr"]),
    ("Expert Medical", ["medxpertqa"]),
    ("Legal", ["lexam"]),
    ("Coding", ["codemmlu"]),
    ("Safety & Security", ["wmdp"]),
]

# metric key -> (row label, uses paired pooled bootstrap)
METRICS = [
    ("gmean2", "g-mean² (mean of 3 hints) [paired-boot 95% CI]", True),
    ("te_on_v", "Mean TE on V (mean of 3 hints) [paired-boot 95% CI]", True),
    ("r_effect_on_v", "Mean R_effect on V (mean of 3 hints) [paired-boot 95% CI]", True),
    ("pct_te_pos", "% with TE > 0 (mean of 3 hints)", True),
    ("acc_h", "Hinted accuracy (mean of 3 hints) [paired-boot 95% CI]", True),
    ("acc_u", "Unhinted accuracy (single arm) [95% CI]", False),
]

POOLED_KEYS = [k for k, _, pooled in METRICS if pooled]


def _f(raw):
    return None if raw in ("", "None", "N/A") else float(raw)


def load_hint_type(analysis_dir: Path):
    """Read one results_per_instance.csv into id-keyed metric maps.

    Returns None when this (dataset, hint) cell has not been computed.
    Undefined values are simply absent from the map (NaN downstream).
    """
    per_instance = analysis_dir / "results_per_instance.csv"
    if not per_instance.exists():
        return None

    maps = {k: {} for k, _, _ in METRICS}
    pool = []
    with per_instance.open() as f:
        for row in csv.DictReader(f):
            iid = int(row["instance_id"])
            pool.append(iid)
            level1 = row["level1_pass"] == "True"
            te = _f(row["te_i"])
            if (g := _f(row["gmean2_i"])) is not None:
                maps["gmean2"][iid] = g
            if level1:
                if te is not None:
                    maps["te_on_v"][iid] = te
                if (r := _f(row["r_effect_i"])) is not None:
                    maps["r_effect_on_v"][iid] = r
            if te is not None:
                maps["pct_te_pos"][iid] = 1.0 if te > 0 else 0.0
            if (ah := _f(row["acc_h_i"])) is not None:
                maps["acc_h"][iid] = ah
            if (au := _f(row["acc_u_i"])) is not None:
                maps["acc_u"][iid] = au
    return {"maps": maps, "pool": pool}


def _nan_vector(id_map: dict, pool: np.ndarray) -> np.ndarray:
    v = np.full(len(pool), np.nan)
    for j, iid in enumerate(pool):
        if int(iid) in id_map:
            v[j] = id_map[int(iid)]
    return v


def merged_acc_u(by_hint: dict, pool: np.ndarray) -> np.ndarray:
    """Merge the shared unhinted arm across hint dirs (asserting agreement
    wherever two dirs both define an instance)."""
    acc = np.full(len(pool), np.nan)
    for h, entry in by_hint.items():
        v = _nan_vector(entry["maps"]["acc_u"], pool)
        both = ~np.isnan(v) & ~np.isnan(acc)
        if not np.allclose(v[both], acc[both], atol=1e-9):
            raise ValueError(f"acc_u_i disagrees between hint dirs at {h}; "
                             f"the unhinted arm should be hint-type-invariant")
        acc = np.where(np.isnan(acc), v, acc)
    return acc


def summarize_dataset(loaded: dict, n_resamples: int, seed: int) -> dict:
    """Pool every metric over hint types for one dataset, with bootstrap CIs.

    `loaded` maps hint_type -> the dict returned by load_hint_type().
    One shared instance resample per replicate is reused across all hint types
    and all metrics (paired cluster bootstrap; questions are the resampling
    unit). Draw sequence matches 14_pooled_gmean2.py exactly.
    """
    hint_types = [h for h in HINT_TYPES if h in loaded]

    # Same union-pool logic and divergence guard as 14_pooled_gmean2.py.
    pools = {h: set(loaded[h]["pool"]) for h in hint_types}
    union = sorted(set().union(*pools.values()))
    pool_mismatch = {h: len(union) - len(p) for h, p in pools.items() if len(p) != len(union)}
    if pool_mismatch:
        worst = max(pool_mismatch.values()) / len(union)
        msg = ", ".join(f"{h} missing {n}" for h, n in pool_mismatch.items())
        if worst > 0.05:
            raise ValueError(
                f"instance pools diverge by >5% of the union ({len(union)}): {msg}; "
                f"the paired bootstrap assumes a shared question pool")
        print(f"    note: pool sizes differ slightly (union {len(union)}; {msg}) — "
              f"resampling from the union")
    pool = np.array(union)

    # The unhinted arm is one shared condition: assert acc_u_i agrees verbatim
    # across hint dirs wherever both define it (an instance can be absent from
    # one hint dir when 04 dropped it there), then merge definedness.
    acc_u = merged_acc_u(loaded, pool)

    vecs = {k: {h: _nan_vector(loaded[h]["maps"][k], pool) for h in hint_types}
            for k in POOLED_KEYS}

    # Point estimates: per-hint mean over that hint's defined subset, then the
    # unweighted mean of the per-hint values.
    per_hint = {k: {h: float(np.nanmean(vecs[k][h])) for h in hint_types
                    if not np.all(np.isnan(vecs[k][h]))}
                for k in POOLED_KEYS}
    point = {k: (float(np.mean(list(per_hint[k].values()))) if per_hint[k] else None)
             for k in POOLED_KEYS}
    point["acc_u"] = float(np.nanmean(acc_u))

    rng = np.random.default_rng(seed)
    reps = {k: [] for k in POOLED_KEYS + ["acc_u"]}
    with np.errstate(invalid="ignore"):
        for _ in range(n_resamples):
            drawn = rng.choice(pool, size=len(pool), replace=True)
            idx = np.searchsorted(pool, drawn)
            for k in POOLED_KEYS:
                means = [m for h in per_hint[k]
                         if not np.isnan(m := np.nanmean(vecs[k][h][idx]))]
                if means:
                    reps[k].append(float(np.mean(means)))
            r = np.nanmean(acc_u[idx])
            if not np.isnan(r):
                reps["acc_u"].append(float(r))

    out = {"n_pool": len(pool), "hint_types": hint_types,
           "missing_hint_types": [h for h in HINT_TYPES if h not in hint_types]}
    for k in POOLED_KEYS + ["acc_u"]:
        lo, hi = (float(v) for v in np.percentile(reps[k], [2.5, 97.5])) \
            if reps[k] else (None, None)
        entry = {"estimate": point[k], "ci95": [lo, hi],
                 "n_resamples": len(reps[k])}
        if k in POOLED_KEYS:
            entry["per_hint"] = per_hint[k]
            entry["per_hint_n_defined"] = {h: int((~np.isnan(vecs[k][h])).sum())
                                           for h in per_hint[k]}
        out[k] = entry
    return out


def delta_dataset_level(base_loaded: dict, trained_loaded: dict,
                        base_res: dict, trained_res: dict,
                        n_resamples: int, seed: int) -> dict:
    """Paired-bootstrap 95% CI on (trained − base), per dataset and metric.

    Both models ran the identical prepared instance pool, so each replicate
    draws one set of instance ids per dataset and reuses it for BOTH models
    (and, within each model, across the three hint types) — propagating the
    shared-question covariance that makes the CI on the difference valid and
    far tighter than comparing the two marginal CIs. Each model averages over
    its own defined subset of the drawn ids (same convention as the per-model
    pooling), so level-1-filtering differences are propagated too. The point
    estimate is the difference of the two REPORTED pooled estimates.
    """
    out = {}
    for ds in base_res:
        if ds not in trained_res:
            continue
        pool_sets = [set(e["pool"]) for e in base_loaded[ds].values()]
        pool_sets += [set(e["pool"]) for e in trained_loaded[ds].values()]
        pool = np.array(sorted(set().union(*pool_sets)))

        vecs = {}
        for tag, loaded in (("b", base_loaded[ds]), ("t", trained_loaded[ds])):
            for k in POOLED_KEYS:
                vecs[(tag, k)] = [_nan_vector(loaded[h]["maps"][k], pool)
                                  for h in loaded]
            vecs[(tag, "acc_u")] = [merged_acc_u(loaded, pool)]

        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(pool), size=(n_resamples, len(pool)))
        out[ds] = {}
        with np.errstate(invalid="ignore"):
            for key in POOLED_KEYS + ["acc_u"]:
                def pooled_reps(tag):
                    hint_means = np.stack([np.nanmean(v[idx], axis=1)
                                           for v in vecs[(tag, key)]])
                    return np.nanmean(hint_means, axis=0)
                reps = pooled_reps("t") - pooled_reps("b")
                lo, hi = (float(v) for v in
                          np.nanpercentile(reps, [2.5, 97.5]))
                b = base_res[ds][key]["estimate"]
                t = trained_res[ds][key]["estimate"]
                out[ds][key] = {
                    "delta": (t - b) if b is not None and t is not None
                    else None,
                    "ci95": [lo, hi],
                }
    return out


# --- markdown rendering ------------------------------------------------------

def _fmt_cell(entry: dict, key: str, n_hints_expected: int) -> str:
    est, (lo, hi) = entry["estimate"], entry["ci95"]
    if est is None:
        return "N/A"
    if key == "pct_te_pos":
        cell = f"{100 * est:.0f}%"
    else:
        cell = f"{est:.3f} [{lo:.3f}, {hi:.3f}]"
    if key != "acc_u" and len(entry.get("per_hint", {})) < n_hints_expected:
        cell += f" †({len(entry['per_hint'])} hints)"
    return cell


def render_model_table(model_id: str, results: dict) -> str:
    """One markdown table: rows = metrics, columns = datasets grouped by domain."""
    columns = [(dom, ds) for dom, keys in DOMAINS for ds in keys if ds in results]
    header = "| Metric | " + " | ".join(
        f"**{dom}**<br>`{ds}`" for dom, ds in columns) + " |"
    sep = "|---" * (len(columns) + 1) + "|"
    lines = [header, sep]
    for key, label, _ in METRICS:
        cells = [_fmt_cell(results[ds][key], key, len(HINT_TYPES))
                 for _, ds in columns]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def delta_cell_str(entry: dict, key: str) -> str:
    """Format one delta cell: point estimate + paired-bootstrap 95% CI."""
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
    """Delta table: (trained − base) with paired-bootstrap 95% CIs."""
    columns = [(dom, ds) for dom, keys in DOMAINS for ds in keys
               if ds in deltas]
    header = "| Metric (Δ = trained − base) [paired-boot 95% CI] | " + \
        " | ".join(f"**{dom}**<br>`{ds}`" for dom, ds in columns) + " |"
    lines = [header, "|---" * (len(columns) + 1) + "|"]
    for key, label, _ in METRICS:
        label = label.split(" [")[0]  # drop the per-model CI annotation
        cells = [delta_cell_str(deltas[ds].get(key), key) for _, ds in columns]
        lines.append(f"| Δ {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


FOOTNOTES = """\
**Delta tables**: each cell is the trained model's pooled point estimate minus
the base model's, with its own **paired-bootstrap 95% CI**: each replicate
draws one set of instance ids per dataset and reuses it for BOTH models and
all three hint types (10,000 resamples, seed 88), so the shared-question
covariance is propagated — these intervals are much tighter than comparing the
two models' separate CIs, which is the correct behavior for a paired design.
Each model averages over its own defined subset of the drawn ids, so
level-1-filtering differences are propagated too. A delta CI excluding 0 is a
paired test at the 5% level. The `% with TE > 0` row is in percentage points
(pp).

**CI conventions** (per-model tables) — the two interval types are not the same
estimator:

- **Paired-boot 95% CI** (pooled rows): paired cluster bootstrap over instances —
  each replicate draws n instance ids with replacement once and reuses that same
  draw across the three hint types (10,000 resamples, seed 88). The pooled point
  estimate is the unweighted mean of the three per-hint-type estimates
  (professor, grader_hacking, unethical), never a mean over pooled instances.
  Hint types are treated as fixed, so the interval covers question-sampling
  uncertainty only.
- **95% CI** (unhinted accuracy row): plain percentile bootstrap over instances —
  the unhinted arm is a single condition, shared verbatim by all hint types, so
  no pooling applies.

*V* = level-1-passing instances (TE > 0 and p1 > 0). Each dataset: n = 200
instances, 8 samples per arm. † marks a cell pooled over fewer than 3 hint types.\
"""


def render_report(models: list, all_results: dict, deltas_all: dict,
                  split: str) -> str:
    out = ["# Main results — monitorability pooled over hint types", ""]
    if split != "test":
        out += [f"*Split: {split}*", ""]
    out += ["## Per-model results", ""]
    for label, model_id in models:
        out += [f"### {label} — `{model_id}`", "",
                render_model_table(model_id, all_results[model_id]), ""]
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
        description="Pooled-over-hint-types summary tables for the main results.")
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

    grouped = {ds for _, keys in DOMAINS for ds in keys}
    ungrouped = [ds for ds in datasets if ds not in grouped]
    if ungrouped:
        print(f"WARNING: dataset(s) not in any DOMAINS group, omitted from "
              f"tables: {', '.join(ungrouped)}")

    models = [("Base", args.base),
              ("Harmlessness DPO", args.harmless),
              ("Anti-sycophancy DPO", args.antisyc)]

    all_results, loaded_all, out_dirs = {}, {}, {}
    for label, model_id in models:
        model_name = model_dir_name(model_id)
        print(f"\n=== {label}: {model_name} (split: {args.data}, "
              f"{n_resamples} resamples, seed {args.seed}) ===")
        results = {}
        loaded_all[model_id] = {}
        for dataset_key in datasets:
            loaded = {}
            for hint_type in HINT_TYPES:
                paths = resolve_paths(root, config, dataset_key, model_name,
                                      hint_type, split=args.data)
                entry = load_hint_type(paths["analysis_dir"])
                if entry is not None:
                    loaded[hint_type] = entry
            if not loaded:
                print(f"  {dataset_key:<14} SKIP — no results_per_instance.csv found")
                continue
            loaded_all[model_id][dataset_key] = loaded
            res = summarize_dataset(loaded, n_resamples, args.seed)
            if res["missing_hint_types"]:
                print(f"  WARNING: {dataset_key} pools only "
                      f"{len(res['hint_types'])} hint type(s); missing "
                      f"{', '.join(res['missing_hint_types'])}")
            g = res["gmean2"]
            print(f"  {dataset_key:<14} g-mean²={g['estimate']:.3f} "
                  f"CI [{g['ci95'][0]:.3f}, {g['ci95'][1]:.3f}]  "
                  f"acc_u={res['acc_u']['estimate']:.3f}")
            results[dataset_key] = res

        if not results:
            print(f"ERROR: no datasets produced results for {model_name}.")
            sys.exit(1)
        all_results[model_id] = results

        out_dir = root / config["paths"]["analysis_dir"] / model_name
        if args.data != "test":
            out_dir = out_dir / args.data
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": model_id, "split": args.data,
            "n_resamples": n_resamples, "seed": args.seed,
            "pooling": "unweighted mean of per-hint-type estimates",
            "ci_method": {
                "pooled_rows": "paired cluster bootstrap over instances; one "
                               "shared resample reused across hint types; hint "
                               "types fixed",
                "acc_u": "plain percentile bootstrap over instances "
                         "(single shared arm)",
            },
            "datasets": results,
        }
        json_path = out_dir / "pooled_summary.json"
        with json_path.open("w") as f:
            json.dump(payload, f, indent=2)
        print(f"  Wrote {json_path}")
        out_dirs[model_id] = out_dir

    # Paired-bootstrap CIs on (trained − base), per trained model.
    base_id = models[0][1]
    deltas_all = {}
    for label, model_id in models[1:]:
        print(f"\n=== deltas: {label} − Base ===")
        deltas = delta_dataset_level(
            loaded_all[base_id], loaded_all[model_id],
            all_results[base_id], all_results[model_id],
            n_resamples, args.seed)
        for ds, d in deltas.items():
            g = d["gmean2"]
            print(f"  {ds:<14} Δg-mean²={g['delta']:+.3f} "
                  f"CI [{g['ci95'][0]:+.3f}, {g['ci95'][1]:+.3f}]")
        deltas_all[model_id] = deltas
        payload = {
            "trained_model": model_id, "base_model": base_id,
            "split": args.data, "n_resamples": n_resamples,
            "seed": args.seed,
            "method": "paired cluster bootstrap on the difference; one "
                      "instance resample per replicate shared across both "
                      "models and all hint types; point estimate = difference "
                      "of the reported pooled estimates",
            "datasets": deltas,
        }
        json_path = out_dirs[model_id] / "delta_vs_base.json"
        with json_path.open("w") as f:
            json.dump(payload, f, indent=2)
        print(f"  Wrote {json_path}")

    report_dir = root / config["paths"]["analysis_dir"]
    if args.data != "test":
        report_dir = report_dir / args.data
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "main_results_dataset_level_tables.md"
    report_path.write_text(render_report(models, all_results, deltas_all,
                                         args.data))
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
