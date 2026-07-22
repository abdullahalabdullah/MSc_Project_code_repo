#!/usr/bin/env python3
"""
Step 20: Main-results summary tables at (dataset, hint-type) level — one table
per (dataset, hint type), same six metric rows as steps 15/19. Columns = the
three models PLUS two delta columns (each trained model − base) carrying a
paired-bootstrap 95% CI on the difference.

Per-model columns involve NO pooling, so their CIs are the plain percentile
cluster bootstrap over instances written by 04_compute_gmean2.py into
results_summary.json (bootstrap_cis: 10,000 resamples, seed 88), read
straight from that file:
    g-mean^2            <- gmean2            + ci95.gmean2
    Mean TE on V        <- mean_TE_after     + ci95.TE_after
    Mean R_effect on V  <- mean_R_effect_after + ci95.R_effect_after
    % with TE > 0       <- n_te_positive / n_total   (point estimate only in
                           the per-model columns, matching the pooled reports)
    Hinted accuracy     <- mean_acc_h        + ci95.acc_h
    Unhinted accuracy   <- mean_acc_u        + ci95.acc_u
The unhinted arm is shared verbatim by the three hint types of a dataset, so
its row repeats across those three tables.

Delta columns: both models ran the identical prepared instances, so each
replicate draws one set of instance ids from the cell's shared pool and
reuses it for BOTH models (10,000 resamples, seed 88), propagating the
shared-question covariance; each model averages over its own defined subset
of the drawn ids. Point estimate = difference of the two reported values.
Computed from results_per_instance.csv (via step 15's loader).

Usage:
    python scripts/20_dataset_hint_summary_tables.py

    Options: --config, --data test|eval, --resamples, --seed,
             --base / --harmless / --antisyc   (as in step 15)

Input (per model / dataset / hint type, produced by 04_compute_gmean2.py):
    {analysis_dir}/{model}/{dataset}/{hint}/results_summary.json
    {analysis_dir}/{model}/{dataset}/{hint}/results_per_instance.csv

Output:
    {analysis_dir}/main_results_dataset_hint_level_tables.md
    {analysis_dir}/{trained_model}/delta_vs_base_per_cell.json  (per trained model)
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

# Reuse step 15's domain grouping/order and per-instance loader so the
# reports stay consistent.
sys.path.insert(0, str(Path(__file__).resolve().parent))
_s15 = __import__("15_pooled_summary_tables")
DOMAINS = _s15.DOMAINS
load_hint_type = _s15.load_hint_type
_nan_vector = _s15._nan_vector

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

# row label -> (summary estimate key, summary ci95 key, per-instance key)
ROWS = [
    ("g-mean² [95% CI]", "gmean2", "gmean2", "gmean2"),
    ("Mean TE on V [95% CI]", "mean_TE_after", "TE_after", "te_on_v"),
    ("Mean R_effect on V [95% CI]", "mean_R_effect_after", "R_effect_after",
     "r_effect_on_v"),
    ("% with TE > 0", None, None, "pct_te_pos"),
    ("Hinted accuracy [95% CI]", "mean_acc_h", "acc_h", "acc_h"),
    ("Unhinted accuracy (single arm) [95% CI]", "mean_acc_u", "acc_u",
     "acc_u"),
]

FOOTNOTES = """\
**CI conventions** — the two interval types are not the same estimator:

- **Per-model columns**: a plain percentile cluster bootstrap over instances
  (10,000 resamples, seed 88), computed per (model, dataset, hint type) by
  `04_compute_gmean2.py` (`results_summary.json` → `ci95`). Nothing here is
  pooled.
- **Δ columns** (trained − base): a **paired-bootstrap 95% CI** on the
  difference — both models ran the identical prepared instances, so each
  replicate draws one set of instance ids from the cell's shared pool and
  reuses it for BOTH models (10,000 resamples, seed 88); each model averages
  over its own defined subset of the drawn ids. This propagates the
  shared-question covariance, so these intervals are much tighter than
  comparing the two per-model CIs; a Δ CI excluding 0 is a paired test at the
  5% level. The `% with TE > 0` Δ is in percentage points (pp).

*V* = level-1-passing instances (TE > 0 and p1 > 0). Each (dataset, hint type):
n = 200 instances, 8 samples per arm. The unhinted arm is a single condition
shared verbatim by a dataset's three hint types, so its rows repeat across
those tables.\
"""


def load_summary(root, config, model_name, dataset, hint, split):
    paths = resolve_paths(root, config, dataset, model_name, hint, split=split)
    p = paths["analysis_dir"] / "results_summary.json"
    if not p.exists():
        return None
    with p.open() as f:
        return json.load(f)


def point_value(summary, est_key):
    """The displayed point estimate for a metric (fractional for % row)."""
    if summary is None:
        return None
    if est_key is None:  # % with TE > 0
        n_total, n_pos = summary.get("n_total"), summary.get("n_te_positive")
        return n_pos / n_total if n_total and n_pos is not None else None
    return summary.get(est_key)


def cell(summary, est_key, ci_key, row_label):
    if summary is None:
        return "N/A"
    est = point_value(summary, est_key)
    if est is None:
        return "N/A"
    if row_label == "% with TE > 0":
        return f"{100 * est:.0f}%"
    ci = (summary.get("ci95") or {}).get(ci_key)
    if ci is None:
        return f"{est:.3f}"
    return f"{est:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def cell_deltas(per_instance: dict, summaries: dict, base_id: str,
                trained_ids: list, n_resamples: int, seed: int) -> dict:
    """Paired-bootstrap 95% CI on (trained − base) for one (dataset, hint).

    One instance resample per replicate, shared across the base and every
    trained model; each model averages over its own defined subset. Point
    estimate = difference of the displayed per-model values.
    """
    entries = {m: per_instance.get(m) for m in [base_id] + trained_ids}
    if any(e is None for e in entries.values()):
        return {t: None for t in trained_ids}
    pool = np.array(sorted(set().union(
        *(set(e["pool"]) for e in entries.values()))))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pool), size=(n_resamples, len(pool)))
    reps = {}
    with np.errstate(invalid="ignore"):
        for m, e in entries.items():
            reps[m] = {inst_key: np.nanmean(
                _nan_vector(e["maps"][inst_key], pool)[idx], axis=1)
                for _, _, _, inst_key in ROWS}
    out = {}
    for t in trained_ids:
        out[t] = {}
        for _, est_key, _, inst_key in ROWS:
            delta_reps = reps[t][inst_key] - reps[base_id][inst_key]
            lo, hi = (float(v) for v in
                      np.nanpercentile(delta_reps, [2.5, 97.5]))
            b = point_value(summaries.get(base_id), est_key)
            tv = point_value(summaries.get(t), est_key)
            out[t][inst_key] = {
                "delta": (tv - b) if b is not None and tv is not None
                else None,
                "ci95": [lo, hi],
            }
    return out


def delta_cell_str(entry, row_label):
    if entry is None or entry["delta"] is None:
        return "N/A"
    d, (lo, hi) = entry["delta"], entry["ci95"]
    if row_label == "% with TE > 0":
        dp = round(100 * d)
        return (f"{dp:+.0f}" if dp else "0") + \
            f" pp [{100 * lo:+.0f}, {100 * hi:+.0f}]"
    dp = round(d, 3)
    return (f"{dp:+.3f}" if dp else "0.000") + f" [{lo:+.3f}, {hi:+.3f}]"


def render_table(summaries: dict, deltas: dict, models: list) -> str:
    delta_cols = [f"Δ {lbl} − Base [paired-boot 95% CI]"
                  for lbl, _ in models[1:]]
    header = ("| Metric | " + " | ".join(lbl for lbl, _ in models)
              + " | " + " | ".join(delta_cols) + " |")
    lines = [header, "|---" * (len(models) + len(delta_cols) + 1) + "|"]
    for row_label, est_key, ci_key, inst_key in ROWS:
        cells = [cell(summaries[model_id], est_key, ci_key, row_label)
                 for _, model_id in models]
        cells += [delta_cell_str((deltas.get(model_id) or {}).get(inst_key),
                                 row_label)
                  for _, model_id in models[1:]]
        lines.append(f"| {row_label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Per-(dataset, hint-type) summary tables, plain "
                    "bootstrap CIs only.")
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
    datasets = list(config.get("datasets", {}).keys())
    n_resamples = args.resamples or config.get("gmean2", {}).get(
        "bootstrap_resamples", DEFAULT_RESAMPLES)

    models = [("Base", args.base),
              ("Harmlessness DPO", args.harmless),
              ("Anti-sycophancy DPO", args.antisyc)]
    base_id = models[0][1]
    trained_ids = [mid for _, mid in models[1:]]

    ordered = [(dom, ds) for dom, keys in DOMAINS for ds in keys
               if ds in datasets]
    leftover = [ds for ds in datasets
                if ds not in {d for _, d in ordered}]
    ordered += [("Other", ds) for ds in leftover]

    out = ["# Main results — per (dataset, hint type), no pooling", ""]
    if args.data != "test":
        out += [f"*Split: {args.data}*", ""]
    out += ["Columns: " + " · ".join(f"**{lbl}** = `{mid}`"
                                     for lbl, mid in models), ""]
    n_tables = 0
    all_deltas = {t: {} for t in trained_ids}
    for dom, ds in ordered:
        out += [f"## {ds} ({dom})", ""]
        for hint in HINT_TYPES:
            summaries, per_instance = {}, {}
            for _, model_id in models:
                model_name = model_dir_name(model_id)
                summaries[model_id] = load_summary(
                    root, config, model_name, ds, hint, args.data)
                paths = resolve_paths(root, config, ds, model_name, hint,
                                      split=args.data)
                per_instance[model_id] = load_hint_type(paths["analysis_dir"])
            missing = [mid for mid, s in summaries.items() if s is None]
            if missing:
                print(f"WARNING: {ds}/{hint} missing results_summary.json "
                      f"for: {', '.join(missing)}")
            deltas = cell_deltas(per_instance, summaries, base_id,
                                 trained_ids, n_resamples, args.seed)
            for t in trained_ids:
                if deltas[t] is not None:
                    all_deltas[t][f"{ds}/{hint}"] = deltas[t]
            out += [f"### {ds} — {HINT_LABELS.get(hint, hint)}", "",
                    render_table(summaries, deltas, models), ""]
            n_tables += 1

    out += ["---", "", FOOTNOTES, ""]

    for t in trained_ids:
        t_dir = root / config["paths"]["analysis_dir"] / model_dir_name(t)
        if args.data != "test":
            t_dir = t_dir / args.data
        t_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "trained_model": t, "base_model": base_id, "split": args.data,
            "n_resamples": n_resamples, "seed": args.seed,
            "method": "paired cluster bootstrap on the difference; one "
                      "instance resample per replicate shared across both "
                      "models; point estimate = difference of the reported "
                      "values",
            "cells": all_deltas[t],
        }
        json_path = t_dir / "delta_vs_base_per_cell.json"
        with json_path.open("w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {json_path}")

    report_dir = root / config["paths"]["analysis_dir"]
    if args.data != "test":
        report_dir = report_dir / args.data
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "main_results_dataset_hint_level_tables.md"
    report_path.write_text("\n".join(out))
    print(f"Wrote {report_path} ({n_tables} tables)")


if __name__ == "__main__":
    main()
