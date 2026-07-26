#!/usr/bin/env python3
"""
Step 49: Step 47's construction (no cross-hint averaging) applied PER DATASET
— one 2 x 4 small multiple, one panel per dataset, each pooling all three hint
conditions as SEPARATE observations.

Relation to the neighbouring steps:
  - Step 42 also draws 8 per-dataset panels, but first collapses each question
    by averaging its length and `gmean2_i` over the hint conditions it entered.
    That collapse is confounded: a question contributes 1, 2 or 3 conditions
    (base: 541 / 317 / 525) and the count correlates with length (Spearman
    +0.34) and with monitorability (mean g-mean^2 0.946 / 0.850 / 0.674).
  - Step 47 removes the collapse but pools all datasets into one panel per hint.
  - THIS step removes the collapse AND keeps the dataset split: within a panel
    every (question, hint) measurement is its own row, paired with the length
    actually produced under that hint. No averaging anywhere.

The cost of not averaging is CLUSTERING: one question can supply up to 3 rows,
so rows are not independent. Every interval here is therefore a CLUSTER
bootstrap that resamples QUESTIONS (with all their rows) rather than rows —
both for the per-bin mean g-mean^2 and for Spearman rho. Treating the rows as
independent would understate the CIs by roughly sqrt(rows/questions).

Binning and axes match steps 42/47: equal-count quantile bins computed within
each dataset, dots at the bin median length on a shared log x-axis, y = bin
mean `gmean2_i`. Panels share x and y limits.

rho is reported per panel with its cluster-bootstrap CI, so whether an effect
is distinguishable from zero is read straight off the interval.
Note the caveats that apply to every rho in this folder: `gmean2_i` takes only
69 distinct values over 2750 measurements (57% exactly at the 1.0 ceiling), so
rho is attenuated and should be read as a conservative lower bound.

Usage:
    python scripts/49_hinted_length_gmean2_quantile_by_dataset_nocollapse.py
    python scripts/49_hinted_length_gmean2_quantile_by_dataset_nocollapse.py \
        --model <hf-id>

    Options:
      --config     path to config YAML (default: config/mvp_config.yaml)
      --data       split: test (default) | eval
      --model      model/adapter HF id (default: Qwen/Qwen3-4B)
      --bins       equal-count quantile bins per dataset (default: 5)
      --no-rho     omit the rho box, leaving only n (default: show rho)
      --linear-x   linear length axis (default: log10)
      --resamples  cluster-bootstrap replicates (default: 5000)
      --seed       bootstrap seed (default: 88)
      --out        output PNG path (default:
                   {plots_dir}/cot-length-vs-monitorability/
                   gmean2_by_hinted_length_quantile_by_dataset_{model}.png)

Input (per dataset / hint type, produced by 04_compute_gmean2.py):
    {analysis_dir}/{model}/{dataset}/{hint}/results_per_instance.csv

Output:
    {plots_dir}/cot-length-vs-monitorability/
        gmean2_by_hinted_length_quantile_by_dataset_{model}.png
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import yaml

from src.dataset import SPLITS, model_dir_name, resolve_paths
from src.hints import HINT_TYPES

DEFAULT_MODEL = "Qwen/Qwen3-4B"
DATASETS = ["mmlu_pro", "gpqa_diamond", "lsat_ar", "lsat_lr",
            "medxpertqa", "lexam", "codemmlu", "wmdp"]
PLUM, PLUM_FC, PLUM_EC = "#6A4C93", "#F4F1F8", "#C9BEDD"


def load_dataset(root, config, model_name, dataset, split):
    """All (question, hint) measurements for one dataset — no averaging.
    Returns length, gmean2, and the question id each row belongs to."""
    L, G, Q = [], [], []
    for hint in HINT_TYPES:
        paths = resolve_paths(root, config, dataset, model_name, hint,
                              split=split)
        csv_path = paths["analysis_dir"] / "results_per_instance.csv"
        if not csv_path.exists():
            continue
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                if row.get("level1_pass") != "True":
                    continue
                g = row.get("gmean2_i", "")
                length = row.get("mean_cot_tokens_hinted", "")
                if g in ("", "None", "N/A") or length in ("", "None", "N/A"):
                    continue
                L.append(float(length))
                G.append(float(g))
                Q.append(int(row["instance_id"]))
    if not L:
        return None, None, None
    return np.array(L), np.array(G), np.array(Q)


def cluster_indices(qids, rng, n_resamples):
    """Yield row-index arrays from resampling QUESTIONS with replacement."""
    uniq = np.unique(qids)
    by_q = {q: np.flatnonzero(qids == q) for q in uniq}
    for _ in range(n_resamples):
        drawn = rng.integers(0, len(uniq), len(uniq))
        yield np.concatenate([by_q[uniq[i]] for i in drawn])


def cluster_ci_mean(vals, qids, rng, n_resamples):
    if len(np.unique(qids)) < 2:
        return float("nan"), float("nan")
    m = np.fromiter((vals[idx].mean()
                     for idx in cluster_indices(qids, rng, n_resamples)),
                    float, n_resamples)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def analyse(L, G, Q, nb, resamples, rng):
    edges = np.percentile(L, np.linspace(0, 100, nb + 1))
    edges[-1] = np.nextafter(edges[-1], np.inf)
    drawn = []
    for i in range(nb):
        m = (L >= edges[i]) & (L < edges[i + 1])
        if m.sum() == 0:
            continue
        lo, hi = cluster_ci_mean(G[m], Q[m], rng, resamples)
        drawn.append(dict(med=float(np.median(L[m])), mean=float(G[m].mean()),
                          lo=lo, hi=hi, n=int(m.sum()),
                          nq=int(len(np.unique(Q[m])))))
    rho, pval = stats.spearmanr(L, G)
    bs = np.fromiter((stats.spearmanr(L[idx], G[idx])[0]
                      for idx in cluster_indices(Q, rng, resamples)),
                     float, resamples)
    rlo, rhi = np.percentile(bs, [2.5, 97.5])
    return dict(drawn=drawn, rho=float(rho), rlo=float(rlo), rhi=float(rhi),
                p=float(pval), n=len(L), nq=int(len(np.unique(Q))),
                lmin=float(L.min()), lmax=float(L.max()))


def main():
    parser = argparse.ArgumentParser(
        description="Monitorability vs HINTED length per dataset, pooling "
                    "hint conditions without averaging them.")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--data", default="test", choices=SPLITS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--no-rho", action="store_true")
    parser.add_argument("--linear-x", action="store_true")
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=88)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    with config_path.open() as f:
        config = yaml.safe_load(f)

    model_name = model_dir_name(args.model)
    rng = np.random.default_rng(args.seed)
    use_log = not args.linear_x

    panels = {}
    print(f"{'dataset':<15}{'rows':>6}{'questions':>11}{'rho':>9}"
          f"{'cluster 95% CI':>20}")
    for dataset in DATASETS:
        L, G, Q = load_dataset(root, config, model_name, dataset, args.data)
        if L is None:
            panels[dataset] = None
            print(f"{dataset:<15}     -- no entering measurements, blanked")
            continue
        res = analyse(L, G, Q, args.bins, args.resamples, rng)
        panels[dataset] = res
        print(f"{dataset:<15}{res['n']:>6}{res['nq']:>11}{res['rho']:>+9.3f}"
              f"   [{res['rlo']:+.3f}, {res['rhi']:+.3f}]")

    live = [p for p in panels.values() if p]
    if not live:
        print("\nERROR: no populated datasets — nothing to plot.")
        sys.exit(1)

    print(f"\nRows are clustered ({sum(p['n'] for p in live)} rows from "
          f"{sum(p['nq'] for p in live)} questions): every CI above resamples "
          f"QUESTIONS, not rows.")
    print("Caveat: gmean2_i has only 69 distinct values (57% at the 1.0 "
          "ceiling), so every rho here is attenuated — a lower bound.")

    bounds = [b for p in live for d in p["drawn"] for b in (d["lo"], d["hi"])]
    ylo, yhi = min(bounds) - 0.05, max(bounds) + 0.05
    xlo = min(p["lmin"] for p in live) * 0.8
    xhi = max(p["lmax"] for p in live) * 1.2

    fig, axes = plt.subplots(2, 4, figsize=(16, 7.8), sharex=True, sharey=True)
    for ax, dataset in zip(axes.flat, DATASETS):
        ax.set_axisbelow(True)
        if use_log:
            ax.set_xscale("log")
        ax.grid(True, color="#EEEEEE", linewidth=0.8)
        ax.set_title(dataset, fontsize=11, fontweight="bold")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#999999")

        res = panels.get(dataset)
        if not res:
            ax.text(0.5, 0.5, "insufficient data", transform=ax.transAxes,
                    ha="center", va="center", color="#999999", fontsize=9)
            continue

        d = res["drawn"]
        xs = [b["med"] for b in d]
        ys = [b["mean"] for b in d]
        elo = [b["mean"] - b["lo"] for b in d]
        ehi = [b["hi"] - b["mean"] for b in d]
        ax.errorbar(xs, ys, yerr=[elo, ehi], color=PLUM, linewidth=1.8,
                    marker="o", markersize=6, markeredgecolor="#333333",
                    markeredgewidth=0.4, ecolor=PLUM, elinewidth=1.2,
                    capsize=3, capthick=1.2, zorder=3)

        top_frac = max((b["hi"] - ylo) / (yhi - ylo) for b in d[-2:])
        box_y, box_va = (0.05, "bottom") if top_frac > 0.66 else (0.95, "top")
        if args.no_rho:
            label = f"n={res['n']} rows\n{res['nq']} questions"
        else:
            label = (f"ρ = {res['rho']:+.2f}\n"
                     f"[{res['rlo']:+.2f}, {res['rhi']:+.2f}]\n"
                     f"n={res['n']} rows / {res['nq']} q")
        ax.text(0.96, box_y, label, transform=ax.transAxes, ha="right",
                va=box_va, fontsize=8, color=PLUM,
                bbox=dict(boxstyle="round,pad=0.3", fc=PLUM_FC, ec=PLUM_EC))

    axes.flat[0].set_xlim(xlo, xhi)
    axes.flat[0].set_ylim(ylo, yhi)
    for ax in axes[:, 0]:
        ax.set_ylabel("Monitorability (g-mean²)\nbars: cluster bootstrap 95% CI",
                      fontsize=9)
    fig.supxlabel("HINTED response length — per-(question, hint) mean over 8 "
                  f"runs (tokens{', log scale' if use_log else ''}; dots at "
                  "bin median)", fontsize=10, y=0.02)
    subtitle = ("every (question, hint) measurement is its own row — no "
                "cross-hint averaging and no level-1 entry-count confound "
                "(contrast step 42)\nrows are clustered within questions, so "
                "all intervals resample QUESTIONS")
    fig.suptitle(f"Monitorability by HINTED length, per dataset — {model_name}\n"
                 + subtitle, fontsize=12)

    out = args.out
    if out is None:
        plots_dir = ((root / config["paths"]["plots_dir"]).parent
                     / "extra-plots" / "cot-length-vs-monitorability")
        if args.data != "test":
            plots_dir = plots_dir / args.data
        plots_dir.mkdir(parents=True, exist_ok=True)
        out = (plots_dir / "gmean2_by_hinted_length_quantile_by_dataset_"
                           f"{model_name}.png")
    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
