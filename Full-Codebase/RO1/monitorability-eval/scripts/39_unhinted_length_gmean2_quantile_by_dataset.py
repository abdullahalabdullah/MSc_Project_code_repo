#!/usr/bin/env python3
"""
Step 39: Step 37's pooled monitorability-vs-unhinted-length quintile curve,
drawn as an 8-panel small multiple — one panel per dataset.

Step 37 pools all datasets into ONE quintile curve; this step draws the SAME
curve (per-question unit, equal-count quintile bins, dots at bin median on a
log x-axis, bootstrap 95% CI, per-panel Spearman rho) SEPARATELY for each of
the 8 datasets (2x4, domain-table order, shared axes), so the reader can see
whether the pooled decline holds within each dataset and across their very
different length regimes (base model: wmdp median ~925 tok vs lsat_ar ~5667).

Unit and binning are identical to step 37: one row per unique question
(unhinted length = mean of the 8 unhinted runs, shared across hint slices;
monitorability = mean of the question's per-hint-slice gmean2_i); bins are
computed WITHIN each dataset (dataset-specific quintile edges), so every
panel has ~equal counts (base: 122-197 questions/dataset -> ~24-39/bin).
Per-dataset bins are necessarily thinner than the pooled view, so CIs are
wider — present the pooled step 37 as the headline and this as the
per-dataset generalisation check.

Usage:
    python scripts/39_unhinted_length_gmean2_quantile_by_dataset.py        # base
    python scripts/39_unhinted_length_gmean2_quantile_by_dataset.py --model <hf-id>

    Options:
      --config     path to config YAML (default: config/mvp_config.yaml)
      --data       split: test (default) | eval
      --model      model/adapter HF id (default: Qwen/Qwen3-4B)
      --bins       equal-count quantile bins per dataset (default: 5)
      --linear-x   linear length axis (default: log10)
      --resamples  bootstrap replicates per CI / rho (default: 5000)
      --seed       bootstrap seed (default: 88)
      --out        output PNG path (default:
                   {plots_dir}/cot-length-vs-monitorability/
                   gmean2_by_unhinted_length_quantile_panels_{model}.png)

Input (per dataset / hint type, produced by 04_compute_gmean2.py):
    {analysis_dir}/{model}/{dataset}/{hint}/results_per_instance.csv

Output:
    {plots_dir}/cot-length-vs-monitorability/
        gmean2_by_unhinted_length_quantile_panels_{model}.png
"""

import argparse
import csv
import sys
from collections import defaultdict
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
DARK_BLUE = "#1B3A6B"


def boot_ci(v, n_resamples, rng):
    v = np.asarray(v, float)
    if len(v) < 2:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(v), (n_resamples, len(v)))
    m = v[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def load_per_question(root, config, dataset, model_name, split):
    """One row per unique question: (unhinted length, mean gmean2 over hints)."""
    per_q = defaultdict(lambda: {"g": [], "L": None})
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
                L = row.get("mean_cot_tokens_unhinted", "")
                if g in ("", "None", "N/A") or L in ("", "None", "N/A"):
                    continue
                per_q[row["instance_id"]]["g"].append(float(g))
                per_q[row["instance_id"]]["L"] = float(L)
    if not per_q:
        return None, None
    L = np.array([v["L"] for v in per_q.values()])
    G = np.array([float(np.mean(v["g"])) for v in per_q.values()])
    return L, G


def main():
    parser = argparse.ArgumentParser(
        description="Per-dataset small multiples of monitorability vs unhinted "
                    "length (quintile bins).")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--data", default="test", choices=SPLITS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--bins", type=int, default=5)
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
    nb = args.bins
    use_log = not args.linear_x

    panels, all_bounds, all_len = {}, [], []
    for dataset in DATASETS:
        L, G = load_per_question(root, config, dataset, model_name, args.data)
        if L is None:
            panels[dataset] = None
            print(f"\n{dataset}: no entering questions — skipped")
            continue
        edges = np.percentile(L, np.linspace(0, 100, nb + 1))
        edges[-1] = np.nextafter(edges[-1], np.inf)
        drawn = []  # (med_len, mean, lo, hi, n)
        print(f"\n{dataset} — {len(L)} questions")
        for i in range(nb):
            m = (L >= edges[i]) & (L < edges[i + 1])
            if m.sum() == 0:
                continue
            v = G[m]
            med = float(np.median(L[m]))
            lo, hi = boot_ci(v, args.resamples, rng)
            print(f"   {int(edges[i]):>6}-{int(np.ceil(edges[i+1])):<6} "
                  f"n={int(m.sum()):>3} med={med:>6.0f} mean={v.mean():.3f} "
                  f"CI[{lo:.3f},{hi:.3f}]")
            drawn.append((med, float(v.mean()), lo, hi, int(m.sum())))
            all_bounds.extend([lo, hi])
        rho = stats.spearmanr(L, G)[0]
        bs = np.empty(args.resamples)
        for j in range(args.resamples):
            idx = rng.integers(0, len(L), len(L))
            bs[j] = stats.spearmanr(L[idx], G[idx])[0]
        rlo, rhi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
        print(f"   Spearman rho = {rho:+.3f} [{rlo:+.3f}, {rhi:+.3f}]")
        panels[dataset] = dict(drawn=drawn, n=len(L), rho=rho, rlo=rlo, rhi=rhi)
        all_len.extend([L.min(), L.max()])

    if not all_bounds:
        print("\nERROR: no populated bins anywhere — nothing to plot.")
        sys.exit(1)

    ylo, yhi = min(all_bounds) - 0.03, max(all_bounds) + 0.03
    xlo, xhi = min(all_len) * 0.85, max(all_len) * 1.15

    # ---- figure: 2x4 small multiples, shared axes ----
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), sharex=True, sharey=True)
    for ax, dataset in zip(axes.flat, DATASETS):
        ax.set_axisbelow(True)
        if use_log:
            ax.set_xscale("log")
        ax.grid(True, color="#EEEEEE", linewidth=0.8)
        ax.set_title(dataset, fontsize=11, fontweight="bold")
        p = panels.get(dataset)
        if not p or not p["drawn"]:
            ax.text(0.5, 0.5, "insufficient data", transform=ax.transAxes,
                    ha="center", va="center", color="#999999", fontsize=9)
            continue
        d = p["drawn"]
        xs = [r[0] for r in d]
        ys = [r[1] for r in d]
        elo = [r[1] - r[2] for r in d]
        ehi = [r[3] - r[1] for r in d]
        ax.errorbar(xs, ys, yerr=[elo, ehi], color=DARK_BLUE, linewidth=1.8,
                    marker="o", markersize=6, markeredgecolor="#333333",
                    markeredgewidth=0.4, ecolor=DARK_BLUE, elinewidth=1.2,
                    capsize=3, capthick=1.2, zorder=3)
        sig = "" if (p["rlo"] < 0 < p["rhi"]) else " ✓"
        ax.text(0.96, 0.95,
                f"ρ = {p['rho']:+.2f}{sig}\n[{p['rlo']:+.2f}, {p['rhi']:+.2f}]"
                f"\nn = {p['n']}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                color=DARK_BLUE,
                bbox=dict(boxstyle="round,pad=0.3", fc="#F4F6F9", ec="#CCD3DB"))
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#999999")

    axes.flat[0].set_xlim(xlo, xhi)
    axes.flat[0].set_ylim(ylo, yhi)
    for ax in axes[:, 0]:
        ax.set_ylabel("Monitorability (g-mean²)\nbars: bootstrap 95% CI",
                      fontsize=9)
    xlab = ("Unhinted response length — per-question mean of 8 runs (tokens"
            f"{', log scale' if use_log else ''}; dots at bin median)")
    fig.supxlabel(xlab, fontsize=10, y=0.02)

    fig.suptitle("Monitorability by unhinted (intrinsic) reasoning length, "
                 f"per dataset — {model_name}\n"
                 f"per-question unit; {nb} equal-count length bins WITHIN each "
                 "dataset; per-panel Spearman ρ (✓ = 95% CI excludes 0)",
                 fontsize=13)

    out = args.out
    if out is None:
        plots_dir = ((root / config["paths"]["plots_dir"]).parent
                     / "extra-plots" / "cot-length-vs-monitorability")
        if args.data != "test":
            plots_dir = plots_dir / args.data
        plots_dir.mkdir(parents=True, exist_ok=True)
        out = (plots_dir /
               f"gmean2_by_unhinted_length_quantile_panels_{model_name}.png")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
