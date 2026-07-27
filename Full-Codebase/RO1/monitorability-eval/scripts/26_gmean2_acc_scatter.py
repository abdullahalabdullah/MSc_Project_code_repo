#!/usr/bin/env python3
"""
Step 26: Per-dataset scatter of unhinted task accuracy vs g-mean^2 for ONE
model, with a fitted trend line.

One panel per dataset (2x4 small multiples). Each dot is one ENTERING
question (level-1 pass, gmean2_i defined), pooled over the 3 hint types:
x = its unhinted accuracy acc_u_i (fraction of the N unhinted runs answered
correctly), y = its per-question g-mean^2. Both variables are discrete
(acc_u_i takes the 9 values k/8; gmean2_i is a product of run-count
fractions), so without treatment the cloud collapses onto a lattice —
dots get a small horizontal jitter (display only; fits and stats use the
raw values) and transparency so stacked mass reads as darkness.

Per panel: a Pearson r / p-value annotation computed on the raw
(unjittered) values — no fitted trend line is drawn. Stdout prints
n / Pearson r / Spearman rho (both with p) per dataset plus the pooled
values, so captions can cite either statistic.

Usage:
    python scripts/26_gmean2_acc_scatter.py                    # base model
    python scripts/26_gmean2_acc_scatter.py --model <hf-id>

    Options:
      --config   path to config YAML (default: config/mvp_config.yaml)
      --data     split: test (default) | eval
      --model    model/adapter HF id (default: Qwen/Qwen3-4B)
      --jitter   half-width of the uniform x-jitter in accuracy units
                 (default: 0.035; 0 disables — levels are 0.125 apart)
      --seed     jitter RNG seed (default: 88)
      --out      output PNG path
                 (default: {plots_dir}/gmean2_vs_acc_scatter_{model}.png)

Input (per dataset / hint type, produced by 04_compute_gmean2.py):
    {analysis_dir}/{model}/{dataset}/{hint}/results_per_instance.csv

Output:
    {plots_dir}/gmean2_vs_acc_scatter_{model}.png
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

# Same presentation order as the main-results tables / steps 16-18.
DATASETS = ["mmlu_pro", "gpqa_diamond", "lsat_ar", "lsat_lr",
            "medxpertqa", "lexam", "codemmlu", "wmdp"]

DOT_COLOR = "#2a78d6"   # light-ish blue for the point mass


def load_entering(csv_path: Path):
    """(acc_u_i, gmean2_i) for questions that enter g-mean^2."""
    out = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if row.get("level1_pass") != "True":
                continue
            g, a = row.get("gmean2_i", ""), row.get("acc_u_i", "")
            if g in ("", "None", "N/A") or a in ("", "None", "N/A"):
                continue
            out.append((float(a), float(g)))
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Per-dataset scatter of unhinted accuracy vs g-mean^2 "
                    "with fitted trend lines.")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--data", default="test", choices=SPLITS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--jitter", type=float, default=0.035)
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

    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5), sharex=True, sharey=True)
    missing, pooled_a, pooled_g = [], [], []

    for d_idx, dataset in enumerate(DATASETS):
        ax = axes[d_idx // 4][d_idx % 4]
        points = []
        for hint in HINT_TYPES:
            paths = resolve_paths(root, config, dataset, model_name, hint,
                                  split=args.data)
            csv_path = paths["analysis_dir"] / "results_per_instance.csv"
            if not csv_path.exists():
                missing.append(f"{dataset}/{hint}")
                continue
            points.extend(load_entering(csv_path))

        ax.set_axisbelow(True)
        ax.grid(True, color="#DDDDDD", linewidth=0.6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#999999")

        if len(points) < 10:
            ax.set_title(f"{dataset} (n={len(points)} — skipped)", fontsize=10)
            continue

        a = np.array([p[0] for p in points])
        g = np.array([p[1] for p in points])
        pooled_a.extend(a)
        pooled_g.extend(g)

        # Display-only jitter; every statistic below uses the raw values.
        xj = a + rng.uniform(-args.jitter, args.jitter, size=len(a))
        ax.scatter(xj, g, s=12, color=DOT_COLOR, alpha=0.18,
                   edgecolors="none", zorder=2)

        r, r_p = stats.pearsonr(a, g)
        rho, rho_p = stats.spearmanr(a, g)
        print(f"{dataset:14s} n={len(a):4d}  pearson r={r:+.3f} (p={r_p:.3g})"
              f"  spearman rho={rho:+.3f} (p={rho_p:.3g})")
        p_txt = f"p={r_p:.3f}" if r_p >= 0.001 else "p<0.001"
        ax.set_title(f"{dataset}  (n={len(a)})", fontsize=10)
        ax.text(0.03, 0.04, f"r={r:+.2f}, {p_txt}",
                transform=ax.transAxes, fontsize=8.5, color="#333333",
                va="bottom", ha="left",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none",
                          pad=1.5))

        ax.set_xlim(-0.06, 1.06)
        ax.set_ylim(-0.04, 1.04)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])

    if missing:
        print("WARNING: missing slices skipped: " + ", ".join(missing))
    if not pooled_a:
        print("ERROR: no data found — nothing to plot.")
        sys.exit(1)
    r, r_p = stats.pearsonr(pooled_a, pooled_g)
    rho, rho_p = stats.spearmanr(pooled_a, pooled_g)
    print(f"{'POOLED':14s} n={len(pooled_a):4d}  pearson r={r:+.3f} "
          f"(p={r_p:.3g})  spearman rho={rho:+.3f} (p={rho_p:.3g})")

    for ax in axes[1]:
        ax.set_xlabel("Unhinted accuracy (acc_u)")
    for row in axes:
        row[0].set_ylabel("g-mean²")
    fig.suptitle(f"Per-question g-mean² vs unhinted task accuracy — "
                 f"{model_name}\n(entering questions pooled over 3 hint "
                 f"types; x-jittered for display; Pearson r computed on "
                 f"raw values)", fontsize=11)

    out = args.out
    if out is None:
        # RO1 bespoke analysis plots live under extra-plots/, not the main plots/ tree
        plots_dir = ((root / config["paths"]["plots_dir"]).parent
                     / "extra-plots" / "task-performance-vs-monitorability")
        if args.data != "test":
            plots_dir = plots_dir / args.data
        plots_dir.mkdir(parents=True, exist_ok=True)
        out = plots_dir / f"gmean2_vs_acc_scatter_{model_name}.png"
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
