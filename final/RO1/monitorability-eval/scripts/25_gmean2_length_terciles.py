#!/usr/bin/env python3
"""
Step 25: Within-dataset length-tercile dot plot of g-mean^2 for ONE model.

The methodologically clean view of "monitorability vs response length per
dataset" given ~340 entering questions per dataset: instead of fixed token
bins (5-20 questions each at dataset level — pure noise), each dataset's
entering questions are split into ITS OWN shortest / middle / longest third
by mean hinted response length (full completion: thinking trace + final
answer, averaged over the N valid hinted runs). Each tercile dot is a
conditional mean over ~70-140 questions with a percentile-bootstrap 95% CI,
so a flat trio is evidence of no length effect rather than an artifact of
sparse bins, and dataset-difficulty composition is controlled by construction
(terciles are within-dataset).

x = the 8 datasets (domain-table order); per dataset three dots in
light -> dark shades of one blue (shortest -> longest), with CI whiskers.
Stdout prints, per dataset, each tercile's n / length range / median length /
g-mean^2 / CI, plus the per-dataset Spearman rho between length and gmean2_i
(with p-value) for use in the figure caption.

Usage:
    python scripts/25_gmean2_length_terciles.py                    # base model
    python scripts/25_gmean2_length_terciles.py --model <hf-id>

    Options:
      --config     path to config YAML (default: config/mvp_config.yaml)
      --data       split: test (default) | eval
      --model      model/adapter HF id (default: Qwen/Qwen3-4B)
      --n-bins     equal-count length bins per dataset (default: 3 =
                   terciles; e.g. 5 = quintiles/0-20%%-...-100%%; the output
                   filename carries the bin count when it is not 3)
      --resamples  bootstrap replicates per bin CI (default: 2000)
      --seed       bootstrap seed (default: 88)
      --out        output PNG path
                   (default: {plots_dir}/gmean2_length_terciles_{model}.png)

Input (per dataset / hint type, produced by 04_compute_gmean2.py):
    {analysis_dir}/{model}/{dataset}/{hint}/results_per_instance.csv

Output:
    {plots_dir}/gmean2_length_terciles_{model}.png
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy import stats
import yaml

from src.dataset import SPLITS, model_dir_name, resolve_paths
from src.hints import HINT_TYPES

DEFAULT_MODEL = "Qwen/Qwen3-4B"

# Same presentation order as the main-results tables / steps 16-18.
DATASETS = ["mmlu_pro", "gpqa_diamond", "lsat_ar", "lsat_lr",
            "medxpertqa", "lexam", "codemmlu", "wmdp"]

# Shortest -> longest: light -> dark steps of one blue, so the length ordering
# survives grayscale and CVD (lightness carries the identity, not hue).
TERCILE_LABELS = ["Shortest third", "Middle third", "Longest third"]


def bin_specs(n_bins):
    """[(label, color, x_offset)] for n_bins equal-count length bins."""
    if n_bins == 3:
        labels = TERCILE_LABELS
    else:
        labels = [f"{100 * i // n_bins}-{100 * (i + 1) // n_bins}% (short"
                  f"est)" if i == 0 else
                  f"{100 * i // n_bins}-{100 * (i + 1) // n_bins}% (long"
                  f"est)" if i == n_bins - 1 else
                  f"{100 * i // n_bins}-{100 * (i + 1) // n_bins}%"
                  for i in range(n_bins)]
    cmap = matplotlib.colormaps["Blues"]
    colors = [matplotlib.colors.to_hex(cmap(0.35 + 0.6 * i / (n_bins - 1)))
              for i in range(n_bins)]
    span = 0.62
    offsets = [span * (i / (n_bins - 1) - 0.5) for i in range(n_bins)]
    return list(zip(labels, colors, offsets))


def load_entering(csv_path: Path):
    """(mean_hinted_tokens, gmean2_i) for questions that enter g-mean^2."""
    out = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if row.get("level1_pass") != "True":
                continue
            g, L = row.get("gmean2_i", ""), row.get("mean_cot_tokens_hinted", "")
            if g in ("", "None", "N/A") or L in ("", "None", "N/A"):
                continue
            out.append((float(L), float(g)))
    return out


def boot_ci(values, n_resamples, rng):
    arr = np.array(values)
    idx = rng.integers(0, len(arr), size=(n_resamples, len(arr)))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    parser = argparse.ArgumentParser(
        description="Within-dataset length-tercile dot plot of g-mean^2.")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--data", default="test", choices=SPLITS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--n-bins", type=int, default=3)
    parser.add_argument("--resamples", type=int, default=2000)
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
    n_bins = args.n_bins
    if n_bins < 2:
        print("ERROR: --n-bins must be >= 2.")
        sys.exit(1)
    specs = bin_specs(n_bins)
    rng = np.random.default_rng(args.seed)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#DDDDDD", linewidth=0.8)

    all_pts, missing, plotted_datasets = [], [], []
    for d_idx, dataset in enumerate(DATASETS):
        points = []
        for hint in HINT_TYPES:
            paths = resolve_paths(root, config, dataset, model_name, hint,
                                  split=args.data)
            csv_path = paths["analysis_dir"] / "results_per_instance.csv"
            if not csv_path.exists():
                missing.append(f"{dataset}/{hint}")
                continue
            points.extend(load_entering(csv_path))
        if len(points) < 3 * n_bins:  # need a few questions per bin
            print(f"\n{dataset}: only {len(points)} entering questions — skipped")
            continue

        L = np.array([p[0] for p in points])
        g = np.array([p[1] for p in points])
        rho, pval = stats.spearmanr(L, g)
        print(f"\n{dataset} — {len(points)} entering questions, "
              f"Spearman rho(length, gmean2) = {rho:+.3f} (p = {pval:.3g})")
        print(f"{'bin':>15} | {'n':>4} | {'len range':>13} | {'median':>6} "
              f"| {'g-mean2':>8} | 95% CI")
        print("-" * 76)

        # Equal-count bins of THIS dataset's length distribution.
        qs = np.quantile(L, [i / n_bins for i in range(1, n_bins)])
        edges = [-np.inf, *qs, np.inf]
        masks = [(L > edges[i]) & (L <= edges[i + 1]) for i in range(n_bins)]
        for (label, color, off), mask in zip(specs, masks):
            vals = g[mask]
            mean = float(vals.mean())
            lo, hi = boot_ci(vals, args.resamples, rng)
            print(f"{label:>15} | {mask.sum():>4} | "
                  f"{L[mask].min():>5.0f}-{L[mask].max():<7.0f} | "
                  f"{np.median(L[mask]):>6.0f} | {mean:>8.3f} | "
                  f"[{lo:.3f}, {hi:.3f}]")
            x = d_idx + off
            ax.errorbar(x, mean, yerr=[[mean - lo], [hi - mean]], fmt="o",
                        color=color, markersize=7, markeredgecolor="#333333",
                        markeredgewidth=0.5, ecolor=color, elinewidth=1.4,
                        capsize=3, capthick=1.4, zorder=3)
            all_pts.extend([lo, hi])
        plotted_datasets.append((d_idx, dataset))

    if missing:
        print("\nWARNING: missing slices skipped: " + ", ".join(missing))
    if not plotted_datasets:
        print("ERROR: no dataset had enough entering questions — nothing to plot.")
        sys.exit(1)

    ax.set_xticks([i for i, _ in plotted_datasets])
    ax.set_xticklabels([ds for _, ds in plotted_datasets])
    ax.set_xlim(-0.55, len(DATASETS) - 0.45)
    ax.set_ylim(min(all_pts) - 0.02, max(all_pts) + 0.02)
    # Light separators between dataset groups.
    for i in range(len(DATASETS) - 1):
        ax.axvline(i + 0.5, color="#EEEEEE", linewidth=0.8, zorder=1)

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Monitorability (g-mean²)")
    bin_word = "tercile" if n_bins == 3 else f"{n_bins}-quantile"
    split_desc = ("its own shortest / middle / longest third" if n_bins == 3
                  else f"its own {n_bins} equal-count length bins")
    ax.set_title(f"Monitorability by within-dataset response-length {bin_word}"
                 f" — {model_name}\n(each dataset split into {split_desc}; "
                 f"bootstrap 95% CI)")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#999999")

    handles = [Line2D([], [], marker="o", linestyle="none", color=color,
                      markeredgecolor="#333333", markeredgewidth=0.5,
                      markersize=7, label=label) for label, color, _ in specs]
    ax.legend(handles=handles, title="Response length\n(within dataset)",
              loc="upper left", bbox_to_anchor=(1.005, 1.0), frameon=True,
              edgecolor="#CCCCCC", fontsize=9)

    out = args.out
    if out is None:
        # RO1 bespoke analysis plots live under extra-plots/, not the main plots/ tree
        plots_dir = ((root / config["paths"]["plots_dir"]).parent
                     / "extra-plots" / "cot-length-vs-monitorability")
        if args.data != "test":
            plots_dir = plots_dir / args.data
        plots_dir.mkdir(parents=True, exist_ok=True)
        stem = ("gmean2_length_terciles" if n_bins == 3
                else f"gmean2_length_q{n_bins}")
        out = plots_dir / f"{stem}_{model_name}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
