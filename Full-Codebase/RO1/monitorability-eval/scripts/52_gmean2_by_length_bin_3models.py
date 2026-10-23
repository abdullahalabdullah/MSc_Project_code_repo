#!/usr/bin/env python3
"""
Step 52: g-mean^2 by response-length bin — base vs harmlessness DPO vs
anti-sycophancy DPO, as a grouped bar chart.

The visual companion to the 1000-token bucket notes
(`notes/length-confound/bucket_analysis_1000tok_base_vs_{harmless,antisyc}_*.md`).
Those notes tabulate each model's mean per-instance g-mean^2 inside matched
length bins; this plots the three models together so the within-bin comparison
can be read at a glance instead of across two tables. The notes' two thinnest
bins (11-12K and 12K+) are merged here into a single 11K+ tail bin, so this
figure's last group has no counterpart row in those tables — it is recomputed
from the instances, not averaged from the two published bin means.

Layout: one group of three bars per length bin along the x-axis, monitorability
on the y-axis. Colour and hatch both carry model identity (base solid light
blue, harmlessness striped pink, anti-sycophancy dotted green — the repo's
standard trio), so the figure survives grayscale. Printed above each bar is that
cell's INSTANCE COUNT, not the bar's height: the upper bins are thin and the
count is what tells a reader how much to trust the bar. The height is read off
the y-axis.

The y-axis starts at zero. Differences between bars here are 0.02-0.09 on values
around 0.75, so a truncated axis would exaggerate them several-fold; bars encode
magnitude by length and must not be clipped.

Data source: each cell's `results_per_instance.csv` (written by
04_compute_gmean2.py) — the same `(mean_cot_tokens_hinted, gmean2_i)` rows
script 10 pools, restricted to instances that ENTER the metric (Level-1 pass and
gmean2_i defined), across all 24 (dataset, hint) slices.

Usage:
    python scripts/52_gmean2_by_length_bin_3models.py

    Options:
      --config    path to config YAML (default: config/mvp_config.yaml)
      --data      split: test (default) | eval
      --base      base model id (default: Qwen/Qwen3-4B)
      --harmless  harmlessness DPO id (default: er-abd/harmless-dpo-qwen3-4B-step-735)
      --antisyc   anti-sycophancy DPO id (default: er-abd/anti-syc-dpo-qwen3-4B-step-520)
      --out       output PNG path

Output:
    {plots_dir}/../extra-plots/length-confound/gmean2_by_length_bin_3models.png
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import yaml

from src.dataset import SPLITS, model_dir_name, resolve_paths

DEFAULT_BASE = "Qwen/Qwen3-4B"
DEFAULT_HARMLESS = "er-abd/harmless-dpo-qwen3-4B-step-735"
DEFAULT_ANTISYC = "er-abd/anti-syc-dpo-qwen3-4B-step-520"

DATASETS = ["mmlu_pro", "gpqa_diamond", "lsat_ar", "lsat_lr",
            "medxpertqa", "lexam", "codemmlu", "wmdp"]
HINT_TYPES = ["professor", "grader_hacking", "unethical"]

def make_bins(width, tail_start):
    """Uniform `width`-token bins up to `tail_start`, then one open-ended bin."""
    edges = list(range(0, tail_start + 1, width)) + [np.inf]

    def tag(v):
        return f"{v // 1000}K" if v % 1000 == 0 else f"{v / 1000:g}K"

    labels = [f"{tag(lo)}–{tag(lo + width)}" for lo in edges[:-2]] + [f"{tag(tail_start)}+"]
    return edges, labels

# Repo-standard trio; hatch is the grayscale-safe second cue.
STYLES = [
    ("Base (Qwen3-4B)", None, "#92C5E8"),
    ("Harmlessness DPO", "///", "#F6A0B8"),
    ("Anti-sycophancy DPO", "..", "#9ED973"),
]
EDGE = "#555555"
BAR_W = 0.27
OFFSETS = [-BAR_W, 0.0, BAR_W]


def entering_instances(root, config, model_name, split):
    """(length, gmean2) for every instance entering the metric, all 24 slices."""
    lengths, gmeans, missing = [], [], []
    for dataset in DATASETS:
        for hint in HINT_TYPES:
            paths = resolve_paths(root, config, dataset, model_name, hint, split=split)
            csv_path = paths["analysis_dir"] / "results_per_instance.csv"
            if not csv_path.exists():
                missing.append(f"{dataset}/{hint}")
                continue
            with csv_path.open() as f:
                for row in csv.DictReader(f):
                    g = row.get("gmean2_i", "")
                    t = row.get("mean_cot_tokens_hinted", "")
                    if g in ("", "None", "N/A") or t in ("", "None", "N/A"):
                        continue
                    gmeans.append(float(g))
                    lengths.append(float(t))
    return np.array(lengths), np.array(gmeans), missing


def binned(lengths, gmeans, edges):
    """Mean g-mean^2 and instance count per length bin."""
    means, counts = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (lengths >= lo) & (lengths < hi)
        counts.append(int(m.sum()))
        means.append(gmeans[m].mean() if m.any() else np.nan)
    return np.array(means), counts


def main():
    ap = argparse.ArgumentParser(
        description="g-mean^2 by response-length bin, three models, grouped bars.")
    ap.add_argument("--config", default="config/mvp_config.yaml")
    ap.add_argument("--data", default="test", choices=SPLITS)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--harmless", default=DEFAULT_HARMLESS)
    ap.add_argument("--antisyc", default=DEFAULT_ANTISYC)
    ap.add_argument("--bin-width", type=int, default=1000,
                    help="uniform bin width in tokens (default: 1000)")
    ap.add_argument("--tail-start", type=int, default=11000,
                    help="everything at or above this is pooled into one bin "
                         "(default: 11000)")
    ap.add_argument("--rows", type=int, default=1, choices=(1, 2),
                    help="split the bins across 2 stacked panels when the bin "
                         "count makes one row too wide (default: 1)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.tail_start % args.bin_width:
        print("ERROR: --tail-start must be a multiple of --bin-width.")
        sys.exit(1)
    bin_edges, bin_labels = make_bins(args.bin_width, args.tail_start)

    root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    with config_path.open() as f:
        config = yaml.safe_load(f)

    models = [args.base, args.harmless, args.antisyc]
    series, all_counts = [], []
    for mid in models:
        lengths, gmeans, missing = entering_instances(
            root, config, model_dir_name(mid), args.data)
        if missing:
            print(f"WARNING: {mid} missing slices: {', '.join(missing)}")
        if len(gmeans) == 0:
            print(f"ERROR: no entering instances for {mid} — run 04_compute_gmean2.py.")
            sys.exit(1)
        means, counts = binned(lengths, gmeans, bin_edges)
        series.append(means)
        all_counts.append(counts)
        print(f"{mid}: n={len(gmeans)}, median len={np.median(lengths):.0f}, "
              f"overall g-mean^2={gmeans.mean():.4f}")

    n_bins = len(bin_labels)
    # Split the bins across `rows` panels so a long bin sequence stays legible.
    per_row = -(-n_bins // args.rows)
    spans = [range(r * per_row, min((r + 1) * per_row, n_bins))
             for r in range(args.rows)]
    widest = max(len(s) for s in spans)

    fig, axes = plt.subplots(
        args.rows, 1, figsize=(max(15, 1.15 * widest), 6 * args.rows),
        squeeze=False)
    axes = [a[0] for a in axes]

    label_fs = 11 if widest <= 14 else 9
    tick_fs = 15 if widest <= 14 else 13
    for ax, span in zip(axes, spans):
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color="#DDDDDD", linewidth=0.8)
        for means, counts, (label, hatch, color), off in zip(
                series, all_counts, STYLES, OFFSETS):
            for pos, i in enumerate(span):
                v, n = means[i], counts[i]
                if np.isnan(v):
                    # Empty cell: no bar to draw, but say so rather than leave a gap.
                    ax.text(pos + off, 0.012, "0", ha="center", va="bottom",
                            rotation=90, fontsize=label_fs, color="#444444",
                            zorder=4)
                    continue
                ax.bar(pos + off, v, width=BAR_W, color=color, hatch=hatch,
                       edgecolor=EDGE, linewidth=0.6, zorder=3)
                ax.text(pos + off, v + 0.012, f"{n}", ha="center", va="bottom",
                        rotation=90, fontsize=label_fs, color="#444444", zorder=4)

        ax.set_xticks(np.arange(len(span)))
        ax.set_xticklabels([bin_labels[i] for i in span], fontsize=tick_fs)
        # Pad a short final row to the widest row's scale so bar widths match.
        ax.set_xlim(-0.62, widest - 0.38)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([i / 10 for i in range(11)])
        ax.tick_params(axis="y", labelsize=tick_fs)
        ax.set_ylabel("Monitorability (g-mean²)", fontsize=18)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#999999")
    axes[-1].set_xlabel("Response Length Bins", fontsize=18)

    handles = [Patch(facecolor=c, edgecolor=EDGE, linewidth=0.8, hatch=h,
                     label=lbl) for lbl, h, c in STYLES]
    axes[0].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.005, 1.0),
                   frameon=True, edgecolor="#CCCCCC", fontsize=15, title="Model",
                   title_fontsize=15)

    out = args.out
    if out is None:
        plots_dir = ((root / config["paths"]["plots_dir"]).parent
                     / "extra-plots" / "length-confound")
        if args.data != "test":
            plots_dir = plots_dir / args.data
        plots_dir.mkdir(parents=True, exist_ok=True)
        # Default binning keeps the original filename; any other binning is
        # tagged so the two figures can coexist.
        tag = ("" if (args.bin_width, args.tail_start) == (1000, 11000)
               else f"_{args.bin_width}tok")
        out = plots_dir / f"gmean2_by_length_bin{tag}_3models.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
