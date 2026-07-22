#!/usr/bin/env python3
"""
Step 22: Monitorability-by-response-length curve for ONE model, pooled over
every (dataset, hint type) slice.

Replicates the "Monitorability by CoT length" plot style from the GDM
hint-taking paper, but for a single model. Every question that ENTERS the
g-mean^2 metric (Level-1 pass AND gmean2_i defined) across all dataset/hint
slices contributes one point: (mean token length of its N hinted responses,
its per-instance gmean2_i). `mean_cot_tokens_hinted` is the FULL completion
length (thinking trace + final answer) averaged over the valid hinted runs.

Questions are then bucketed by that length into fixed-width bins
((k-1)*W, k*W], W = 500 tokens by default; each bin is drawn as one dot at the
bin's upper edge (0.5k, 1.0k, ...) at the bin-mean g-mean^2, connected by a
single line. Questions longer than --tail-from tokens (default 13000) are
POOLED into a single open-ended "13k+" bin drawn one bin-width past the last
regular dot — the per-0.5k tail bins hold only a handful of questions each
and their sampling noise dominates the right of the plot (same tail-merging
convention as the GDM hint-taking paper's figure). A percentile bootstrap
95% CI per bin is always printed to stdout and drawn as error bars with --ci
(off by default: even merged, the sparse-tail intervals stretch the y-axis).

Usage:
    python scripts/22_length_gmean2_curve.py                       # base model
    python scripts/22_length_gmean2_curve.py --model <hf-id>

    Options:
      --config     path to config YAML (default: config/mvp_config.yaml)
      --data       split: test (default) | eval
      --model      model/adapter HF id (default: Qwen/Qwen3-4B)
      --bin-width  bin width in tokens (default: 500)
      --min-length exclude questions whose mean hinted length is <= this
                   (default: 500, so the curve starts at the 1k dot — the
                   sub-500-token bin holds only ~11 questions; 0 disables)
      --tail-from  pool questions longer than this into one open-ended final
                   bin (default: 13000; 0 disables merging)
      --min-n      drop bins with fewer questions than this (default: 1 = keep all)
      --ci         draw the per-bin bootstrap 95% CI as error bars
      --resamples  bootstrap replicates per bin CI (default: 2000)
      --seed       bootstrap seed (default: 88)
      --out        output PNG path (default: {plots_dir}/gmean2_by_length_{model}.png)

Input (per dataset / hint type, produced by 04_compute_gmean2.py):
    {analysis_dir}/{model}/{dataset}/{hint}/results_per_instance.csv

Output:
    {plots_dir}/gmean2_by_length_{model}.png
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
import yaml

from src.dataset import SPLITS, model_dir_name, resolve_paths
from src.hints import HINT_TYPES

DEFAULT_MODEL = "Qwen/Qwen3-4B"
DARK_BLUE = "#1B3A6B"


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


def bin_ci(values, n_resamples, rng):
    """Percentile bootstrap 95% CI of the mean; None for singleton bins."""
    if len(values) < 2:
        return None
    arr = np.array(values)
    idx = rng.integers(0, len(arr), size=(n_resamples, len(arr)))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    parser = argparse.ArgumentParser(
        description="Monitorability-by-response-length curve, pooled over all slices.")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--data", default="test", choices=SPLITS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--bin-width", type=int, default=500)
    parser.add_argument("--min-length", type=int, default=500)
    parser.add_argument("--tail-from", type=int, default=13000)
    parser.add_argument("--min-n", type=int, default=1)
    parser.add_argument("--ci", action="store_true")
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
    width = args.bin_width

    points, missing = [], []
    for dataset in config.get("datasets", {}):
        for hint in HINT_TYPES:
            paths = resolve_paths(root, config, dataset, model_name, hint,
                                  split=args.data)
            csv_path = paths["analysis_dir"] / "results_per_instance.csv"
            if not csv_path.exists():
                missing.append(f"{dataset}/{hint}")
                continue
            points.extend(load_entering(csv_path))

    if missing:
        print("WARNING: missing slices skipped: " + ", ".join(missing))
    if not points:
        print("ERROR: no entering instances found — nothing to plot.")
        sys.exit(1)

    if args.min_length > 0:
        kept = [p for p in points if p[0] > args.min_length]
        print(f"Excluded {len(points) - len(kept)} question(s) with mean hinted "
              f"length <= {args.min_length} tokens")
        points = kept
    if not points:
        print("ERROR: min-length filter removed every instance — nothing to plot.")
        sys.exit(1)

    lengths = np.array([p[0] for p in points])
    print(f"Pooled questions entering g-mean^2: {len(points)} "
          f"(length min {lengths.min():.0f}, median {np.median(lengths):.0f}, "
          f"max {lengths.max():.0f} tokens)")

    # Bin ((k-1)*W, k*W] -> dot at the upper edge k*W. Everything past
    # --tail-from pools into one open-ended bin one bin-width further right.
    tail_from = args.tail_from if args.tail_from > 0 else None
    bins, tail = defaultdict(list), []
    for L, g in points:
        if tail_from is not None and L > tail_from:
            tail.append(g)
        else:
            bins[int(np.ceil(L / width)) * width].append(g)

    rng = np.random.default_rng(args.seed)
    xs, ys, err_lo, err_hi, tick_labels = [], [], [], [], []
    print(f"\n{'bin (tok)':>15} | {'x':>6} | {'n':>5} | {'g-mean2':>8} | 95% CI")
    print("-" * 62)
    rows = [(edge, f"{edge - width:>6}-{edge:<8}", f"{edge / 1000:g}k", bins[edge])
            for edge in sorted(bins)]
    if tail:
        rows.append((max(bins) + width if bins else width,
                     f"{tail_from:>6}+{'':<8}", f"{tail_from / 1000:g}k+", tail))
    for x, span, label, vals in rows:
        mean = float(np.mean(vals))
        ci = bin_ci(vals, args.resamples, rng)
        ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "--"
        dropped = " (dropped)" if len(vals) < args.min_n else ""
        print(f"{span} | {x:>6} | {len(vals):>5} | {mean:>8.3f} | {ci_str}{dropped}")
        if len(vals) < args.min_n:
            continue
        xs.append(x)
        ys.append(mean)
        err_lo.append(mean - ci[0] if ci else 0.0)
        err_hi.append(ci[1] - mean if ci else 0.0)
        tick_labels.append(label)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_axisbelow(True)
    ax.grid(True, color="#DDDDDD", linewidth=0.8)

    if args.ci:
        ax.errorbar(xs, ys, yerr=[err_lo, err_hi], color=DARK_BLUE,
                    linewidth=2.0, marker="o", markersize=6,
                    ecolor=DARK_BLUE, elinewidth=1.2, capsize=3, capthick=1.2,
                    zorder=3)
        lo_pts = [y - e for y, e in zip(ys, err_lo)]
        hi_pts = [y + e for y, e in zip(ys, err_hi)]
    else:
        ax.plot(xs, ys, color=DARK_BLUE, linewidth=2.0, marker="o",
                markersize=6, zorder=3)
        lo_pts, hi_pts = ys, ys

    ax.set_xticks(xs)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlim(float(lengths.min()), xs[-1] + 0.4 * width)
    ax.set_ylim(min(lo_pts) - 0.02, max(hi_pts) + 0.02)

    ax.set_xlabel("Response tokens (CoT + final answer)")
    ax.set_ylabel("Monitorability (g-mean²)")
    ax.set_title(f"Monitorability by response length — {model_name}\n"
                 f"(pooled over all datasets and hint types)")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#999999")

    out = args.out
    if out is None:
        # RO1 bespoke analysis plots live under extra-plots/, not the main plots/ tree
        plots_dir = ((root / config["paths"]["plots_dir"]).parent
                     / "extra-plots" / "cot-length-vs-monitorability")
        if args.data != "test":
            plots_dir = plots_dir / args.data
        plots_dir.mkdir(parents=True, exist_ok=True)
        out = plots_dir / f"gmean2_by_length_{model_name}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
