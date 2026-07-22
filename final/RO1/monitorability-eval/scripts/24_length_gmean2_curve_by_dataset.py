#!/usr/bin/env python3
"""
Step 24: Monitorability-by-response-length curves for ONE model, one line per
DATASET (each pooled over its three hint types).

Same construction as 22_length_gmean2_curve.py — every g-mean^2-entering
question (Level-1 pass AND gmean2_i defined) contributes (mean hinted
full-completion tokens over its N valid runs, gmean2_i); questions bucketed
into fixed-width ((k-1)*W, k*W] bins with the dot at the bin's upper edge —
but binned per dataset instead of pooled, giving 8 colored lines.

Per-dataset bins are ~8x sparser than the pooled ones, so bins with fewer
than --min-n questions (default 5) are dropped from a dataset's line, and
each dataset's questions longer than --tail-from pool into its own
open-ended "13k+" dot at a fixed x (tail_from + W) shared by all datasets.
A line BREAKS (gap, no segment) wherever a bin was dropped or empty, so a
segment never bridges a region with no data.
Per-bin bootstrap 95% CIs are printed to stdout only — error bars on 8
overlapping lines are unreadable.

Colors follow a CVD-validated 8-slot categorical palette in fixed dataset
order (the domain-table order used by the summary tables and step 16).

Usage:
    python scripts/24_length_gmean2_curve_by_dataset.py            # base model
    python scripts/24_length_gmean2_curve_by_dataset.py --model <hf-id>

    Options:
      --config     path to config YAML (default: config/mvp_config.yaml)
      --data       split: test (default) | eval
      --model      model/adapter HF id (default: Qwen/Qwen3-4B)
      --bin-width  bin width in tokens (default: 500)
      --min-length exclude questions with mean hinted length <= this
                   (default: 500, matching step 22; 0 disables)
      --tail-from  pool questions longer than this into one open-ended final
                   bin per dataset (default: 13000; 0 disables merging)
      --min-n      drop per-dataset bins with fewer questions than this
                   (default: 5)
      --resamples  bootstrap replicates per bin CI (default: 2000)
      --seed       bootstrap seed (default: 88)
      --out        output PNG path
                   (default: {plots_dir}/gmean2_by_length_by_dataset_{model}.png)

Input (per dataset / hint type, produced by 04_compute_gmean2.py):
    {analysis_dir}/{model}/{dataset}/{hint}/results_per_instance.csv

Output:
    {plots_dir}/gmean2_by_length_by_dataset_{model}.png
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

# Same presentation order as the main-results tables / step 16; colors are the
# 8-slot categorical palette validated for adjacent-pair CVD separation on a
# light surface (worst adjacent CVD dE 9.1, normal-vision dE 19.6).
DATASET_COLORS = [
    ("mmlu_pro", "#2a78d6"),      # blue
    ("gpqa_diamond", "#eb6834"),  # orange
    ("lsat_ar", "#1baf7a"),       # aqua
    ("lsat_lr", "#eda100"),       # yellow
    ("medxpertqa", "#e87ba4"),    # magenta
    ("lexam", "#008300"),         # green
    ("codemmlu", "#4a3aa7"),      # violet
    ("wmdp", "#e34948"),          # red
]


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


def bin_dataset(points, width, tail_from, min_length):
    """Return {dot_x: [gmean2_i, ...]} plus the tail values (or None)."""
    bins, tail = defaultdict(list), []
    for L, g in points:
        if min_length > 0 and L <= min_length:
            continue
        if tail_from is not None and L > tail_from:
            tail.append(g)
        else:
            bins[int(np.ceil(L / width)) * width].append(g)
    return bins, tail


def main():
    parser = argparse.ArgumentParser(
        description="Per-dataset monitorability-by-response-length curves.")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--data", default="test", choices=SPLITS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--bin-width", type=int, default=500)
    parser.add_argument("--min-length", type=int, default=500)
    parser.add_argument("--tail-from", type=int, default=13000)
    parser.add_argument("--min-n", type=int, default=5)
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
    tail_from = args.tail_from if args.tail_from > 0 else None
    tail_x = (tail_from + width) if tail_from is not None else None

    config_datasets = set(config.get("datasets", {}))
    unknown = config_datasets - {d for d, _ in DATASET_COLORS}
    if unknown:
        print(f"WARNING: dataset(s) {sorted(unknown)} in config have no color "
              f"slot — extend DATASET_COLORS to include them.")

    rng = np.random.default_rng(args.seed)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_axisbelow(True)
    ax.grid(True, color="#DDDDDD", linewidth=0.8)

    all_x, all_y, missing, plotted = set(), [], [], 0
    for dataset, color in DATASET_COLORS:
        if dataset not in config_datasets:
            continue
        points = []
        for hint in HINT_TYPES:
            paths = resolve_paths(root, config, dataset, model_name, hint,
                                  split=args.data)
            csv_path = paths["analysis_dir"] / "results_per_instance.csv"
            if not csv_path.exists():
                missing.append(f"{dataset}/{hint}")
                continue
            points.extend(load_entering(csv_path))
        if not points:
            continue

        bins, tail = bin_dataset(points, width, tail_from, args.min_length)
        rows = [(edge, f"{edge / 1000:g}k", bins[edge]) for edge in sorted(bins)]
        if tail:
            rows.append((tail_x, f"{tail_from / 1000:g}k+", tail))

        xs, ys = [], []
        print(f"\n{dataset} ({len(points)} entering questions)")
        print(f"{'dot':>7} | {'n':>5} | {'g-mean2':>8} | 95% CI")
        print("-" * 48)
        for x, label, vals in rows:
            mean = float(np.mean(vals))
            ci = bin_ci(vals, args.resamples, rng)
            ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "--"
            dropped = " (dropped)" if len(vals) < args.min_n else ""
            print(f"{label:>7} | {len(vals):>5} | {mean:>8.3f} | {ci_str}{dropped}")
            if len(vals) < args.min_n:
                continue
            xs.append(x)
            ys.append(mean)
        if not xs:
            print(f"    every bin below --min-n {args.min_n} — line skipped")
            continue

        # Break the line across dropped/empty bins: consecutive dots are one
        # bin-width apart; anything wider gets a NaN gap instead of a segment.
        gx, gy = [xs[0]], [ys[0]]
        for (x0, _), (x1, y1) in zip(zip(xs, ys), list(zip(xs, ys))[1:]):
            if x1 - x0 > width:
                gx.append(np.nan)
                gy.append(np.nan)
            gx.append(x1)
            gy.append(y1)
        ax.plot(gx, gy, color=color, linewidth=1.8, marker="o", markersize=5,
                label=dataset, zorder=3)
        all_x.update(xs)
        all_y.extend(ys)
        plotted += 1

    if missing:
        print("\nWARNING: missing slices skipped: " + ", ".join(missing))
    if not plotted:
        print("ERROR: no dataset produced a line — nothing to plot.")
        sys.exit(1)

    ticks = sorted(all_x)
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [f"{tail_from / 1000:g}k+" if tail_x is not None and t == tail_x
         else f"{t / 1000:g}k" for t in ticks],
        rotation=45, ha="right", fontsize=8)
    ax.set_xlim(ticks[0] - 0.6 * width, ticks[-1] + 0.6 * width)
    ax.set_ylim(min(all_y) - 0.02, max(all_y) + 0.02)

    ax.set_xlabel("Response tokens (CoT + final answer)")
    ax.set_ylabel("Monitorability (g-mean²)")
    ax.set_title(f"Monitorability by response length per dataset — {model_name}\n"
                 f"(each line pooled over all hint types)")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#999999")
    ax.legend(title="Dataset", loc="upper left", bbox_to_anchor=(1.005, 1.0),
              frameon=True, edgecolor="#CCCCCC", fontsize=9)

    out = args.out
    if out is None:
        # RO1 bespoke analysis plots live under extra-plots/, not the main plots/ tree
        plots_dir = ((root / config["paths"]["plots_dir"]).parent
                     / "extra-plots" / "cot-length-vs-monitorability")
        if args.data != "test":
            plots_dir = plots_dir / args.data
        plots_dir.mkdir(parents=True, exist_ok=True)
        out = plots_dir / f"gmean2_by_length_by_dataset_{model_name}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
