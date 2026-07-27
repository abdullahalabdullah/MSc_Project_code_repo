#!/usr/bin/env python3
"""
Step 28: Forest plot of the per-dataset accuracy-threshold effect on
g-mean^2 for ONE model.

The pooled analysis (step 27) shows monitorability is FLAT across partial
unhinted-accuracy levels and elevated only for questions the model answers
correctly in all N unhinted runs — a threshold, not a graded trend. This
figure reduces each dataset to that single contrast:

    Delta_d = mean gmean2_i (always correct, acc=1)
            - mean gmean2_i (not always correct, acc<1)

over the dataset's entering questions (hint types pooled), with a
percentile-bootstrap 95% CI (independent resampling within the two
groups). One row per dataset (domain-table order, top to bottom) plus a
POOLED row over all questions; a dashed zero line makes the question "is
the effect consistent across datasets?" read as "do the intervals sit
right of zero?". Stdout prints per-row group sizes / means / Delta / CI.

--placebo adds a second, grey dot per row: the PLACEBO contrast
    mean gmean2_i (partial, 1-7/8) - mean gmean2_i (never correct, 0/8),
which TESTS the dichotomy rather than assuming it. If monitorability rose
with accuracy generally, this contrast would also be positive; instead it
straddles zero on 7/8 datasets and pooled (-0.005 [-0.042, +0.029]), so
collapsing 0/8..7/8 into one group loses no signal. Use this variant when
a reader may object that "7/8 is not the same as 0/8".

Usage:
    python scripts/28_gmean2_acc_delta_forest.py                    # base
    python scripts/28_gmean2_acc_delta_forest.py --model <hf-id>

    Options:
      --config     path to config YAML (default: config/mvp_config.yaml)
      --data       split: test (default) | eval
      --model      model/adapter HF id (default: Qwen/Qwen3-4B)
      --min-n      min questions in EACH group to draw a dataset row
                   (default: 10)
      --placebo    also draw the partial(1-7/8) - never(0/8) contrast per
                   row, and write …_placebo_{model}.png instead
      --resamples  bootstrap replicates per CI (default: 5000)
      --seed       bootstrap seed (default: 88)
      --out        output PNG path
                   (default: {plots_dir}/gmean2_acc_delta_forest_{model}.png)

Input (per dataset / hint type, produced by 04_compute_gmean2.py):
    {analysis_dir}/{model}/{dataset}/{hint}/results_per_instance.csv

Output:
    {plots_dir}/gmean2_acc_delta_forest_{model}.png
    {plots_dir}/gmean2_acc_delta_forest_placebo_{model}.png  (--placebo)
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
import yaml

from src.dataset import SPLITS, model_dir_name, resolve_paths
from src.hints import HINT_TYPES

DEFAULT_MODEL = "Qwen/Qwen3-4B"
N_RUNS = 8

DATASETS = ["mmlu_pro", "gpqa_diamond", "lsat_ar", "lsat_lr",
            "medxpertqa", "lexam", "codemmlu", "wmdp"]

DARK_BLUE = "#1B3A6B"
PLACEBO_COLOR = "#9AA7B4"


def boot_diff_ci(hi_vals, lo_vals, n_resamples, rng):
    h, l = np.asarray(hi_vals), np.asarray(lo_vals)
    ih = rng.integers(0, len(h), size=(n_resamples, len(h)))
    il = rng.integers(0, len(l), size=(n_resamples, len(l)))
    diffs = h[ih].mean(axis=1) - l[il].mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    parser = argparse.ArgumentParser(
        description="Forest plot of per-dataset Delta g-mean^2 "
                    "(always-correct minus rest).")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--data", default="test", choices=SPLITS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--min-n", type=int, default=10)
    parser.add_argument("--placebo", action="store_true",
                        help="also draw the PLACEBO contrast per row "
                             "(partial 1-7/8 minus never 0/8), which tests "
                             "the dichotomy instead of assuming it")
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

    per_ds, missing = {ds: ([], []) for ds in DATASETS}, []
    for dataset in DATASETS:
        for hint in HINT_TYPES:
            paths = resolve_paths(root, config, dataset, model_name, hint,
                                  split=args.data)
            csv_path = paths["analysis_dir"] / "results_per_instance.csv"
            if not csv_path.exists():
                missing.append(f"{dataset}/{hint}")
                continue
            with csv_path.open() as f:
                for row in csv.DictReader(f):
                    if row.get("level1_pass") != "True":
                        continue
                    g = row.get("gmean2_i", "")
                    a = row.get("acc_u_i", "")
                    if g in ("", "None", "N/A") or a in ("", "None", "N/A"):
                        continue
                    per_ds[dataset][0].append(float(a))
                    per_ds[dataset][1].append(float(g))
    if missing:
        print("WARNING: missing slices skipped: " + ", ".join(missing))

    def contrasts(A, G, label):
        """Main (always - rest) and placebo (partial - never) contrasts."""
        lev = np.clip(np.round(A * N_RUNS) / N_RUNS, 0, 1)
        always = lev == 1.0
        main = placebo = None
        if always.sum() >= args.min_n and (~always).sum() >= args.min_n:
            d = float(G[always].mean() - G[~always].mean())
            lo, hi = boot_diff_ci(G[always], G[~always], args.resamples, rng)
            main = (d, lo, hi, int(always.sum()), int((~always).sum()),
                    float(G[always].mean()), float(G[~always].mean()))
        part = (lev > 0.0) & (lev < 1.0)
        never = lev == 0.0
        if part.sum() >= args.min_n and never.sum() >= args.min_n:
            d = float(G[part].mean() - G[never].mean())
            lo, hi = boot_diff_ci(G[part], G[never], args.resamples, rng)
            placebo = (d, lo, hi, int(part.sum()), int(never.sum()),
                       float(G[part].mean()), float(G[never].mean()))
        return main, placebo

    hdr = (f"{'row':>14} | {'contrast':>16} | {'n hi':>5} | {'n lo':>5} | "
           f"{'mean hi':>7} | {'mean lo':>7} | {'Delta':>6} | 95% CI")
    print(hdr)
    print("-" * len(hdr))

    def emit(label, main, placebo):
        for name, c in (("always − rest", main),
                        ("partial − never", placebo)):
            if c is None:
                if name == "always − rest" or args.placebo:
                    print(f"{label:>14} | {name:>16} | group under "
                          f"{args.min_n} questions — skipped")
                continue
            if name == "partial − never" and not args.placebo:
                continue
            d, lo, hi, n_hi, n_lo, m_hi, m_lo = c
            print(f"{label:>14} | {name:>16} | {n_hi:>5} | {n_lo:>5} | "
                  f"{m_hi:>7.3f} | {m_lo:>7.3f} | {d:>+6.3f} | "
                  f"[{lo:+.3f}, {hi:+.3f}]")

    rows = []  # (label, main, placebo, is_pooled)
    all_A, all_G = [], []
    for dataset in DATASETS:
        A, G = (np.array(v) for v in per_ds[dataset])
        if len(G) == 0:
            print(f"{dataset:>14} | no data — skipped")
            continue
        all_A.extend(A)
        all_G.extend(G)
        main, placebo = contrasts(A, G, dataset)
        emit(dataset, main, placebo)
        if main is None and placebo is None:
            continue
        rows.append((dataset, main, placebo, False))
    if not all_A:
        print("ERROR: no data found — nothing to plot.")
        sys.exit(1)
    A, G = np.array(all_A), np.array(all_G)
    main, placebo = contrasts(A, G, "POOLED")
    emit("POOLED", main, placebo)
    rows.append(("POOLED", main, placebo, True))

    row_h = 0.78 if args.placebo else 0.55
    fig, ax = plt.subplots(figsize=(9.2 if args.placebo else 8,
                                    row_h * len(rows) + 2.0))
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#DDDDDD", linewidth=0.8)
    ax.axvline(0, color="#999999", linestyle="--", linewidth=1.0, zorder=2)

    # Top row = first dataset; POOLED at the bottom, visually separated.
    ys = list(range(len(rows), 0, -1))
    all_hi = [c[2] for _, m, p, _ in rows for c in (m, p) if c is not None]
    all_lo = [c[1] for _, m, p, _ in rows for c in (m, p) if c is not None]
    text_x = max(all_hi) + 0.02
    off = 0.19 if args.placebo else 0.0
    for (label, main, placebo, pooled), y in zip(rows, ys):
        series = [(main, DARK_BLUE, y + off)]
        if args.placebo:
            series.append((placebo, PLACEBO_COLOR, y - off))
        for c, color, yy in series:
            if c is None:
                continue
            delta, lo, hi = c[0], c[1], c[2]
            marker = "D" if pooled else "o"
            size = 9 if pooled else 7
            ax.errorbar(delta, yy, xerr=[[delta - lo], [hi - delta]],
                        fmt=marker, color=color, markersize=size,
                        markeredgecolor="#333333", markeredgewidth=0.5,
                        ecolor=color, elinewidth=1.6, capsize=3,
                        capthick=1.6, zorder=3)
            ax.text(text_x, yy, f"{delta:+.3f} [{lo:+.3f}, {hi:+.3f}]",
                    va="center", ha="left", fontsize=8.5, color=color,
                    fontweight="bold" if pooled else "normal")
    ax.axhline(1.5, color="#CCCCCC", linewidth=0.8, zorder=1)

    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows],
                       fontsize=10)
    for tick, (_, _, _, pooled) in zip(ax.get_yticklabels(), rows):
        if pooled:
            tick.set_fontweight("bold")
    ax.set_ylim(0.4, len(rows) + 0.6)
    xmax = max(all_hi)
    xmin = min(min(all_lo), 0)
    ax.set_xlim(xmin - 0.02, xmax + 0.30)

    if args.placebo:
        ax.set_xlabel("Δ g-mean²")
        handles = [
            Line2D([], [], marker="o", linestyle="none", color=DARK_BLUE,
                   markeredgecolor="#333333", markeredgewidth=0.5,
                   markersize=8,
                   label="always correct (8/8) − rest   [the effect]"),
            Line2D([], [], marker="o", linestyle="none",
                   color=PLACEBO_COLOR, markeredgecolor="#333333",
                   markeredgewidth=0.5, markersize=8,
                   label="partial (1–7/8) − never (0/8)   [placebo]"),
        ]
        ax.legend(handles=handles, loc="lower left", frameon=True,
                  edgecolor="#CCCCCC", fontsize=8.5)
        ax.set_title(f"Accuracy-threshold effect on monitorability — "
                     f"{model_name}\n(the threshold is earned, not assumed: "
                     f"the placebo contrast straddles zero)")
    else:
        ax.set_xlabel("Δ g-mean²  (always correct − not always correct)")
        ax.set_title(f"Accuracy-threshold effect on monitorability — "
                     f"{model_name}\n(per dataset, hint types pooled; "
                     f"bootstrap 95% CI; POOLED = all questions)")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#999999")

    out = args.out
    if out is None:
        # RO1 bespoke analysis plots live under extra-plots/, not the main plots/ tree
        plots_dir = ((root / config["paths"]["plots_dir"]).parent
                     / "extra-plots" / "task-performance-vs-monitorability")
        if args.data != "test":
            plots_dir = plots_dir / args.data
        plots_dir.mkdir(parents=True, exist_ok=True)
        suffix = "_placebo" if args.placebo else ""
        out = (plots_dir /
               f"gmean2_acc_delta_forest{suffix}_{model_name}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
