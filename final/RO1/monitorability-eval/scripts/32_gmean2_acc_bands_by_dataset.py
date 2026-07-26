#!/usr/bin/env python3
"""
Step 32: Per-dataset dot plot of g-mean^2 grouped into CUSTOM unhinted-
accuracy bands — the by-dataset figure of step 27, but with the caller's
own binning of the correct-run count instead of the fixed never/sometimes/
always trichotomy.

Each ENTERING question (level-1 pass, gmean2_i defined) has an unhinted
correct-run count k in 0..N (N=8, so k in 0..8). A band is any set of k
values; per (dataset, band) the mean gmean2_i is drawn with a percentile-
bootstrap 95% CI. One axes, x = the 8 datasets, one light->dark dot per
band per dataset (same visual language as step 27's --per-dataset), with
per-band n printed to stdout.

Two presets (--scheme):
    1 : {0} {1,2} {3,4} {5,6} {7,8} {9}
    2 : {0,1} {2,3,4} {5,6,7} {8,9}
Scores top out at k=8, so any band that only contains 9 comes up empty and
is dropped automatically (reported in stdout).

Bands with fewer than --min-n questions in a dataset are dropped from that
dataset (the middle bands are thin per-dataset — that is exactly why step
27's per-dataset view collapsed to 3 groups; this step honours the finer
binning but still guards n).

Usage:
    python scripts/32_gmean2_acc_bands_by_dataset.py --scheme 1
    python scripts/32_gmean2_acc_bands_by_dataset.py --scheme 2
    python scripts/32_gmean2_acc_bands_by_dataset.py --bands "0|1,2|3,4|5,6,7,8"

    Options:
      --config     path to config YAML (default: config/mvp_config.yaml)
      --data       split: test (default) | eval
      --model      model/adapter HF id (default: Qwen/Qwen3-4B)
      --scheme     1 or 2 (preset bands; ignored if --bands is given)
      --bands      custom bands as k-lists separated by '|', e.g.
                   "0|1,2|3,4|5,6|7,8" — overrides --scheme
      --min-n      min questions per (dataset, band) to draw the dot
                   (default: 5)
      --resamples  bootstrap replicates per CI (default: 2000)
      --seed       bootstrap seed (default: 88)
      --out        output PNG path (default:
                   {plots_dir}/task-performance-vs-monitorability/
                   gmean2_acc_bands_scheme{N}_by_dataset_{model}.png)

Input (per dataset / hint type, produced by 04_compute_gmean2.py):
    {analysis_dir}/{model}/{dataset}/{hint}/results_per_instance.csv

Output:
    {plots_dir}/task-performance-vs-monitorability/
        gmean2_acc_bands_scheme{N}_by_dataset_{model}.png
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

# Preset band schemes, as lists of k-value sets over the correct-run count.
SCHEMES = {
    "1": [[0], [1, 2], [3, 4], [5, 6], [7, 8], [9]],
    "2": [[0, 1], [2, 3, 4], [5, 6, 7], [8, 9]],
}


def band_label(ks):
    ks = sorted(ks)
    return str(ks[0]) if len(ks) == 1 else f"{ks[0]}–{ks[-1]}"


def parse_bands(spec):
    bands = []
    for chunk in spec.split("|"):
        ks = [int(x) for x in chunk.split(",") if x.strip() != ""]
        if ks:
            bands.append(ks)
    return bands


def boot_mean_ci(values, n_resamples, rng):
    arr = np.asarray(values)
    idx = rng.integers(0, len(arr), size=(n_resamples, len(arr)))
    lo, hi = np.percentile(arr[idx].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def main():
    parser = argparse.ArgumentParser(
        description="Per-dataset g-mean^2 by custom unhinted-accuracy bands.")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--data", default="test", choices=SPLITS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--scheme", choices=["1", "2"], default="1")
    parser.add_argument("--bands", default=None,
                        help="override preset, e.g. '0|1,2|3,4|5,6|7,8'")
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
    rng = np.random.default_rng(args.seed)

    if args.bands:
        bands = parse_bands(args.bands)
        scheme_tag = "custom"
    else:
        bands = SCHEMES[args.scheme]
        scheme_tag = args.scheme

    # light -> dark blue ramp, one shade per band
    cmap = plt.get_cmap("Blues")
    n_bands = len(bands)
    colors = [cmap(0.30 + 0.65 * i / max(n_bands - 1, 1))
              for i in range(n_bands)]
    labels = [band_label(b) for b in bands]

    print(f"Bands (correct-run count k, N={N_RUNS}): "
          + "  ".join(f"[{lab}]" for lab in labels))

    # ---- load per-dataset entering questions ----
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
    if not any(a for a, _ in per_ds.values()):
        print("ERROR: no data found — nothing to plot.")
        sys.exit(1)

    # RO1 bespoke analysis plots live under extra-plots/, not the main plots/ tree
    plots_dir = ((root / config["paths"]["plots_dir"]).parent
                 / "extra-plots" / "task-performance-vs-monitorability")
    if args.data != "test":
        plots_dir = plots_dir / args.data
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5.4))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#DDDDDD", linewidth=0.8)

    # horizontal offsets: spread the band dots within each dataset column
    half = 0.34
    offsets = (np.linspace(-half, half, n_bands) if n_bands > 1 else [0.0])

    empty_bands = set(range(n_bands))
    all_pts, plotted = [], []
    for d_idx, dataset in enumerate(DATASETS):
        A, G = (np.array(v) for v in per_ds[dataset])
        if len(G) == 0:
            print(f"\n{dataset}: no entering questions — skipped")
            continue
        k = np.clip(np.round(A * N_RUNS).astype(int), 0, N_RUNS)
        print(f"\n{dataset} — {len(G)} entering questions")
        print(f"{'band':>8} | {'n':>4} | {'g-mean2':>8} | 95% CI")
        print("-" * 46)
        drew_any = False
        for b_idx, ks in enumerate(bands):
            mask = np.isin(k, ks)
            n = int(mask.sum())
            if n == 0:
                print(f"{labels[b_idx]:>8} | {n:>4} | (empty)")
                continue
            empty_bands.discard(b_idx)
            vals = G[mask]
            mean = float(vals.mean())
            if n < args.min_n:
                print(f"{labels[b_idx]:>8} | {n:>4} | (dropped: n < "
                      f"{args.min_n})")
                continue
            lo, hi = boot_mean_ci(vals, args.resamples, rng)
            print(f"{labels[b_idx]:>8} | {n:>4} | {mean:>8.3f} | "
                  f"[{lo:.3f}, {hi:.3f}]")
            x = d_idx + offsets[b_idx]
            ax.errorbar(x, mean, yerr=[[mean - lo], [hi - mean]], fmt="o",
                        color=colors[b_idx], markersize=7,
                        markeredgecolor="#333333", markeredgewidth=0.5,
                        ecolor=colors[b_idx], elinewidth=1.4, capsize=3,
                        capthick=1.4, zorder=3)
            all_pts.extend([lo, hi])
            drew_any = True
        if drew_any:
            plotted.append((d_idx, dataset))

    if not plotted:
        print("\nERROR: no dataset had enough questions in any band.")
        sys.exit(1)

    dropped = [labels[i] for i in sorted(empty_bands)]
    if dropped:
        print(f"\nNOTE: band(s) {', '.join(dropped)} are empty in every "
              f"dataset (scores top out at k={N_RUNS}) — not drawn.")

    ax.set_xticks([i for i, _ in plotted])
    ax.set_xticklabels([ds for _, ds in plotted])
    ax.set_xlim(-0.6, len(DATASETS) - 0.4)
    ax.set_ylim(min(all_pts) - 0.02, max(all_pts) + 0.02)
    for i in range(len(DATASETS) - 1):
        ax.axvline(i + 0.5, color="#EEEEEE", linewidth=0.8, zorder=1)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Mean g-mean²")
    ax.set_title(f"Monitorability by unhinted-accuracy band, per dataset "
                 f"— {model_name}\n(entering questions, hint types pooled; "
                 f"band = correct unhinted runs out of {N_RUNS}; "
                 f"bootstrap 95% CI)")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#999999")

    # legend: only the bands that actually got drawn somewhere
    handles = [Line2D([], [], marker="o", linestyle="none", color=colors[i],
                      markeredgecolor="#333333", markeredgewidth=0.5,
                      markersize=7, label=labels[i])
               for i in range(n_bands) if i not in empty_bands]
    ax.legend(handles=handles, title="Correct runs (k / 8)",
              loc="upper left", bbox_to_anchor=(1.005, 1.0),
              frameon=True, edgecolor="#CCCCCC", fontsize=9)

    out = args.out or (plots_dir /
                       f"gmean2_acc_bands_scheme{scheme_tag}_by_dataset_"
                       f"{model_name}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
