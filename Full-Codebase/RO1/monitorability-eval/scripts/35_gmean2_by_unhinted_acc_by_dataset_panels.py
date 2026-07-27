#!/usr/bin/env python3
"""
Step 35: The pooled accuracy-level plot of step 27 (--no-line), drawn as an
8-panel small multiple — one panel per dataset.

Step 27 pools all datasets into one panel of mean gmean2_i at each unhinted-
accuracy level k/8. This step draws the SAME level plot (dots + bootstrap
95% CI, n under each dot, NO connecting line) separately for each of the 8
datasets (2x4, domain-table order, shared axes), so the reader can see
whether the pooled shape — flat across 0/8..7/8, up at 8/8 — holds per
dataset.

Per dataset the 9 levels are thin (middle levels hold ~5-40 questions), so
levels with n < --min-n are dropped from that panel (reported); this is the
same power caveat that made step 27's --per-dataset view collapse to 3
groups. Present with that caveat, or use step 27 --per-dataset / step 32 for
the powered per-dataset views.

Usage:
    python scripts/35_gmean2_by_unhinted_acc_by_dataset_panels.py          # base
    python scripts/35_gmean2_by_unhinted_acc_by_dataset_panels.py --model <hf-id>

    Options:
      --config     path to config YAML (default: config/mvp_config.yaml)
      --data       split: test (default) | eval
      --model      model/adapter HF id (default: Qwen/Qwen3-4B)
      --min-n      min questions per (dataset, level) to draw the dot
                   (default: 8)
      --resamples  bootstrap replicates per CI (default: 5000)
      --seed       bootstrap seed (default: 88)
      --out        output PNG path (default:
                   {plots_dir}/task-performance-vs-monitorability/
                   gmean2_by_unhinted_acc_panels_{model}.png)

Input (per dataset / hint type, produced by 04_compute_gmean2.py):
    {analysis_dir}/{model}/{dataset}/{hint}/results_per_instance.csv

Output:
    {plots_dir}/task-performance-vs-monitorability/
        gmean2_by_unhinted_acc_panels_{model}.png
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
import yaml

from src.dataset import SPLITS, model_dir_name, resolve_paths
from src.hints import HINT_TYPES

DEFAULT_MODEL = "Qwen/Qwen3-4B"
N_RUNS = 8

DATASETS = ["mmlu_pro", "gpqa_diamond", "lsat_ar", "lsat_lr",
            "medxpertqa", "lexam", "codemmlu", "wmdp"]

DARK_BLUE = "#1B3A6B"


def band_label(ks):
    ks = sorted(ks)
    core = str(ks[0]) if len(ks) == 1 else f"{ks[0]}–{ks[-1]}"
    return f"{core}/{N_RUNS}"


def parse_bands(spec):
    bands = []
    for chunk in spec.split("|"):
        ks = [int(x) for x in chunk.split(",") if x.strip() != ""]
        if ks:
            bands.append(ks)
    return bands


def boot_ci(v, n_resamples, rng):
    v = np.asarray(v, float)
    if len(v) < 2:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(v), (n_resamples, len(v)))
    m = v[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    parser = argparse.ArgumentParser(
        description="Per-dataset small multiples of g-mean^2 by accuracy "
                    "level.")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--data", default="test", choices=SPLITS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--bands", default=None,
                        help="group accuracy levels into bands, k-lists "
                             "separated by '|', e.g. '0,1,2,3|4,5,6,7|8' "
                             "(default: one dot per level 0..8). Banding "
                             "boosts per-dataset n; keep 8 in its OWN band so "
                             "the consistency jump is not mistaken for a "
                             "correctness gradient")
    parser.add_argument("--min-n", type=int, default=8)
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

    # ---- per-dataset entering questions (acc level, gmean2) ----
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
                    g, a = row.get("gmean2_i", ""), row.get("acc_u_i", "")
                    if g in ("", "None", "N/A") or a in ("", "None", "N/A"):
                        continue
                    per_ds[dataset][0].append(float(a))
                    per_ds[dataset][1].append(float(g))
    if missing:
        print("WARNING: missing slices skipped: " + ", ".join(missing))

    # ---- units: either 9 individual levels or caller-defined bands ----
    if args.bands:
        bands = parse_bands(args.bands)
        units = [(band_label(ks), set(ks)) for ks in bands]  # (label, kset)
        x_pos = list(range(len(units)))                       # categorical x
    else:
        units = [(str(k), {k}) for k in range(N_RUNS + 1)]
        x_pos = [k / N_RUNS for k in range(N_RUNS + 1)]

    # ---- per-dataset per-unit stats ----
    panels = {}
    all_bounds = []
    for dataset in DATASETS:
        A = np.array(per_ds[dataset][0])
        G = np.array(per_ds[dataset][1])
        if len(G) == 0:
            panels[dataset] = None
            print(f"\n{dataset}: no entering questions — skipped")
            continue
        lev = np.clip(np.round(A * N_RUNS).astype(int), 0, N_RUNS)
        print(f"\n{dataset} — {len(G)} entering questions")
        drawn = []  # (x, mean, lo, hi, n)
        for (label, kset), x in zip(units, x_pos):
            m = np.isin(lev, list(kset))
            n = int(m.sum())
            if n == 0:
                continue
            v = G[m]
            lo, hi = boot_ci(v, args.resamples, rng)
            kept = n >= args.min_n
            print(f"   {label:>4}  n={n:4d}  mean={v.mean():.3f}  "
                  f"CI[{lo:.3f},{hi:.3f}]{'' if kept else '  (dropped)'}")
            if kept:
                drawn.append((x, float(v.mean()), lo, hi, n))
                all_bounds.extend([lo, hi])
        panels[dataset] = drawn

    if not all_bounds:
        print("\nERROR: no populated units anywhere — nothing to plot.")
        sys.exit(1)

    ylo, yhi = min(all_bounds) - 0.03, max(all_bounds) + 0.03

    # ---- figure: 2x4 small multiples, shared axes ----
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True, sharey=True)
    for ax, dataset in zip(axes.flat, DATASETS):
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color="#EEEEEE", linewidth=0.8)
        ax.set_title(dataset, fontsize=10, fontweight="bold")
        drawn = panels.get(dataset)
        if not drawn:
            ax.text(0.5, 0.5, "insufficient data", transform=ax.transAxes,
                    ha="center", va="center", color="#999999", fontsize=9)
        else:
            xs = [d[0] for d in drawn]
            ys = [d[1] for d in drawn]
            los = [d[2] for d in drawn]
            his = [d[3] for d in drawn]
            ns = [d[4] for d in drawn]
            ax.errorbar(xs, ys,
                        yerr=[np.array(ys) - np.array(los),
                              np.array(his) - np.array(ys)],
                        fmt="o", color=DARK_BLUE, markersize=6,
                        markeredgecolor="#333333", markeredgewidth=0.4,
                        ecolor=DARK_BLUE, elinewidth=1.2, capsize=2.5,
                        capthick=1.2, zorder=3)
            for x, n in zip(xs, ns):
                ax.annotate(f"{n}", xy=(x, 0),
                            xycoords=("data", "axes fraction"),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=6.5,
                            color="#999999")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([u[0] for u in units], fontsize=8)
        pad = 0.5 if args.bands else 0.08
        ax.set_xlim(x_pos[0] - pad, x_pos[-1] + pad)
        ax.set_ylim(ylo, yhi)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#999999")

    for ax in axes[:, 0]:
        ax.set_ylabel("Mean g-mean²", fontsize=9)
    xlab = ("Correct unhinted runs, banded (/8)" if args.bands
            else "Correct unhinted runs (/8)")
    for ax in axes[-1, :]:
        ax.set_xlabel(xlab, fontsize=9)

    band_note = ("accuracy banded" if args.bands
                 else "one dot per accuracy level")
    fig.suptitle("Monitorability by unhinted task accuracy, per dataset — "
                 f"{model_name}\n(entering questions, hint types pooled; "
                 f"{band_note}; groups with n < {args.min_n} dropped; "
                 "bootstrap 95% CI; n under each dot)", fontsize=12)

    out = args.out
    if out is None:
        plots_dir = ((root / config["paths"]["plots_dir"]).parent
                     / "extra-plots" / "task-performance-vs-monitorability")
        if args.data != "test":
            plots_dir = plots_dir / args.data
        plots_dir.mkdir(parents=True, exist_ok=True)
        suffix = "_banded" if args.bands else ""
        out = (plots_dir /
               f"gmean2_by_unhinted_acc_panels{suffix}_{model_name}.png")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
