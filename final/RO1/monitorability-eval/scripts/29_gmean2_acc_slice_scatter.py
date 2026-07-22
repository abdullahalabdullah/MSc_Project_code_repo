#!/usr/bin/env python3
"""
Step 29: Slice-level scatter of unhinted task accuracy vs g-mean^2 for ONE
model — 24 dots, one per (dataset, hint type), with an OLS fit line.

Deliberately the CONTRAST figure to steps 27/28: at this level of
aggregation the association is null (base model: Pearson r ~ -0.01) —
between-dataset difficulty differences dominate and are uncorrelated with
monitorability (e.g. lsat_lr = most accurate, least monitorable), so the
flat fit line is the point: task-level accuracy does not predict
monitorability, while question-level solution reliability does (step 28's
threshold effect). Present the two figures together.

x = the slice's mean unhinted accuracy (mean_acc_u), y = the slice's
g-mean^2, both from results_summary.json. Color = dataset (validated
8-color categorical palette, domain-table order); marker shape = hint type
(so identity survives grayscale). OLS fit line over the 24 points with the
Pearson r / p annotation; stdout prints the slice table plus Pearson and
Spearman coefficients.

Usage:
    python scripts/29_gmean2_acc_slice_scatter.py                  # base
    python scripts/29_gmean2_acc_slice_scatter.py --model <hf-id>

    Options:
      --config   path to config YAML (default: config/mvp_config.yaml)
      --data     split: test (default) | eval
      --model    model/adapter HF id (default: Qwen/Qwen3-4B)
      --out      output PNG path
                 (default: {plots_dir}/gmean2_vs_acc_slices_{model}.png)

Input (per dataset / hint type, produced by 04_compute_gmean2.py):
    {analysis_dir}/{model}/{dataset}/{hint}/results_summary.json

Output:
    {plots_dir}/gmean2_vs_acc_slices_{model}.png
"""

import argparse
import csv
import json
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

# Domain-table order with the validated 8-slot categorical palette
# (same binding as 24_length_gmean2_curve_by_dataset.py).
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

# Marker shape carries hint-type identity, so it survives grayscale.
HINT_MARKERS = [
    ("professor", "o", "Professor hint"),
    ("grader_hacking", "s", "Grader-hacking hint"),
    ("unethical", "^", "Unethical hint"),
]

LINE_COLOR = "#1B3A6B"


def main():
    parser = argparse.ArgumentParser(
        description="24-slice scatter of unhinted accuracy vs g-mean^2 "
                    "with an OLS fit line.")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--data", default="test", choices=SPLITS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--pool-hints", action="store_true",
                        help="one point per DATASET (pool every question "
                             "across all 3 hint types) instead of 24 "
                             "dataset×hint slices; writes …_datasets_… .png")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    with config_path.open() as f:
        config = yaml.safe_load(f)

    model_name = model_dir_name(args.model)

    xs, ys, missing = [], [], []
    pts = []  # (dataset, hint, acc, gmean2)  [hint = None in pool mode]
    for dataset, _ in DATASET_COLORS:
        if args.pool_hints:
            # pool every entering question across the 3 hint types
            accs, gs = [], []
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
                        if g in ("", "None", "N/A") or a in ("", "None",
                                                             "N/A"):
                            continue
                        accs.append(float(a))
                        gs.append(float(g))
            if not gs:
                missing.append(f"{dataset} (no entering questions)")
                continue
            acc, g = float(np.mean(accs)), float(np.mean(gs))
            pts.append((dataset, None, acc, g))
            xs.append(acc)
            ys.append(g)
        else:
            for hint in HINT_TYPES:
                paths = resolve_paths(root, config, dataset, model_name, hint,
                                      split=args.data)
                summary = paths["analysis_dir"] / "results_summary.json"
                if not summary.exists():
                    missing.append(f"{dataset}/{hint}")
                    continue
                with summary.open() as f:
                    s = json.load(f)
                acc, g = s.get("mean_acc_u"), s.get("gmean2")
                if acc is None or g is None:
                    missing.append(f"{dataset}/{hint}")
                    continue
                pts.append((dataset, hint, float(acc), float(g)))
                xs.append(float(acc))
                ys.append(float(g))
    if missing:
        print("WARNING: missing slices skipped: " + ", ".join(missing))
    if len(pts) < 3:
        print("ERROR: not enough points — nothing to plot.")
        sys.exit(1)

    if args.pool_hints:
        print(f"{'dataset':>14} | {'acc_u':>6} | {'g-mean2':>7}")
        print("-" * 34)
        for dataset, _, acc, g in pts:
            print(f"{dataset:>14} | {acc:>6.3f} | {g:>7.3f}")
    else:
        print(f"{'dataset':>14} | {'hint':>15} | {'acc_u':>6} | {'g-mean2':>7}")
        print("-" * 52)
        for dataset, hint, acc, g in pts:
            print(f"{dataset:>14} | {hint:>15} | {acc:>6.3f} | {g:>7.3f}")

    xs, ys = np.array(xs), np.array(ys)
    r, r_p = stats.pearsonr(xs, ys)
    rho, rho_p = stats.spearmanr(xs, ys)
    unit = "datasets" if args.pool_hints else "slices"
    print(f"\n{len(pts)} {unit}: Pearson r = {r:+.3f} (p = {r_p:.3g}); "
          f"Spearman rho = {rho:+.3f} (p = {rho_p:.3g})")

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.set_axisbelow(True)
    ax.grid(True, color="#DDDDDD", linewidth=0.8)

    color_of = dict(DATASET_COLORS)
    marker_of = {h: m for h, m, _ in HINT_MARKERS}
    for dataset, hint, acc, g in pts:
        marker = "o" if args.pool_hints else marker_of[hint]
        size = 120 if args.pool_hints else 70
        ax.scatter(acc, g, s=size, color=color_of[dataset],
                   marker=marker, edgecolors="#333333",
                   linewidths=0.6, zorder=3)

    ax.set_xlim(0, 1)
    ax.set_ylim(min(ys) - 0.05, max(ys) + 0.05)
    if args.pool_hints:
        ax.set_xlabel("Unhinted task accuracy (dataset mean)")
        ax.set_title(f"Dataset-level g-mean² vs unhinted accuracy — "
                     f"{model_name}\n(one point per dataset, all questions "
                     f"pooled across the 3 hint types)")
    else:
        ax.set_xlabel("Unhinted task accuracy (mean_acc_u of the slice)")
        ax.set_title(f"Slice-level g-mean² vs unhinted accuracy — "
                     f"{model_name}\n(one point per dataset × hint type)")
    ax.set_ylabel("g-mean²")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#999999")

    handles = [Line2D([], [], marker="o", linestyle="none", color=c,
                      markeredgecolor="#333333", markeredgewidth=0.6,
                      markersize=8, label=ds)
               for ds, c in DATASET_COLORS]
    if not args.pool_hints:
        handles += [Line2D([], [], marker=m, linestyle="none",
                           color="#888888", markeredgecolor="#333333",
                           markeredgewidth=0.6, markersize=8, label=lbl)
                    for _, m, lbl in HINT_MARKERS]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.005, 1.0),
              frameon=True, edgecolor="#CCCCCC", fontsize=9)

    out = args.out
    if out is None:
        # RO1 bespoke analysis plots live under extra-plots/, not the main plots/ tree
        plots_dir = ((root / config["paths"]["plots_dir"]).parent
                     / "extra-plots" / "task-performance-vs-monitorability")
        if args.data != "test":
            plots_dir = plots_dir / args.data
        plots_dir.mkdir(parents=True, exist_ok=True)
        stem = ("gmean2_vs_acc_datasets" if args.pool_hints
                else "gmean2_vs_acc_slices")
        out = plots_dir / f"{stem}_{model_name}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
