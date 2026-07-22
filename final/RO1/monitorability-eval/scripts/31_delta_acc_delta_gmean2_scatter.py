#!/usr/bin/env python3
"""
Step 31: Change-vs-change scatter — does the accuracy shift a fine-tune
induces explain the monitorability shift it induces?

Steps 26-30 ask whether accuracy predicts monitorability WITHIN a model.
This step asks the confound question that matters for the training-side
results: when fine-tuning moves g-mean^2, is it just moving task
capability? If the two deltas were tightly coupled, every monitorability
claim would reduce to "the model got better/worse at the task".

One point per (trained model x dataset x hint type) cell:
    x = delta mean unhinted accuracy vs the base model
    y = delta g-mean^2 vs the base model
both read straight from delta_vs_base_per_cell.json (paired cluster
bootstrap point estimates, so the pairing to the base model is exact).

Color = dataset (validated 8-slot palette, domain-table order), marker
shape = trained model. An OLS fit over all cells with the Pearson r
annotation; stdout prints the pooled Pearson/Spearman plus the same two
statistics computed WITHIN each trained model, so the caption can report
whether the null holds per-model as well as pooled.

Usage:
    python scripts/31_delta_acc_delta_gmean2_scatter.py
    python scripts/31_delta_acc_delta_gmean2_scatter.py --models a b

    Options:
      --config   path to config YAML (default: config/mvp_config.yaml)
      --data     split: test (default) | eval
      --models   trained model HF ids (default: the two complete DPO runs)
      --out      output PNG path
                 (default: {plots_dir}/delta_acc_vs_delta_gmean2.png)

Input (per trained model, produced by 05_analyze_gmean2.py):
    {analysis_dir}/{model}/delta_vs_base_per_cell.json

Output:
    {plots_dir}/delta_acc_vs_delta_gmean2.png
"""

import argparse
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

from src.dataset import SPLITS, model_dir_name

DEFAULT_MODELS = [
    "er-abd/harmless-dpo-qwen3-4B-step-735",
    "er-abd/anti-syc-dpo-qwen3-4B-step-520",
]

# Domain-table order with the validated 8-slot categorical palette
# (same binding as steps 24/29).
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

# Marker shape carries model identity, so it survives grayscale.
MODEL_MARKERS = ["o", "s", "^", "D"]

LINE_COLOR = "#1B3A6B"
ZERO_COLOR = "#999999"


def short_label(model_id):
    """Compact legend label: drop the org prefix and the shared suffix."""
    tail = model_id.split("/")[-1]
    return tail.replace("-qwen3-4B", "").replace("qwen3-4B-", "")


def main():
    parser = argparse.ArgumentParser(
        description="Change-vs-change scatter of delta unhinted accuracy "
                    "vs delta g-mean^2 across fine-tuned models.")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--data", default="test", choices=SPLITS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    with config_path.open() as f:
        config = yaml.safe_load(f)

    analysis_root = root / config["paths"]["analysis_dir"]
    if args.data != "test":
        analysis_root = analysis_root / args.data

    color_of = dict(DATASET_COLORS)
    order_of = {ds: i for i, (ds, _) in enumerate(DATASET_COLORS)}

    pts = []  # (model_id, dataset, hint, d_acc, d_gmean2)
    for model_id in args.models:
        per_cell = (analysis_root / model_dir_name(model_id)
                    / "delta_vs_base_per_cell.json")
        if not per_cell.exists():
            print(f"WARNING: no delta_vs_base_per_cell.json for {model_id} "
                  f"— skipped")
            continue
        with per_cell.open() as f:
            payload = json.load(f)
        for cell, vals in payload.get("cells", {}).items():
            dataset, _, hint = cell.partition("/")
            if dataset not in color_of:
                continue
            d_acc = vals.get("acc_u", {}).get("delta")
            d_g = vals.get("gmean2", {}).get("delta")
            if d_acc is None or d_g is None:
                continue
            pts.append((model_id, dataset, hint, float(d_acc), float(d_g)))

    if len(pts) < 3:
        print("ERROR: not enough cells — nothing to plot.")
        sys.exit(1)

    pts.sort(key=lambda p: (args.models.index(p[0]) if p[0] in args.models
                            else 99, order_of[p[1]], p[2]))

    print(f"{'model':>22} | {'dataset':>14} | {'hint':>15} | "
          f"{'d acc_u':>8} | {'d g-mean2':>9}")
    print("-" * 82)
    for model_id, dataset, hint, d_acc, d_g in pts:
        print(f"{short_label(model_id):>22} | {dataset:>14} | {hint:>15} | "
              f"{d_acc:>+8.3f} | {d_g:>+9.3f}")

    xs = np.array([p[3] for p in pts])
    ys = np.array([p[4] for p in pts])
    r, r_p = stats.pearsonr(xs, ys)
    rho, rho_p = stats.spearmanr(xs, ys)
    print(f"\nPOOLED over {len(pts)} cells: "
          f"Pearson r = {r:+.3f} (p = {r_p:.3g}); "
          f"Spearman rho = {rho:+.3f} (p = {rho_p:.3g})")
    print(f"  R^2 = {r ** 2:.3f} — share of the cell-level g-mean^2 shift "
          f"linearly explained by the accuracy shift")

    print("\nWITHIN each trained model:")
    for model_id in args.models:
        mx = np.array([p[3] for p in pts if p[0] == model_id])
        my = np.array([p[4] for p in pts if p[0] == model_id])
        if len(mx) < 3:
            continue
        mr, mr_p = stats.pearsonr(mx, my)
        mrho, mrho_p = stats.spearmanr(mx, my)
        print(f"  {short_label(model_id):>22} (n={len(mx):>2}): "
              f"Pearson r = {mr:+.3f} (p = {mr_p:.3g}); "
              f"Spearman rho = {mrho:+.3f} (p = {mrho_p:.3g})")
        print(f"  {'':>22}          mean d acc_u = {mx.mean():+.3f}, "
              f"mean d g-mean^2 = {my.mean():+.3f}")

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.set_axisbelow(True)
    ax.grid(True, color="#DDDDDD", linewidth=0.8)
    ax.axhline(0, color=ZERO_COLOR, linewidth=1.0, zorder=1)
    ax.axvline(0, color=ZERO_COLOR, linewidth=1.0, zorder=1)

    marker_of = {m: MODEL_MARKERS[i % len(MODEL_MARKERS)]
                 for i, m in enumerate(args.models)}
    for model_id, dataset, hint, d_acc, d_g in pts:
        ax.scatter(d_acc, d_g, s=70, color=color_of[dataset],
                   marker=marker_of.get(model_id, "o"),
                   edgecolors="#333333", linewidths=0.6, zorder=3)

    slope, intercept = np.polyfit(xs, ys, 1)
    fx = np.array([xs.min(), xs.max()])
    ax.plot(fx, slope * fx + intercept, color=LINE_COLOR, linewidth=2.0,
            linestyle="--", zorder=2)
    p_txt = f"p={r_p:.3f}" if r_p >= 0.001 else "p<0.001"
    # Top-left: the bottom-left corner can hold a large negative outlier.
    ax.text(0.03, 0.97,
            f"OLS fit; Pearson r={r:+.2f}, {p_txt}, R²={r ** 2:.2f}",
            transform=ax.transAxes, fontsize=9, color="#333333",
            va="top", ha="left",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none",
                      pad=2))

    ax.set_xlabel("Δ unhinted task accuracy vs base")
    ax.set_ylabel("Δ g-mean²  vs base")
    ax.set_title("Does a fine-tune's accuracy shift explain its "
                 "monitorability shift?\n"
                 "(one point per trained model × dataset × hint type)")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#999999")

    handles = ([Line2D([], [], marker="o", linestyle="none", color=c,
                       markeredgecolor="#333333", markeredgewidth=0.6,
                       markersize=8, label=ds)
                for ds, c in DATASET_COLORS]
               + [Line2D([], [], marker=marker_of[m], linestyle="none",
                         color="#888888", markeredgecolor="#333333",
                         markeredgewidth=0.6, markersize=8,
                         label=short_label(m))
                  for m in args.models if m in marker_of])
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
        out = plots_dir / "delta_acc_vs_delta_gmean2.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
