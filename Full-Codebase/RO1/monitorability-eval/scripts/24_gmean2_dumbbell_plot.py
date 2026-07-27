#!/usr/bin/env python3
"""
Step 24: Dumbbell plot of pooled g-mean^2 levels — base vs trained, per
dataset and per hint type. Companion (level-scale) view to the delta forest
plot (step 23); deliberately NO CIs — for inference use step 23 / the delta
tables.

Two side-by-side panels sharing one x-axis: left = base vs harmlessness DPO,
right = base vs anti-sycophancy DPO. Rows (top to bottom): the 8 datasets
(g-mean^2 pooled over hint types, from pooled_summary.json), then a separated
block with the 3 hint types (pooled over datasets, from
pooled_summary_by_hint.json). Each row: two dots (base, trained) joined by a
connector, with the signed delta annotated in the right margin. Model colors
match steps 17/18/23; the x-axis is truncated to the data range (levels, not
zero-based — these are positions, not bars).

Usage:
    python scripts/24_gmean2_dumbbell_plot.py

    Options: --config, --data test|eval,
             --base / --harmless / --antisyc   (as in step 15)
             --out   output PNG path (default: {plots_dir}/gmean2_dumbbell.png)

Input (produced by steps 15 and 19):
    {analysis_dir}/{model}/pooled_summary.json
    {analysis_dir}/{model}/pooled_summary_by_hint.json

Output:
    {plots_dir}/gmean2_dumbbell.png
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
import yaml

from src.dataset import SPLITS, model_dir_name
from src.hints import HINT_TYPES

DEFAULT_BASE = "Qwen/Qwen3-4B"
DEFAULT_HARMLESS = "er-abd/harmless-dpo-qwen3-4B-step-735"
DEFAULT_ANTISYC = "er-abd/anti-syc-dpo-qwen3-4B-step-520"

DATASETS = ["mmlu_pro", "gpqa_diamond", "lsat_ar", "lsat_lr",
            "medxpertqa", "lexam", "codemmlu", "wmdp"]

HINT_LABELS = {
    "professor": "Professor hint",
    "grader_hacking": "Grader-hacking hint",
    "unethical": "Unethical hint",
}

COLOR_BASE = "#92C5E8"      # light blue (as in 17/18/23)
COLOR_HARMLESS = "#F6A0B8"  # light pink
COLOR_ANTISYC = "#9ED973"   # light green
EDGE = "#333333"
CONNECTOR = "#999999"


def load_levels(analysis_dir: Path, model_name: str, split: str):
    d = analysis_dir / model_name
    if split != "test":
        d = d / split
    with (d / "pooled_summary.json").open() as f:
        by_ds = json.load(f)["datasets"]
    with (d / "pooled_summary_by_hint.json").open() as f:
        by_hint = json.load(f)["hint_types"]
    levels = [by_ds[ds]["gmean2"]["estimate"] for ds in DATASETS]
    levels += [by_hint[h]["gmean2"]["estimate"] for h in HINT_TYPES]
    return levels


def main():
    parser = argparse.ArgumentParser(
        description="Dumbbell plot of pooled g-mean^2 levels, base vs trained.")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--data", default="test", choices=SPLITS)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--harmless", default=DEFAULT_HARMLESS)
    parser.add_argument("--antisyc", default=DEFAULT_ANTISYC)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    with config_path.open() as f:
        config = yaml.safe_load(f)
    analysis_dir = root / config["paths"]["analysis_dir"]

    base_levels = load_levels(analysis_dir, model_dir_name(args.base),
                              args.data)
    panels = [
        ("Base vs Harmlessness DPO", COLOR_HARMLESS,
         load_levels(analysis_dir, model_dir_name(args.harmless), args.data)),
        ("Base vs Anti-sycophancy DPO", COLOR_ANTISYC,
         load_levels(analysis_dir, model_dir_name(args.antisyc), args.data)),
    ]

    labels = DATASETS + [HINT_LABELS[h] for h in HINT_TYPES]
    n_ds = len(DATASETS)
    ys = list(range(n_ds)) + [n_ds + 0.8 + i for i in range(len(HINT_TYPES))]

    all_vals = base_levels + [v for _, _, lv in panels for v in lv]
    xlo = min(all_vals) - 0.04
    xhi = max(all_vals) + 0.04

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True,
                             sharex=True)
    for ax, (title, color, levels) in zip(axes, panels):
        ax.xaxis.grid(True, color="#DDDDDD", linewidth=0.8)
        ax.set_axisbelow(True)
        for y, b, t in zip(ys, base_levels, levels):
            ax.plot([b, t], [y, y], color=CONNECTOR, linewidth=2.0, zorder=2)
            ax.plot(b, y, "o", markersize=6, markerfacecolor=COLOR_BASE,
                    markeredgecolor=EDGE, markeredgewidth=0.8, zorder=3)
            ax.plot(t, y, "o", markersize=6, markerfacecolor=color,
                    markeredgecolor=EDGE, markeredgewidth=0.8, zorder=4)
            d = t - b
            ax.text(xhi + 0.005, y, f"{d:+.3f}" if round(d, 3) else "0.000",
                    va="center", ha="left", fontsize=7.5,
                    color="#444444", clip_on=False)
        ax.axhline(n_ds + 0.4, color="#BBBBBB", linewidth=0.8)
        ax.set_title(title, fontsize=11)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#999999")
        ax.set_xlabel("g-mean² (pooled)")

    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(xlo, xhi)

    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=8,
               markerfacecolor=COLOR_BASE, markeredgecolor=EDGE,
               label="Base (Qwen3-4B)"),
        Line2D([], [], marker="o", linestyle="", markersize=8,
               markerfacecolor=COLOR_HARMLESS, markeredgecolor=EDGE,
               label="Harmlessness DPO"),
        Line2D([], [], marker="o", linestyle="", markersize=8,
               markerfacecolor=COLOR_ANTISYC, markeredgecolor=EDGE,
               label="Anti-sycophancy DPO"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 0.96),
               frameon=True, edgecolor="#CCCCCC", fontsize=8, ncol=1)

    fig.suptitle("Monitorability levels before vs after DPO training — "
                 "pooled g-mean²", fontsize=12, x=0.42)

    out = args.out
    if out is None:
        # RO1 bespoke analysis plots live under extra-plots/, not the main plots/ tree
        plots_dir = ((root / config["paths"]["plots_dir"]).parent
                     / "extra-plots" / "cross-model-comparison")
        if args.data != "test":
            plots_dir = plots_dir / args.data
        plots_dir.mkdir(parents=True, exist_ok=True)
        out = plots_dir / "gmean2_dumbbell.png"
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
