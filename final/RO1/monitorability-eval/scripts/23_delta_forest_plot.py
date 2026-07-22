#!/usr/bin/env python3
"""
Step 23: Forest plot of the g-mean^2 deltas — (trained − base) with
paired-bootstrap 95% CIs, per dataset and per hint type.

Two side-by-side panels sharing one x-axis: left = harmlessness DPO − base,
right = anti-sycophancy DPO − base. Rows (top to bottom): the 8 datasets
(delta pooled over hint types, from step 15's delta_vs_base.json), then a
separated block with the 3 hint types (delta pooled over datasets, from
step 19's delta_vs_base_by_hint.json). Each row: dot at the point estimate,
horizontal whisker spanning the paired-bootstrap 95% CI, dashed line at 0.
A CI clear of 0 is a paired test at the 5% level; whiskers crossing the line
are honest nulls. No new statistics — the plot renders exactly the numbers
in the delta JSONs.

Usage:
    python scripts/23_delta_forest_plot.py

    Options: --config, --data test|eval,
             --base / --harmless / --antisyc   (as in step 15)
             --out   output PNG path (default: {plots_dir}/gmean2_delta_forest.png)

Input (produced by steps 15 and 19):
    {analysis_dir}/{trained_model}/delta_vs_base.json
    {analysis_dir}/{trained_model}/delta_vs_base_by_hint.json

Output:
    {plots_dir}/gmean2_delta_forest.png
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from src.dataset import SPLITS, model_dir_name
from src.hints import HINT_TYPES

DEFAULT_BASE = "Qwen/Qwen3-4B"
DEFAULT_HARMLESS = "er-abd/harmless-dpo-qwen3-4B-step-735"
DEFAULT_ANTISYC = "er-abd/anti-syc-dpo-qwen3-4B-step-520"

# Same presentation order as the main-results tables.
DATASETS = ["mmlu_pro", "gpqa_diamond", "lsat_ar", "lsat_lr",
            "medxpertqa", "lexam", "codemmlu", "wmdp"]

HINT_LABELS = {
    "professor": "Professor hint",
    "grader_hacking": "Grader-hacking hint",
    "unethical": "Unethical hint",
}

# Model colors as in 17/18 (marker fill); darker edge for print contrast.
PANELS = [
    ("Harmlessness DPO − Base", "delta harmless", "#F6A0B8"),
    ("Anti-sycophancy DPO − Base", "delta antisyc", "#9ED973"),
]
EDGE = "#333333"


def load_deltas(analysis_dir: Path, model_name: str, split: str):
    d = analysis_dir / model_name
    if split != "test":
        d = d / split
    with (d / "delta_vs_base.json").open() as f:
        by_ds = json.load(f)["datasets"]
    with (d / "delta_vs_base_by_hint.json").open() as f:
        by_hint = json.load(f)["hint_types"]
    return by_ds, by_hint


def main():
    parser = argparse.ArgumentParser(
        description="Forest plot of g-mean^2 deltas with paired-bootstrap CIs.")
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

    trained = [model_dir_name(args.harmless), model_dir_name(args.antisyc)]
    data = {m: load_deltas(analysis_dir, m, args.data) for m in trained}

    # Row layout (top to bottom): datasets, gap, hint types.
    ds_rows = [(ds, ds) for ds in DATASETS]
    hint_rows = [(h, HINT_LABELS[h]) for h in HINT_TYPES]
    labels = [lab for _, lab in ds_rows] + [lab for _, lab in hint_rows]
    n_ds = len(ds_rows)
    ys = list(range(n_ds)) + [n_ds + 0.8 + i for i in range(len(hint_rows))]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2), sharey=True,
                             sharex=True)
    for ax, (title, _, color), model_name in zip(axes, PANELS, trained):
        by_ds, by_hint = data[model_name]
        ax.axvline(0.0, color="#888888", linewidth=1.0, linestyle="--",
                   zorder=2)
        ax.xaxis.grid(True, color="#DDDDDD", linewidth=0.8)
        ax.set_axisbelow(True)
        entries = ([by_ds[ds]["gmean2"] for ds, _ in ds_rows]
                   + [by_hint[h]["gmean2"] for h, _ in hint_rows])
        for y, e in zip(ys, entries):
            d, (lo, hi) = e["delta"], e["ci95"]
            ax.plot([lo, hi], [y, y], color=EDGE, linewidth=1.4,
                    solid_capstyle="butt", zorder=3)
            for x_end in (lo, hi):
                ax.plot([x_end, x_end], [y - 0.14, y + 0.14], color=EDGE,
                        linewidth=1.2, zorder=3)
            ax.plot(d, y, marker="o", markersize=7, markerfacecolor=color,
                    markeredgecolor=EDGE, markeredgewidth=0.8, zorder=4)
        # Separator between the dataset block and the hint-type block.
        ax.axhline(n_ds + 0.4 - 0.5 + 0.5, color="#BBBBBB", linewidth=0.8)
        ax.set_title(title, fontsize=11)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#999999")
        ax.set_xlabel("Δ g-mean² (trained − base)")

    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    lim = 0.15
    axes[0].set_xlim(-lim, lim)
    fig.suptitle("Change in monitorability after DPO training — Δ g-mean² "
                 "with paired-bootstrap 95% CIs", fontsize=12)
    fig.text(0.5, 0.005,
             "Datasets: Δ pooled over the 3 hint types. Hint types (bottom "
             "block): Δ pooled over the 8 datasets. CI clear of 0 ⇒ paired "
             "test significant at the 5% level.",
             ha="center", fontsize=8, color="#555555")

    out = args.out
    if out is None:
        # RO1 bespoke analysis plots live under extra-plots/, not the main plots/ tree
        plots_dir = ((root / config["paths"]["plots_dir"]).parent
                     / "extra-plots" / "cross-model-comparison")
        if args.data != "test":
            plots_dir = plots_dir / args.data
        plots_dir.mkdir(parents=True, exist_ok=True)
        out = plots_dir / "gmean2_delta_forest.png"
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
