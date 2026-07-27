#!/usr/bin/env python3
"""
Step 17: Grouped bar chart of POOLED g-mean^2 (mean of the 3 hint types) —
base vs harmlessness DPO vs anti-sycophancy DPO, per dataset.

Companion to 16_gmean2_barchart.py (per-hint bars): here each dataset has just
three columns, one per model, showing the hint-type-pooled g-mean^2 from
15_pooled_summary_tables.py's pooled_summary.json. Error bars are the PAIRED
cluster-bootstrap 95% CIs (one shared instance resample reused across hint
types, 10k resamples, seed 88) — not per-hint CIs. The point estimate is
printed above each error bar. Each model has its own color (base light blue,
harmlessness DPO light pink, anti-sycophancy DPO light green — the same trio
as step 16, there bound to hint types) plus the same model hatches as step 16
(base solid, harmlessness DPO striped, anti-sycophancy DPO dotted), so the
figure also survives grayscale.

Usage:
    python scripts/17_pooled_gmean2_barchart.py

    Options:
      --config    path to config YAML (default: config/mvp_config.yaml)
      --data      split: test (default) | eval
      --base      base model id (default: Qwen/Qwen3-4B)
      --harmless  harmlessness DPO id (default: er-abd/harmless-dpo-qwen3-4B-step-735)
      --antisyc   anti-sycophancy DPO id (default: er-abd/anti-syc-dpo-qwen3-4B-step-520)
      --out       output PNG path (default: {plots_dir}/gmean2_pooled_3models.png)

Input (per model, produced by 15_pooled_summary_tables.py):
    {analysis_dir}/{model}/pooled_summary.json

Output:
    {plots_dir}/gmean2_pooled_3models.png
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
from matplotlib.patches import Patch
import yaml

from src.dataset import SPLITS, model_dir_name

DEFAULT_BASE = "Qwen/Qwen3-4B"
DEFAULT_HARMLESS = "er-abd/harmless-dpo-qwen3-4B-step-735"
DEFAULT_ANTISYC = "er-abd/anti-syc-dpo-qwen3-4B-step-520"

# Same presentation order as the main-results tables.
DATASETS = ["mmlu_pro", "gpqa_diamond", "lsat_ar", "lsat_lr",
            "medxpertqa", "lexam", "codemmlu", "wmdp"]

# Model colors: the same trio as 16_gmean2_barchart.py (there bound to hint
# types), here bound to models; hatches add a second, grayscale-safe cue.
COLOR_BASE = "#92C5E8"      # light blue
COLOR_HARMLESS = "#F6A0B8"  # light pink
COLOR_ANTISYC = "#9ED973"   # light green
EDGE = "#555555"
BAR_W = 0.22
MODEL_OFFSETS = [-0.25, 0.0, 0.25]


def main():
    parser = argparse.ArgumentParser(
        description="Grouped bar chart of pooled g-mean^2 (mean of 3 hint "
                    "types) with paired-bootstrap CIs.")
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

    models = [
        (model_dir_name(args.base), "Base (Qwen3-4B)", None, COLOR_BASE),
        (model_dir_name(args.harmless), "Harmlessness DPO", "///",
         COLOR_HARMLESS),
        (model_dir_name(args.antisyc), "Anti-sycophancy DPO", "..",
         COLOR_ANTISYC),
    ]

    analysis_dir = root / config["paths"]["analysis_dir"]
    pooled = {}
    for model_name, _, _, _ in models:
        p = analysis_dir / model_name
        if args.data != "test":
            p = p / args.data
        p = p / "pooled_summary.json"
        if not p.exists():
            print(f"ERROR: {p} not found — run 15_pooled_summary_tables.py first.")
            sys.exit(1)
        with p.open() as f:
            pooled[model_name] = json.load(f)["datasets"]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#DDDDDD", linewidth=0.8)
    ax.set_ylim(0, 1)
    ax.set_yticks([i / 10 for i in range(11)])

    bar_values = []
    for d_idx, dataset in enumerate(DATASETS):
        for (model_name, _, hatch, color), off in zip(models, MODEL_OFFSETS):
            entry = pooled[model_name].get(dataset, {}).get("gmean2")
            if not entry or entry["estimate"] is None:
                print(f"WARNING: no pooled g-mean^2 for {model_name}/{dataset}")
                continue
            est, (lo, hi) = entry["estimate"], entry["ci95"]
            bar_values.append(est)
            x = d_idx + off
            ax.bar(x, est, width=BAR_W, color=color, hatch=hatch,
                   edgecolor=EDGE, linewidth=0.6, zorder=3)
            ax.errorbar(x, est, yerr=[[est - lo], [hi - est]], fmt="none",
                        ecolor="#333333", elinewidth=1.0, capsize=3,
                        capthick=1.0, zorder=4)
            ax.text(x, hi + 0.012, f"{est:.2f}", ha="center", va="bottom",
                    fontsize=8, color="black", zorder=5, clip_on=False)

    # Reference line: unweighted mean of every bar drawn in this figure.
    avg = sum(bar_values) / len(bar_values)
    ax.axhline(avg, color="red", linestyle="--", linewidth=1.2, zorder=4)

    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels(DATASETS)
    ax.set_xlim(-0.55, len(DATASETS) - 0.45)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("g-mean² (mean of 3 hint types)")
    ax.set_title("Pooled g-mean² per dataset (mean of 3 hint types, "
                 "paired-bootstrap 95% CI) — "
                 + " vs ".join(m[1] for m in models), pad=22)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#999999")

    handles = ([Patch(facecolor=color, edgecolor=EDGE, linewidth=0.8,
                      hatch=hatch, label=lbl) for _, lbl, hatch, color in models]
               + [Line2D([], [], color="red", linestyle="--", linewidth=1.2,
                         label=f"Average ({avg * 100:.1f}%)")])
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.005, 1.0),
              frameon=True, edgecolor="#CCCCCC", fontsize=9)

    out = args.out
    if out is None:
        # RO1 bespoke analysis plots live under extra-plots/, not the main plots/ tree
        plots_dir = ((root / config["paths"]["plots_dir"]).parent
                     / "extra-plots" / "cross-model-comparison")
        if args.data != "test":
            plots_dir = plots_dir / args.data
        plots_dir.mkdir(parents=True, exist_ok=True)
        out = plots_dir / "gmean2_pooled_3models.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
