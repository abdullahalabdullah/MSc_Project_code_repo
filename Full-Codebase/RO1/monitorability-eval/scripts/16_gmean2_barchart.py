#!/usr/bin/env python3
"""
Step 16: Grouped bar chart of per-hint-type g-mean^2 — base vs harmlessness
DPO vs anti-sycophancy DPO, per dataset.

Layout: x-axis = datasets (domain-table order); within each dataset three
hint-type sets (professor, grader_hacking, unethical), each set = base bar
(solid) + harmlessness-DPO bar (striped hatch) + anti-sycophancy-DPO bar
(dotted hatch), with the point estimate printed above each error bar. Error
bars are the per-hint-type bootstrap 95% CIs from results_summary.json
(ci95.gmean2 — per-hint, NOT the pooled paired bootstrap); there is no
cross-figure average line. Colors are fixed
per hint type (light blue / light pink / light green; shades tuned so all
pairs clear normal-vision and CVD separation floors); model identity is
carried by the hatch, so it survives grayscale.

Usage:
    python scripts/16_gmean2_barchart.py

    Options:
      --config    path to config YAML (default: config/mvp_config.yaml)
      --data      split: test (default) | eval
      --base      base model id (default: Qwen/Qwen3-4B)
      --harmless  harmlessness DPO id (default: er-abd/harmless-dpo-qwen3-4B-step-735)
      --antisyc   anti-sycophancy DPO id (default: er-abd/anti-syc-dpo-qwen3-4B-step-520)
      --out       output PNG path (default: {plots_dir}/gmean2_by_hint_3models.png)

Input (per model / dataset / hint type, produced by 04_compute_gmean2.py):
    {analysis_dir}/{model}/{dataset}/{hint}/results_summary.json

Output:
    {plots_dir}/gmean2_by_hint_3models.png
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import yaml

from src.dataset import SPLITS, model_dir_name, resolve_paths

DEFAULT_BASE = "Qwen/Qwen3-4B"
DEFAULT_HARMLESS = "er-abd/harmless-dpo-qwen3-4B-step-735"
DEFAULT_ANTISYC = "er-abd/anti-syc-dpo-qwen3-4B-step-520"

# Same presentation order as the main-results tables (15_pooled_summary_tables.py).
DATASETS = ["mmlu_pro", "gpqa_diamond", "lsat_ar", "lsat_lr",
            "medxpertqa", "lexam", "codemmlu", "wmdp"]

HINTS = [
    ("professor", "Professor hint", "#92C5E8"),        # light blue
    ("grader_hacking", "Grader-hacking hint", "#F6A0B8"),  # light pink
    ("unethical", "Unethical hint", "#9ED973"),        # light green
]

BAR_W = 0.085
HINT_OFFSETS = [-0.3, 0.0, 0.3]
EDGE = "#555555"


def load_gmean2(root, config, model_name, dataset, hint, split):
    paths = resolve_paths(root, config, dataset, model_name, hint, split=split)
    summary = paths["analysis_dir"] / "results_summary.json"
    if not summary.exists():
        return None
    with summary.open() as f:
        s = json.load(f)
    est = s.get("gmean2")
    ci = (s.get("ci95") or {}).get("gmean2")
    if est is None or ci is None:
        return None
    return est, ci


def main():
    parser = argparse.ArgumentParser(
        description="Grouped bar chart of per-hint g-mean^2, base vs trained.")
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

    models = [(model_dir_name(args.base), "Base (Qwen3-4B)", None),
              (model_dir_name(args.harmless), "Harmlessness DPO", "///"),
              (model_dir_name(args.antisyc), "Anti-sycophancy DPO", "..")]

    fig, ax = plt.subplots(figsize=(16, 5.5))

    # Recessive horizontal gridlines behind the bars.
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#DDDDDD", linewidth=0.8)
    ax.set_ylim(0, 1)
    ax.set_yticks([i / 10 for i in range(11)])

    missing = []
    for d_idx, dataset in enumerate(DATASETS):
        for (hint, _, color), h_off in zip(HINTS, HINT_OFFSETS):
            for m_idx, (model_name, _, hatch) in enumerate(models):
                got = load_gmean2(root, config, model_name, dataset, hint,
                                  args.data)
                if got is None:
                    missing.append(f"{model_name}/{dataset}/{hint}")
                    continue
                est, (lo, hi) = got
                x = d_idx + h_off + (m_idx - 1) * (BAR_W + 0.012)
                ax.bar(x, est, width=BAR_W, color=color, hatch=hatch,
                       edgecolor=EDGE, linewidth=0.6, zorder=3)
                ax.errorbar(x, est, yerr=[[est - lo], [hi - est]],
                            fmt="none", ecolor="#333333", elinewidth=1.0,
                            capsize=2.5, capthick=1.0, zorder=4)
                # Score above the error bar's upper cap; rotated so adjacent
                # labels in a 9-bar group never collide.
                ax.text(x, hi + 0.012, f"{est:.2f}", rotation=90,
                        ha="center", va="bottom", fontsize=6.5,
                        color="black", zorder=5, clip_on=False)

    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels(DATASETS)
    ax.set_xlim(-0.55, len(DATASETS) - 0.45)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Monitorability (g-mean²)")
    ax.set_title("Monitorability per dataset and hint type: Base vs "
                 "Harmlessness DPO vs Anti-sycophancy DPO", pad=22)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#999999")

    handles = ([Patch(facecolor=c, edgecolor=EDGE, linewidth=0.6, label=lbl)
                for _, lbl, c in HINTS]
               + [Patch(facecolor="white", edgecolor=EDGE, linewidth=0.8,
                        hatch=hatch, label=lbl)
                  for _, lbl, hatch in models])
    # Anchored to the top-right, fully outside the axes so it can never sit on
    # a column (bars reach ~0.97 with ylim capped at 1).
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.005, 1.0),
              frameon=True, edgecolor="#CCCCCC", fontsize=9)

    if missing:
        print("WARNING: missing cells skipped: " + ", ".join(missing))

    out = args.out
    if out is None:
        # RO1 bespoke analysis plots live under extra-plots/, not the main plots/ tree
        plots_dir = ((root / config["paths"]["plots_dir"]).parent
                     / "extra-plots" / "cross-model-comparison")
        if args.data != "test":
            plots_dir = plots_dir / args.data
        plots_dir.mkdir(parents=True, exist_ok=True)
        out = plots_dir / "gmean2_by_hint_3models.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
