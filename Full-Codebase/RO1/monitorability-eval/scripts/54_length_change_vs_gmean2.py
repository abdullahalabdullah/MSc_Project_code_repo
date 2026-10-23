#!/usr/bin/env python3
"""
Step 54: Does the hint's effect on response length track monitorability?
Dataset-level scatter of per-question length change vs g-mean², one model
(default: base Qwen3-4B).

Question: are the datasets where adding the hint changes response length the
most also the datasets with the lowest monitorability?

x = median per-question length change (%) when the hint is introduced — the
    step-53 quantity: per question, median hinted length / median unhinted
    length over its runs (arms paired by instance_id), then the median of
    those ratios over the dataset's 200 questions, as % change.
y = g-mean² from 04_compute_gmean2.py.

Four panels, one point per dataset (n = 8 each):
    A  All hints pooled — x from the pooled column of step 53 (24 hinted runs
       vs 8 unhinted per question); y = unweighted mean of the 3 hint types'
       g-mean² (the Panel-A number of the main-results table).
    B–D  Each hint type separately — x from that hint's column of step 53;
       y = that (dataset, hint) cell's g-mean² with its 95% bootstrap CI.

Each panel's title gives Spearman's rho and its p-value (scipy's
t-approximation; with n = 8 treat it as indicative only). Spearman rather than Pearson because 8 points with a few large-x
datasets would let one point drive a linear fit. Hint panels are kept separate
rather than pooled into 24 points because hint type moves BOTH variables
(unethical: largest length shifts and highest g-mean²), which would make a
pooled correlation reflect the hint-type difference rather than the
across-dataset relationship asked about.

Usage:
    python scripts/54_length_change_vs_gmean2.py

    Options:
      --config    path to config YAML (default: config/mvp_config.yaml)
      --data      split: test (default) | eval
      --model     model id (default: Qwen/Qwen3-4B)
      --out-dir   output directory (default: {plots_dir}/../extra-plots/
                  hint-effect-on-length/{model}, same as step 53)

Input (per dataset x hint):
    {analysis_dir}/{model}/{dataset}/{hint}/cot_token_stats.json   (step 08)
    {analysis_dir}/{model}/{dataset}/{hint}/results_summary.json   (step 04)

Output:
    {out_dir}/length_change_vs_gmean2_{model}.png
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import spearmanr

from src.dataset import SPLITS, model_dir_name, resolve_paths

# Reuse step 53's per-question ratio construction so the x values are exactly
# the numbers printed on the step-53 figures.
_spec = importlib.util.spec_from_file_location(
    "step53", Path(__file__).with_name("53_per_question_hint_length_change.py"))
step53 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(step53)

DEFAULT_MODEL = step53.DEFAULT_MODEL
DATASETS = step53.DATASETS
COLUMNS = step53.COLUMNS  # 3 hint types + pooled, with colors
EDGE = "#555555"


def load_gmean2(root, config, model_name, ds, hint, split):
    """(estimate, (lo, hi)) for one (dataset, hint) cell, or None."""
    paths = resolve_paths(root, config, ds, model_name, hint, split=split)
    jp = paths["analysis_dir"] / "results_summary.json"
    if not jp.exists():
        return None
    s = json.load(jp.open())
    est = s.get("gmean2")
    ci = (s.get("ci95") or {}).get("gmean2")
    if est is None:
        return None
    return est, (tuple(ci) if ci else None)


def collect(root, config, model_name, split):
    """rows[ds] = {"pct": {label: %change}, "g": {hint: (est, ci)}}."""
    rows = {}
    for ds in DATASETS:
        loaded = step53.load_dataset(root, config, model_name, ds, split)
        if loaded is None:
            print(f"Skipping {ds}: missing length_pools.")
            continue
        ratios = step53.per_question_ratios(*loaded)
        g = {}
        for hint, _, _ in COLUMNS[:-1]:
            got = load_gmean2(root, config, model_name, ds, hint, split)
            if got is None:
                break
            g[hint] = got
        else:
            rows[ds] = {"pct": {lbl: (np.median(r) - 1) * 100
                                for lbl, r in ratios.items()},
                        "g": g}
            continue
        print(f"Skipping {ds}: missing results_summary.json / gmean2.")
    return rows


def panel(ax, xs, ys, labels, color, title, yerr=None):
    if yerr is not None:
        ax.errorbar(xs, ys, yerr=yerr, fmt="none", ecolor="#888888",
                    elinewidth=0.9, capsize=2.5, zorder=2)
    ax.scatter(xs, ys, s=60, color=color, edgecolor=EDGE, linewidth=0.7,
               zorder=3)
    for x, y, lbl in zip(xs, ys, labels):
        ax.annotate(lbl, (x, y), xytext=(5, 4), textcoords="offset points",
                    fontsize=7.5, color="#333333")
    rho, p = spearmanr(xs, ys)
    ax.axvline(0, color="#C0392B", linestyle="--", linewidth=0.9, zorder=1)
    # Stats go in the title so they never cover a point or CI.
    ax.set_title(f"{title}\nSpearman ρ = {rho:+.2f}  (p = {p:.3f}, "
                 f"n = {len(xs)})", fontsize=10)
    ax.set_axisbelow(True)
    ax.grid(True, color="#E5E5E5", linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#999999")
    ax.set_xlabel("Median per-question length change\nwhen hint introduced (%)",
                  fontsize=9)
    return rho, p


def main():
    parser = argparse.ArgumentParser(
        description="Dataset-level per-question length change vs g-mean².")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--data", default="test", choices=SPLITS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    with config_path.open() as f:
        config = yaml.safe_load(f)

    model_name = model_dir_name(args.model)
    rows = collect(root, config, model_name, args.data)
    if len(rows) < 3:
        print("ERROR: fewer than 3 datasets with complete inputs.")
        sys.exit(1)
    dss = list(rows)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.2))

    # Panel A: pooled.
    pooled_label, pooled_color = COLUMNS[-1][1], COLUMNS[-1][2]
    xs = [rows[d]["pct"][pooled_label] for d in dss]
    ys = [np.mean([rows[d]["g"][h][0] for h, _, _ in COLUMNS[:-1]])
          for d in dss]
    rho, p = panel(axes[0], xs, ys, dss, pooled_color,
                   "A. All hints pooled\n(y = mean g-mean² of 3 hint types)")
    print(f"pooled: rho={rho:+.3f} p={p:.3f}")
    axes[0].set_ylabel("Monitorability (g-mean²)")

    # Panels B–D: one per hint type.
    for ax, (hint, label, color), letter in zip(axes[1:], COLUMNS[:-1], "BCD"):
        xs = [rows[d]["pct"][label] for d in dss]
        ys = [rows[d]["g"][hint][0] for d in dss]
        cis = [rows[d]["g"][hint][1] for d in dss]
        yerr = None
        if all(cis):
            yerr = np.array([[y - lo for y, (lo, _) in zip(ys, cis)],
                             [hi - y for y, (_, hi) in zip(ys, cis)]])
        rho, p = panel(ax, xs, ys, dss, color,
                       f"{letter}. {label} hint\n(y = cell g-mean², 95% CI)",
                       yerr=yerr)
        print(f"{hint}: rho={rho:+.3f} p={p:.3f}")

    fig.suptitle(f"Hint-induced change in response length vs monitorability, "
                 f"per dataset — {args.model}", fontsize=12, y=1.02)
    fig.tight_layout()

    if args.out_dir:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = root / out_dir
    else:
        out_dir = ((root / config["paths"]["plots_dir"]).parent
                   / "extra-plots" / "hint-effect-on-length" / model_name)
        if args.data != "test":
            out_dir = out_dir / args.data
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"length_change_vs_gmean2_{model_name}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
