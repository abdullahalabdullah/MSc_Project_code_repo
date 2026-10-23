#!/usr/bin/env python3
"""
Step 30: The task-performance-vs-monitorability figure — two panels, one
per level of aggregation, for ONE model.

Steps 27 and 29 each answer half of "does task performance predict
monitorability?", and they answer it OPPOSITE ways. That contrast is the
finding, so this step draws both on one canvas for the write-up:

  (a) QUESTION level — x = a question's unhinted accuracy acc_u_i (k/8
      over its N unhinted runs, 9 levels), y = the mean per-question
      g-mean^2 at that level with a percentile-bootstrap 95% CI.
      Monitorability is FLAT across 0/8..7/8 and jumps only at 8/8: a
      reliability THRESHOLD, not a graded accuracy trend.

  (b) SLICE level — x = a (dataset x hint type) slice's mean unhinted
      accuracy, y = the slice's g-mean^2, 24 points with an OLS fit.
      The fit is flat: between-dataset difficulty is uncorrelated with
      monitorability (lsat_lr = most accurate, least monitorable).

Together: solution reliability at the question level is a real (if small)
correlate of monitorability; task difficulty at the dataset level is not.

Stdout additionally prints a CoT-LENGTH ROBUSTNESS CHECK — the
always-correct vs rest contrast recomputed within unhinted-CoT-length
terciles — because the natural objection to panel (a) is that
always-correct questions are simply the short-CoT ones, and this project
documents a separate length effect (steps 37/39).

Usage:
    python scripts/30_acc_monitorability_panel.py                  # base
    python scripts/30_acc_monitorability_panel.py --model <hf-id>

    Options:
      --config     path to config YAML (default: config/mvp_config.yaml)
      --data       split: test (default) | eval
      --model      model/adapter HF id (default: Qwen/Qwen3-4B)
      --resamples  bootstrap replicates per CI (default: 5000)
      --seed       bootstrap seed (default: 88)
      --out        output PNG path
                   (default: {plots_dir}/acc_vs_monitorability_panel_{model}.png)

Input (per dataset / hint type, produced by 04_compute_gmean2.py):
    {analysis_dir}/{model}/{dataset}/{hint}/results_per_instance.csv   (panel a)
    {analysis_dir}/{model}/{dataset}/{hint}/results_summary.json       (panel b)

Output:
    {plots_dir}/acc_vs_monitorability_panel_{model}.png
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

# Type sizes match step 51 (the length panel): set for the PRINTED size,
# since this canvas lands at ~6.3in of LaTeX \textwidth and is scaled by
# ~0.57 in the PDF.
FS_TITLE, FS_LABEL, FS_TICK, FS_N, FS_BOX, FS_LEG = 15, 12, 13, 10, 11, 10

DEFAULT_MODEL = "Qwen/Qwen3-4B"
N_RUNS = 8

# Domain-table order with the validated 8-slot categorical palette
# (same binding as steps 24 / 29).
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

HINT_MARKERS = [
    ("professor", "o", "Professor hint"),
    ("grader_hacking", "s", "Grader-hacking hint"),
    ("unethical", "^", "Unethical hint"),
]

LINE_COLOR = "#1B3A6B"
DOT_COLOR = "#1B3A6B"
THRESHOLD_COLOR = "#C1462F"


def boot_ci(values, resamples, rng):
    """Percentile-bootstrap 95% CI of the mean."""
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(v), (resamples, len(v)))
    means = v[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def boot_diff_ci(a, b, resamples, rng):
    """Percentile-bootstrap 95% CI of mean(a) - mean(b), independent groups."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    ia = rng.integers(0, len(a), (resamples, len(a)))
    ib = rng.integers(0, len(b), (resamples, len(b)))
    d = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def load_questions(root, config, model_name, split):
    """Every entering question, pooled over datasets x hint types."""
    rows, missing = [], []
    for dataset, _ in DATASET_COLORS:
        for hint in HINT_TYPES:
            paths = resolve_paths(root, config, dataset, model_name, hint,
                                  split=split)
            csv_path = paths["analysis_dir"] / "results_per_instance.csv"
            if not csv_path.exists():
                missing.append(f"{dataset}/{hint}")
                continue
            with csv_path.open() as f:
                for r in csv.DictReader(f):
                    g, a = r.get("gmean2_i", ""), r.get("acc_u_i", "")
                    if g in ("", "None") or a in ("", "None"):
                        continue
                    tok = r.get("mean_cot_tokens_unhinted", "")
                    rows.append({
                        "dataset": dataset,
                        "hint": hint,
                        "acc": float(a),
                        "gmean2": float(g),
                        "cot": float(tok) if tok not in ("", "None") else None,
                    })
    return rows, missing


def load_slices(root, config, model_name, split):
    pts, missing = [], []
    for dataset, _ in DATASET_COLORS:
        for hint in HINT_TYPES:
            paths = resolve_paths(root, config, dataset, model_name, hint,
                                  split=split)
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
    return pts, missing


def main():
    parser = argparse.ArgumentParser(
        description="Two-panel task-performance vs monitorability figure.")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--data", default="test", choices=SPLITS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=88)
    parser.add_argument("--show-threshold-delta", action="store_true",
                        help="annotate panel (a) with the always-correct vs "
                             "rest contrast. Off by default: the contrast is "
                             "the hand-off to the consistency analysis, not "
                             "part of the accuracy result this panel reports. "
                             "The numbers are printed to stdout either way.")
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

    questions, miss_q = load_questions(root, config, model_name, args.data)
    slices, miss_s = load_slices(root, config, model_name, args.data)
    for label, miss in (("per-instance", miss_q), ("summary", miss_s)):
        if miss:
            print(f"WARNING: missing {label} slices skipped: "
                  + ", ".join(miss))
    if not questions or len(slices) < 3:
        print("ERROR: not enough data — nothing to plot.")
        sys.exit(1)

    acc = np.array([q["acc"] for q in questions])
    gm = np.array([q["gmean2"] for q in questions])
    lvl = np.round(acc * N_RUNS).astype(int)

    # ---------------- panel (a): question level ----------------
    print(f"(a) QUESTION LEVEL — {len(questions)} entering questions "
          f"(8 datasets x 3 hint types)")
    print(f"{'acc level':>10} | {'n':>5} | {'g-mean2':>8} | 95% CI")
    print("-" * 52)
    lv_x, lv_mean, lv_lo, lv_hi, lv_n = [], [], [], [], []
    for k in range(N_RUNS + 1):
        v = gm[lvl == k]
        if len(v) == 0:
            continue
        lo, hi = boot_ci(v, args.resamples, rng)
        lv_x.append(k / N_RUNS)
        lv_mean.append(v.mean())
        lv_lo.append(lo)
        lv_hi.append(hi)
        lv_n.append(len(v))
        print(f"{k:>8}/8 | {len(v):>5} | {v.mean():>8.3f} | "
              f"[{lo:.3f}, {hi:.3f}]")

    r_q, p_q = stats.pearsonr(acc, gm)
    rho_q, prho_q = stats.spearmanr(acc, gm)
    print(f"\nper-question Pearson r = {r_q:+.3f} (p = {p_q:.3g}); "
          f"Spearman rho = {rho_q:+.3f} (p = {prho_q:.3g})")

    always = gm[lvl == N_RUNS]
    rest = gm[lvl < N_RUNS]
    d_thr = always.mean() - rest.mean()
    lo_thr, hi_thr = boot_diff_ci(always, rest, args.resamples, rng)
    print(f"always-correct (n={len(always)}, mean {always.mean():.3f}) vs "
          f"rest (n={len(rest)}, mean {rest.mean():.3f}): "
          f"Delta = {d_thr:+.3f}, 95% CI [{lo_thr:+.3f}, {hi_thr:+.3f}]")

    partial = gm[(lvl > 0) & (lvl < N_RUNS)]
    never = gm[lvl == 0]
    d_pn = partial.mean() - never.mean()
    lo_pn, hi_pn = boot_diff_ci(partial, never, args.resamples, rng)
    print(f"partial 1-7/8 (n={len(partial)}, mean {partial.mean():.3f}) vs "
          f"never 0/8 (n={len(never)}, mean {never.mean():.3f}): "
          f"Delta = {d_pn:+.3f}, 95% CI [{lo_pn:+.3f}, {hi_pn:+.3f}] "
          f"— the flat region")

    # ---------------- panel (b): slice level ----------------
    print(f"\n(b) SLICE LEVEL — {len(slices)} (dataset x hint) slices")
    xs = np.array([p[2] for p in slices])
    ys = np.array([p[3] for p in slices])
    r_s, p_s = stats.pearsonr(xs, ys)
    rho_s, prho_s = stats.spearmanr(xs, ys)
    print(f"slice-level Pearson r = {r_s:+.3f} (p = {p_s:.3g}); "
          f"Spearman rho = {rho_s:+.3f} (p = {prho_s:.3g})")

    # ---------------- CoT-length robustness ----------------
    print("\nCoT-LENGTH ROBUSTNESS — always-correct vs rest, within "
          "unhinted-CoT-length terciles")
    with_cot = [q for q in questions if q["cot"] is not None]
    if len(with_cot) >= 30:
        cot = np.array([q["cot"] for q in with_cot])
        g2 = np.array([q["gmean2"] for q in with_cot])
        l2 = np.round(np.array([q["acc"] for q in with_cot])
                      * N_RUNS).astype(int)
        t1, t2 = np.percentile(cot, [100 / 3, 200 / 3])
        bands = [("short", cot <= t1),
                 ("medium", (cot > t1) & (cot <= t2)),
                 ("long", cot > t2)]
        print(f"{'tercile':>8} | {'n alw':>6} | {'n rest':>6} | "
              f"{'mean alw':>8} | {'mean rest':>9} | {'Delta':>7} | 95% CI")
        print("-" * 78)
        for name, mask in bands:
            a_v = g2[mask & (l2 == N_RUNS)]
            r_v = g2[mask & (l2 < N_RUNS)]
            if len(a_v) < 2 or len(r_v) < 2:
                print(f"{name:>8} | too few questions — skipped")
                continue
            d = a_v.mean() - r_v.mean()
            lo, hi = boot_diff_ci(a_v, r_v, args.resamples, rng)
            print(f"{name:>8} | {len(a_v):>6} | {len(r_v):>6} | "
                  f"{a_v.mean():>8.3f} | {r_v.mean():>9.3f} | "
                  f"{d:>+7.3f} | [{lo:+.3f}, {hi:+.3f}]")
        print(f"(tercile cuts at {t1:.0f} and {t2:.0f} unhinted CoT tokens)")
    else:
        print("insufficient CoT-token data — skipped")

    # ---------------- figure ----------------
    # Panel order: the configuration-level view is drawn on the LEFT and the
    # question-level view on the RIGHT, so the (a)/(b) labels follow reading
    # order. axa/axb below still name the CONTENT (a = question level,
    # b = configuration level); only their positions on the canvas swap.
    fig, (axb, axa) = plt.subplots(1, 2, figsize=(11, 4.8))

    # (a)
    axa.set_axisbelow(True)
    axa.grid(True, color="#DDDDDD", linewidth=0.8)
    lv_x = np.array(lv_x)
    lo_err = np.array(lv_mean) - np.array(lv_lo)
    hi_err = np.array(lv_hi) - np.array(lv_mean)
    axa.errorbar(lv_x, lv_mean, yerr=[lo_err, hi_err], fmt="o",
                 color=DOT_COLOR, ecolor=DOT_COLOR, elinewidth=1.4,
                 capsize=3.5, markersize=7, zorder=3)
    # Reference line for the flat region. It spans ONLY the levels it is
    # computed from (0/8 to 7/8) and stops short of the 8/8 level, so the
    # exclusion is visible in the geometry: a line running under the excluded
    # point reads as a fit that point fails, rather than as a baseline the
    # point is being compared against.
    flat_mean = float(rest.mean())
    flat_xmax = (N_RUNS - 1) / N_RUNS + 0.03   # just past the 7/8 level
    axa.hlines(flat_mean, xmin=-0.04, xmax=flat_xmax, color=THRESHOLD_COLOR,
               linestyle=":", linewidth=1.4, zorder=2)
    # flat-region label: upper-left corner. It cannot sit under the line any
    # more — now that the line stops at 7/8 the label would lie over the data
    # and be crossed by the error bars.
    axa.text(0.03, 0.97,
             f"dotted line: mean of 0/8 to 7/8 = {flat_mean:.3f}",
             transform=axa.transAxes, fontsize=FS_BOX, color=THRESHOLD_COLOR,
             va="top", ha="left",
             bbox=dict(facecolor="white", alpha=0.85, edgecolor="none",
                       pad=2))
    # threshold contrast: upper-left, the one empty corner of the panel
    if args.show_threshold_delta:
        axa.text(0.03, 0.97,
                 f"always-correct vs rest:\nΔ = {d_thr:+.3f} "
                 f"[{lo_thr:+.3f}, {hi_thr:+.3f}]",
                 transform=axa.transAxes, fontsize=FS_BOX, color=THRESHOLD_COLOR,
                 va="top", ha="left",
                 bbox=dict(facecolor="white", alpha=0.85, edgecolor="none",
                           pad=2))
    axa.set_ylim(min(lv_lo) - 0.02, max(lv_hi) + 0.045)
    for x, n in zip(lv_x, lv_n):
        axa.annotate(f"{n}", xy=(x, 0), xycoords=("data", "axes fraction"),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=FS_N, color="#333333")
    axa.set_xlim(-0.06, 1.06)
    axa.set_xticks([k / N_RUNS for k in range(N_RUNS + 1)])
    axa.set_xticklabels([f"{k}/8" for k in range(N_RUNS + 1)],
                    fontsize=FS_TICK)
    axa.set_xlabel("Unhinted accuracy per question (correct k/8 runs)",
                   fontsize=FS_LABEL)
    # y-label is drawn ONCE, on the left-hand panel: both panels plot the same
    # quantity. The SCALES stay independent, though — forcing a shared range
    # would squash the question panel's 8/8 step into the configuration
    # panel's much wider spread. Each panel keeps its own ticks, so the
    # single label is not ambiguous.
    # No correlation coefficient in this subtitle. The relationship here is a
    # step, and any single coefficient (r = +0.13, rho = +0.14) compresses
    # eight flat levels plus one jump into a number that reads as a gradient —
    # the opposite of what the panel shows. Both are printed to stdout.
    axa.set_title("(b) Question level", fontsize=FS_TITLE)
    for spine in ("top", "right"):
        axa.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axa.spines[spine].set_color("#999999")

    # (b)
    axb.set_axisbelow(True)
    axb.grid(True, color="#DDDDDD", linewidth=0.8)
    color_of = dict(DATASET_COLORS)
    marker_of = {h: m for h, m, _ in HINT_MARKERS}
    for dataset, hint, a_, g_ in slices:
        axb.scatter(a_, g_, s=70, color=color_of[dataset],
                    marker=marker_of[hint], edgecolors="#333333",
                    linewidths=0.6, zorder=3)
    # No OLS fit and no on-panel Pearson r: the scatter is shown as-is, and the
    # slice-level correlation is reported to stdout (and in the write-up text)
    # rather than drawn over the points.
    axb.set_xlim(0, 1)
    axb.set_xlabel("Mean unhinted task accuracy\n"
                   "of the configuration", fontsize=FS_LABEL)
    axb.set_ylabel("Monitorability (g-mean²)", fontsize=FS_LABEL)
    axb.set_title("(a) Configuration level", fontsize=FS_TITLE)
    axb.tick_params(labelsize=FS_TICK)
    for spine in ("top", "right"):
        axb.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axb.spines[spine].set_color("#999999")

    handles = ([Line2D([], [], marker="o", linestyle="none", color=c,
                       markeredgecolor="#333333", markeredgewidth=0.6,
                       markersize=8, label=ds)
                for ds, c in DATASET_COLORS]
               + [Line2D([], [], marker=m, linestyle="none",
                         color="#888888", markeredgecolor="#333333",
                         markeredgewidth=0.6, markersize=8, label=lbl)
                  for _, m, lbl in HINT_MARKERS])
    # The legend describes the configuration-level markers but is anchored to
    # the RIGHT-hand axes: that keeps it outside the figure's right edge. On the
    # (now left-hand) configuration panel it would sit in the gap between the
    # two panels and overlap the question-level axis.
    axa.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.005, 1.0),
               frameon=True, edgecolor="#CCCCCC", fontsize=FS_LEG)

    # No figure-level title: the caption carries the model.

    out = args.out
    if out is None:
        # RO1 bespoke analysis plots live under extra-plots/, not the main plots/ tree
        plots_dir = ((root / config["paths"]["plots_dir"]).parent
                     / "extra-plots" / "task-performance-vs-monitorability")
        if args.data != "test":
            plots_dir = plots_dir / args.data
        plots_dir.mkdir(parents=True, exist_ok=True)
        out = plots_dir / f"acc_vs_monitorability_panel_{model_name}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
