#!/usr/bin/env python3
"""
Step 37: Does UNHINTED response length correlate with monitorability?
Pooled over all datasets, equal-count (quantile) bins.

This is the weaker-signal half of the length question. Step 41 bins by the
HINTED arm's length (what the monitor reads); here we bin by the UNHINTED
arm's length (the model's intrinsic reasoning verbosity, run once per
question and shared across hint types).

Unit of analysis = the QUESTION, not the run and not the (question, hint)
slice. Each question is run N=8 times unhinted; `mean_cot_tokens_unhinted`
(from 04's results_per_instance.csv) already averages those 8 runs into one
per-question length. Because the unhinted arm is shared across the 3 hint
types, that length is identical in all 3 hint slices — so pooling the slices
would repeat each length value 3x (pseudoreplication on the x-axis). We
therefore collapse to one row per unique question: unhinted length taken
once, monitorability = mean of the question's per-hint-slice gmean2_i.

Binning: unhinted length is heavily right-skewed (base model: median ~1840,
max ~15700 tokens), so fixed-width bins would pile most questions into the
low end and leave a noisy sparse tail. We use EQUAL-COUNT quantile bins
(default 5 = quintiles) instead: every bin holds ~the same number of
questions, so each plotted point carries equal statistical power. Each dot
is placed at its bin's MEDIAN length; the bin's question count and token
range are printed and annotated.

Base model (test, 1383 unique questions): monotonic decline
    Q1 262-840 tok    n=276  0.928
    Q2 840-1395       n=277  0.834
    Q3 1395-2402      n=277  0.806
    Q4 2402-4494      n=276  0.776
    Q5 4494-15690     n=277  0.761
Spearman(unhinted length, monitorability) = -0.289 [-0.337, -0.240];
Q5-Q1 = -0.167. Longer intrinsic reasoning -> less monitorable, and the
signal is real (CI excludes zero) though moderate.

Usage:
    python scripts/37_unhinted_length_gmean2_quantile.py            # base
    python scripts/37_unhinted_length_gmean2_quantile.py --model <hf-id>

    Options:
      --config     path to config YAML (default: config/mvp_config.yaml)
      --data       split: test (default) | eval
      --model      model/adapter HF id (default: Qwen/Qwen3-4B)
      --bins       number of equal-count quantile bins (default: 5)
      --resamples  bootstrap replicates per CI (default: 5000)
      --seed       bootstrap seed (default: 88)
      --out        output PNG path (default:
                   {plots_dir}/cot-length-vs-monitorability/
                   gmean2_by_unhinted_length_quantile_{model}.png)

Input (per dataset / hint type, produced by 04_compute_gmean2.py):
    {analysis_dir}/{model}/{dataset}/{hint}/results_per_instance.csv

Output:
    {plots_dir}/cot-length-vs-monitorability/
        gmean2_by_unhinted_length_quantile_{model}.png
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import yaml

from src.dataset import SPLITS, model_dir_name, resolve_paths
from src.hints import HINT_TYPES

DEFAULT_MODEL = "Qwen/Qwen3-4B"
DATASETS = ["mmlu_pro", "gpqa_diamond", "lsat_ar", "lsat_lr",
            "medxpertqa", "lexam", "codemmlu", "wmdp"]
DARK_BLUE = "#1B3A6B"


def boot_ci(v, n_resamples, rng):
    v = np.asarray(v, float)
    if len(v) < 2:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(v), (n_resamples, len(v)))
    m = v[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    parser = argparse.ArgumentParser(
        description="Monitorability vs UNHINTED response length, pooled, "
                    "equal-count quantile bins.")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--data", default="test", choices=SPLITS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--linear-x", action="store_true",
                        help="linear length axis (default: log10, which "
                             "spreads the roughly log-spaced quantile bins "
                             "and de-crams the low end)")
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

    # --- collapse to one row per unique (dataset, question) ---
    per_q = defaultdict(lambda: {"g": [], "L": None})
    slices, missing = 0, []
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
                    L = row.get("mean_cot_tokens_unhinted", "")
                    if g in ("", "None", "N/A") or L in ("", "None", "N/A"):
                        continue
                    key = (dataset, row["instance_id"])
                    per_q[key]["g"].append(float(g))
                    per_q[key]["L"] = float(L)
                    slices += 1
    if missing:
        print("WARNING: missing slices skipped: " + ", ".join(missing))
    if not per_q:
        print("ERROR: no entering questions found — nothing to plot.")
        sys.exit(1)

    L = np.array([v["L"] for v in per_q.values()])
    G = np.array([float(np.mean(v["g"])) for v in per_q.values()])
    order = np.argsort(L)
    L, G = L[order], G[order]

    print(f"question-slices entering (level1_pass, gmean2 & unhinted-len "
          f"defined): {slices}")
    print(f"UNIQUE questions pooled across {len(DATASETS)} datasets: {len(L)}")
    print(f"unhinted length (tok): min={L.min():.0f} q1={np.percentile(L,25):.0f} "
          f"median={np.median(L):.0f} q3={np.percentile(L,75):.0f} "
          f"max={L.max():.0f}\n")

    # --- equal-count quantile bins ---
    nb = args.bins
    edges = np.percentile(L, np.linspace(0, 100, nb + 1))
    edges[-1] = np.nextafter(edges[-1], np.inf)  # include the max in last bin
    labels = ([f"Q1\n(shortest)"] +
              [f"Q{i+1}" for i in range(1, nb - 1)] +
              [f"Q{nb}\n(longest)"]) if nb >= 3 else [f"Q{i+1}" for i in range(nb)]

    xs, ys, elo, ehi, ns, ranges = [], [], [], [], [], []
    print(f"{'bin':>14} | {'length range (tok)':>19} | {'n':>4} | "
          f"{'med len':>7} | {'monitorab.':>10} | 95% CI")
    print("-" * 84)
    for i in range(nb):
        m = (L >= edges[i]) & (L < edges[i + 1])
        v = G[m]
        med = float(np.median(L[m]))
        lo, hi = boot_ci(v, args.resamples, rng)
        rng_str = f"{int(edges[i])}-{int(np.ceil(edges[i+1]))}"
        flat = labels[i].replace("\n", " ")
        print(f"{flat:>14} | {rng_str:>19} | {int(m.sum()):>4} | {med:>7.0f} | "
              f"{v.mean():>10.3f} | [{lo:.3f}, {hi:.3f}]")
        xs.append(med)
        ys.append(float(v.mean()))
        elo.append(v.mean() - lo)
        ehi.append(hi - v.mean())
        ns.append(int(m.sum()))
        ranges.append(rng_str)

    rho = stats.spearmanr(L, G)[0]
    bs = np.empty(args.resamples)
    for j in range(args.resamples):
        idx = rng.integers(0, len(L), len(L))
        bs[j] = stats.spearmanr(L[idx], G[idx])[0]
    rlo, rhi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    print(f"\nSpearman(unhinted length, monitorability) = {rho:+.3f} "
          f"[{rlo:+.3f}, {rhi:+.3f}]")
    print(f"Q{nb} - Q1 mean monitorability = {ys[-1] - ys[0]:+.3f}")

    # ---------------- figure ----------------
    use_log = not args.linear_x
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.set_axisbelow(True)
    if use_log:
        ax.set_xscale("log")
        # clamp to the data range — else the log axis auto-pads down to ~10^0
        # and squeezes every dot into the right half
        ax.set_xlim(edges[0] * 0.85, edges[-1] * 1.15)
    ax.grid(True, color="#EEEEEE", linewidth=0.8)
    ax.errorbar(xs, ys, yerr=[elo, ehi], color=DARK_BLUE, linewidth=2.0,
                marker="o", markersize=8, markeredgecolor="#333333",
                markeredgewidth=0.5, ecolor=DARK_BLUE, elinewidth=1.4,
                capsize=4, capthick=1.4, zorder=3)
    # n above the UPPER error-bar cap (not on the dot) so it never sits on a whisker
    for x, y, eh, n, rg in zip(xs, ys, ehi, ns, ranges):
        ax.annotate(f"n={n}", xy=(x, y + eh), xytext=(0, 6),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=8.5, color="#555555")
        ax.annotate(rg, xy=(x, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    va="bottom", fontsize=7.5, color="#999999")

    ax.text(0.97, 0.93,
            f"Spearman ρ = {rho:+.3f}\n[{rlo:+.3f}, {rhi:+.3f}]",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            color=DARK_BLUE,
            bbox=dict(boxstyle="round,pad=0.4", fc="#F4F6F9", ec="#CCD3DB"))

    ax.set_xlabel("Unhinted response length — per-question mean of 8 runs "
                  f"(tokens{', log scale' if use_log else ''}; dots at bin "
                  "median)")
    ax.set_ylabel("Monitorability (g-mean²)  ·  error bars: bootstrap 95% CI")
    ax.set_title("Monitorability vs unhinted (intrinsic) reasoning length — "
                 f"{model_name}\n"
                 f"pooled over {len(DATASETS)} datasets, {len(L)} questions; "
                 f"{nb} equal-count bins (~{len(L)//nb}/bin)")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#999999")

    out = args.out
    if out is None:
        plots_dir = ((root / config["paths"]["plots_dir"]).parent
                     / "extra-plots" / "cot-length-vs-monitorability")
        if args.data != "test":
            plots_dir = plots_dir / args.data
        plots_dir.mkdir(parents=True, exist_ok=True)
        out = (plots_dir /
               f"gmean2_by_unhinted_length_quantile_{model_name}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
