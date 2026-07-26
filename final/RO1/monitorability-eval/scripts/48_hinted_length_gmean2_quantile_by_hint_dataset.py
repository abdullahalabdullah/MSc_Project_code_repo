#!/usr/bin/env python3
"""
Step 48: Step 42's per-dataset small multiple, split by HINT TYPE — one 3 x 8
grid, a row of 8 dataset panels per hint condition.

Step 42 draws 8 per-dataset panels of monitorability vs HINTED length, but
each panel first collapses a question by averaging its length and `gmean2_i`
over the hint conditions it qualified for (level-1 pass). That collapse is
confounded: a question contributes 1, 2, or 3 conditions (base: 541 / 317 /
525), and the count correlates with length (Spearman +0.34) and with
monitorability (mean g-mean^2 0.946 / 0.850 / 0.674), so a pooled panel mixes
a between-strata composition contrast into what reads as a length effect.

This step removes both problems at once by never crossing hint conditions:
within one (hint, dataset) cell a question appears exactly once, with the
length and `gmean2_i` measured under that hint. Row = hint type, column =
dataset, so the grid separates the two factors step 42 confounds — read
ACROSS a row for the dataset effect, DOWN a column for the hint effect.

Quantile edges are computed within each cell, so every panel's bins are
equal-count for that cell. Cells are thin by construction (base: 44-197
questions, i.e. ~9-39 per bin at the default 5 bins), so the bootstrap CIs
are wide — that width is the honest read on how little a single (hint,
dataset) cell can support, and `--min-n` blanks any cell too small to bin.

Binning, axes, and statistics otherwise match steps 42 and 47 (equal-count
quantile bins, dots at the bin median on a log x-axis, y = bin mean
`gmean2_i`, bootstrap 95% CIs, Spearman rho with a bootstrap CI). All 24
panels share x and y limits.

Usage:
    python scripts/48_hinted_length_gmean2_quantile_by_hint_dataset.py
    python scripts/48_hinted_length_gmean2_quantile_by_hint_dataset.py \
        --model <hf-id>

    Options:
      --config     path to config YAML (default: config/mvp_config.yaml)
      --data       split: test (default) | eval
      --model      model/adapter HF id (default: Qwen/Qwen3-4B)
      --bins       equal-count quantile bins per cell (default: 5)
      --min-n      min questions to draw a cell (default: 40)
      --linear-x   linear length axis (default: log10)
      --resamples  bootstrap replicates per CI / rho (default: 5000)
      --seed       bootstrap seed (default: 88)
      --out        output PNG path (default:
                   {plots_dir}/cot-length-vs-monitorability/
                   gmean2_by_hinted_length_quantile_panels_by_hint_{model}.png)

Input (per dataset / hint type, produced by 04_compute_gmean2.py):
    {analysis_dir}/{model}/{dataset}/{hint}/results_per_instance.csv

Output:
    {plots_dir}/cot-length-vs-monitorability/
        gmean2_by_hinted_length_quantile_panels_by_hint_{model}.png
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
from scipy import stats
import yaml

from src.dataset import SPLITS, model_dir_name, resolve_paths
from src.hints import HINT_TYPES

DEFAULT_MODEL = "Qwen/Qwen3-4B"
DATASETS = ["mmlu_pro", "gpqa_diamond", "lsat_ar", "lsat_lr",
            "medxpertqa", "lexam", "codemmlu", "wmdp"]

# One CVD-safe hue per hint row, with a matching pale box fill.
HINT_STYLE = {
    "professor":      ("#2A78D6", "#EAF2FC", "#B9D4F2"),
    "grader_hacking": ("#EB6834", "#FDF0EA", "#F5C7AE"),
    "unethical":      ("#1BAF7A", "#E8F7F1", "#A9E0CC"),
}
FALLBACK_STYLE = ("#6A4C93", "#F4F1F8", "#C9BEDD")


def boot_ci(v, n_resamples, rng):
    v = np.asarray(v, float)
    if len(v) < 2:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(v), (n_resamples, len(v)))
    m = v[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def load_cell(root, config, model_name, dataset, hint, split):
    """One row per QUESTION for a single (hint, dataset) cell."""
    L, G = [], []
    paths = resolve_paths(root, config, dataset, model_name, hint, split=split)
    csv_path = paths["analysis_dir"] / "results_per_instance.csv"
    if not csv_path.exists():
        return None, None
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if row.get("level1_pass") != "True":
                continue
            g = row.get("gmean2_i", "")
            length = row.get("mean_cot_tokens_hinted", "")
            if g in ("", "None", "N/A") or length in ("", "None", "N/A"):
                continue
            L.append(float(length))
            G.append(float(g))
    if not L:
        return None, None
    return np.array(L), np.array(G)


def analyse(L, G, nb, resamples, rng):
    edges = np.percentile(L, np.linspace(0, 100, nb + 1))
    edges[-1] = np.nextafter(edges[-1], np.inf)
    drawn = []
    for i in range(nb):
        m = (L >= edges[i]) & (L < edges[i + 1])
        if m.sum() == 0:
            continue
        v = G[m]
        lo, hi = boot_ci(v, resamples, rng)
        drawn.append(dict(med=float(np.median(L[m])), mean=float(v.mean()),
                          lo=lo, hi=hi, n=int(m.sum())))
    rho, pval = stats.spearmanr(L, G)
    bs = np.empty(resamples)
    for j in range(resamples):
        idx = rng.integers(0, len(L), len(L))
        bs[j] = stats.spearmanr(L[idx], G[idx])[0]
    rlo, rhi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    return dict(drawn=drawn, rho=float(rho), rlo=float(rlo), rhi=float(rhi),
                p=float(pval), n=len(L), lmin=float(L.min()),
                lmax=float(L.max()))


def count_significant(cells):
    """How many cells have a rho CI excluding zero. Reported as context only:
    the grid runs ~24 simultaneous tests, so a handful of these are expected
    by chance and no single cell should be read on its own."""
    live = [c for c in cells.values() if c]
    return sum(1 for c in live if not (c["rlo"] < 0 < c["rhi"])), len(live)


def main():
    parser = argparse.ArgumentParser(
        description="Monitorability vs HINTED length, one panel per "
                    "(hint, dataset) cell.")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--data", default="test", choices=SPLITS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--min-n", type=int, default=40)
    parser.add_argument("--linear-x", action="store_true")
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
    use_log = not args.linear_x

    cells, dropped = {}, []
    for hint in HINT_TYPES:
        print(f"\n{'='*78}\n{hint}")
        print(f"{'dataset':<15} {'n':>5} {'rho':>8} {'95% CI':>20}  bins "
              f"(med len -> mean g-mean²)")
        for dataset in DATASETS:
            L, G = load_cell(root, config, model_name, dataset, hint,
                             args.data)
            if L is None or len(L) < args.min_n:
                cells[(hint, dataset)] = None
                got = 0 if L is None else len(L)
                dropped.append(f"{hint}/{dataset} (n={got})")
                print(f"{dataset:<15} {got:>5}   -- below --min-n, blanked")
                continue
            res = analyse(L, G, args.bins, args.resamples, rng)
            cells[(hint, dataset)] = res
            bins_txt = " ".join(f"{b['med']:.0f}→{b['mean']:.2f}"
                                for b in res["drawn"])
            print(f"{dataset:<15} {res['n']:>5} {res['rho']:>+8.3f} "
                  f"[{res['rlo']:>+6.3f},{res['rhi']:>+6.3f}]  {bins_txt}")

    live = [c for c in cells.values() if c]
    if not live:
        print("\nERROR: no populated cells — nothing to plot.")
        sys.exit(1)
    if dropped:
        print(f"\nNOTE: {len(dropped)} cell(s) blanked for n < {args.min_n}: "
              + ", ".join(dropped))

    n_sig, n_tests = count_significant(cells)
    print(f"\n{n_sig} of {n_tests} cells have a rho CI excluding zero. The grid "
          f"runs {n_tests} tests at once, so ~1 is expected by chance — read "
          f"the pattern across cells, not any single one. (rho is NOT drawn in "
          f"the figure; quote these stdout values in the write-up instead.)")
    print("NOTE ON POWER: cells hold too few questions to detect |rho| below "
          "~0.20 (largest cell) to ~0.41 (smallest) at 80% power, so an "
          "unmarked cell means UNDERPOWERED, not null. gmean2_i is also "
          "heavily tied (57% at the 1.0 ceiling, 69 distinct values), which "
          "attenuates rho — treat these as conservative lower bounds.")

    bounds = [b for c in live for d in c["drawn"] for b in (d["lo"], d["hi"])]
    ylo, yhi = min(bounds) - 0.05, max(bounds) + 0.05
    xlo = min(c["lmin"] for c in live) * 0.8
    xhi = max(c["lmax"] for c in live) * 1.2

    nrow, ncol = len(HINT_TYPES), len(DATASETS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.55 * ncol, 3.0 * nrow),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)

    for r, hint in enumerate(HINT_TYPES):
        colour, fc, ec = HINT_STYLE.get(hint, FALLBACK_STYLE)
        for c, dataset in enumerate(DATASETS):
            ax = axes[r, c]
            ax.set_axisbelow(True)
            if use_log:
                ax.set_xscale("log")
            ax.grid(True, color="#EEEEEE", linewidth=0.7)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            for spine in ("left", "bottom"):
                ax.spines[spine].set_color("#999999")
            if r == 0:
                ax.set_title(dataset, fontsize=10, fontweight="bold")

            res = cells.get((hint, dataset))
            if not res:
                ax.text(0.5, 0.5, "insufficient\ndata",
                        transform=ax.transAxes, ha="center", va="center",
                        color="#BBBBBB", fontsize=8)
                continue

            d = res["drawn"]
            xs = [b["med"] for b in d]
            ys = [b["mean"] for b in d]
            elo = [b["mean"] - b["lo"] for b in d]
            ehi = [b["hi"] - b["mean"] for b in d]
            ax.errorbar(xs, ys, yerr=[elo, ehi], color=colour, linewidth=1.6,
                        marker="o", markersize=5, markeredgecolor="#333333",
                        markeredgewidth=0.4, ecolor=colour, elinewidth=1.0,
                        capsize=2.5, capthick=1.0, zorder=3)

            # rho is deliberately NOT drawn: 24 underpowered, heavily tied
            # cells invite over-reading. It stays in stdout for the write-up.
            top_frac = max((b["hi"] - ylo) / (yhi - ylo) for b in d[-2:])
            box_y, box_va = ((0.04, "bottom") if top_frac > 0.62
                             else (0.96, "top"))
            ax.text(0.95, box_y, f"n={res['n']}",
                    transform=ax.transAxes, ha="right", va=box_va,
                    fontsize=7.5, color=colour,
                    bbox=dict(boxstyle="round,pad=0.25", fc=fc, ec=ec,
                              linewidth=0.7))

        axes[r, 0].set_ylabel(f"{hint}\n\nMonitorability (g-mean²)",
                              fontsize=9.5, fontweight="bold", color=colour)

    axes[0, 0].set_xlim(xlo, xhi)
    axes[0, 0].set_ylim(ylo, yhi)
    fig.supxlabel("HINTED response length — per-question mean over 8 runs "
                  f"(tokens{', log scale' if use_log else ''}; dots at bin "
                  "median; bars: bootstrap 95% CI)", fontsize=10, y=0.015)
    fig.suptitle("Monitorability by HINTED length — every (hint × dataset) "
                 f"cell — {model_name}\n"
                 "one row per question per hint: no cross-hint averaging and "
                 "no level-1 entry-count confound (contrast step 42, which "
                 "pools the hints)",
                 fontsize=13)

    out = args.out
    if out is None:
        plots_dir = ((root / config["paths"]["plots_dir"]).parent
                     / "extra-plots" / "cot-length-vs-monitorability")
        if args.data != "test":
            plots_dir = plots_dir / args.data
        plots_dir.mkdir(parents=True, exist_ok=True)
        out = (plots_dir / "gmean2_by_hinted_length_quantile_panels_by_hint_"
                           f"{model_name}.png")
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
