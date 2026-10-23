#!/usr/bin/env python3
"""
Step 51: Two-panel write-up figure combining steps 37 and 41 — monitorability
against UNHINTED length (a) beside monitorability against HINTED length (b).

The pair is the whole length argument in one canvas. Panel (a) is the model's
intrinsic reasoning length, measured on the arm the monitor never sees; panel
(b) is the length of the transcript the monitor actually reads. A mechanism in
which "longer transcripts give the monitor more to find" has to show up in (b).
It does not: (a) declines monotonically while (b) is flat and non-monotone.

Both panels are built on the SAME 1383 entering questions with the SAME
y-values (each question's mean gmean2_i over its entering hint slices), so the
only thing that changes between them is which arm supplies the x-variable.
That parity is what makes the comparison fair, and it is why this script
computes both series in one pass rather than stitching two PNGs together.

Axes are SHARED across the panels — same y-range and same log x-range. This
matters: auto-scaled separately (as the standalone figures 37/41 are), panel
(b)'s noise gets stretched to fill the same vertical space as panel (a)'s real
0.167 decline, and the flat result looks as dramatic as the sloped one. On a
common axis (b) visibly flattens. The shared x-range also shows, for free,
that the hinted arm simply runs longer.

Binning matches 37/41: 5 equal-count quantile bins computed WITHIN each panel
(so each dot carries ~276 questions), dots at the bin median, percentile
bootstrap 95% CI, n above each upper cap and the bin's token range under the
axis. Spearman rho with a bootstrap CI is computed on the raw per-question
values, not on the five plotted points.

Usage:
    python scripts/51_length_gmean2_quantile_panel.py            # base model
    python scripts/51_length_gmean2_quantile_panel.py --model <hf-id>

    Options:
      --config     path to config YAML (default: config/mvp_config.yaml)
      --data       split: test (default) | eval
      --model      model/adapter HF id (default: Qwen/Qwen3-4B)
      --bins       equal-count quantile bins per panel (default: 5)
      --resamples  bootstrap replicates per CI / rho (default: 5000)
      --seed       bootstrap seed (default: 88)
      --out        output PNG path (default:
                   {plots_dir}/extra-plots/cot-length-vs-monitorability/
                   length_gmean2_quantile_panel_{model}.png)

Input (per dataset / hint type, produced by 04_compute_gmean2.py):
    {analysis_dir}/{model}/{dataset}/{hint}/results_per_instance.csv

Output:
    {plots_dir}/extra-plots/cot-length-vs-monitorability/
        length_gmean2_quantile_panel_{model}.png
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
from matplotlib.ticker import FixedLocator, NullLocator
import numpy as np
from scipy import stats
import yaml

from src.dataset import SPLITS, model_dir_name, resolve_paths
from src.hints import HINT_TYPES

DEFAULT_MODEL = "Qwen/Qwen3-4B"
DATASETS = ["mmlu_pro", "gpqa_diamond", "lsat_ar", "lsat_lr",
            "medxpertqa", "lexam", "codemmlu", "wmdp"]

DARK_BLUE = "#1B3A6B"   # step 37's colour
PLUM = "#5B4B8A"        # step 41's colour
TICKS = (250, 500, 1000, 2000, 4000, 8000, 16000)

# Type sizes are set for the PRINTED size, not the canvas. The figure is
# ~10in wide and lands at ~6.3in of LaTeX \textwidth, so everything on it is
# scaled by ~0.63 in the PDF: a 14pt axis label renders at ~9pt. Keep the
# canvas near its printed width and the fonts large, or the reader gets 4pt
# ticks (which is what a 14in canvas at \textwidth produced).
FS_TITLE, FS_LABEL, FS_TICK, FS_N, FS_BOX = 15, 12, 13, 12, 13


def boot_ci(v, n_resamples, rng):
    v = np.asarray(v, float)
    if len(v) < 2:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(v), (n_resamples, len(v)))
    m = v[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def series(L, G, nb, n_resamples, rng, label):
    """Quantile-bin one length arm; returns the drawable rows and rho."""
    order = np.argsort(L)
    L, G = L[order], G[order]
    edges = np.percentile(L, np.linspace(0, 100, nb + 1))
    edges[-1] = np.nextafter(edges[-1], np.inf)

    print(f"\n{label} — {len(L)} questions")
    print(f"{'length range (tok)':>19} | {'n':>4} | {'med len':>7} | "
          f"{'monitorab.':>10} | 95% CI")
    print("-" * 74)
    rows = []
    for i in range(nb):
        m = (L >= edges[i]) & (L < edges[i + 1])
        v = G[m]
        med = float(np.median(L[m]))
        lo, hi = boot_ci(v, n_resamples, rng)
        rng_str = f"{int(edges[i])}-{int(np.ceil(edges[i + 1]))}"
        print(f"{rng_str:>19} | {int(m.sum()):>4} | {med:>7.0f} | "
              f"{v.mean():>10.3f} | [{lo:.3f}, {hi:.3f}]")
        rows.append((med, float(v.mean()), lo, hi, int(m.sum()), rng_str))

    rho = stats.spearmanr(L, G)[0]
    bs = np.empty(n_resamples)
    for j in range(n_resamples):
        idx = rng.integers(0, len(L), len(L))
        bs[j] = stats.spearmanr(L[idx], G[idx])[0]
    rlo, rhi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    print(f"Spearman = {rho:+.3f} [{rlo:+.3f}, {rhi:+.3f}]")
    return rows, (rho, rlo, rhi)


def draw(ax, rows, rho_stats, colour, box_fc, box_ec, title, xlabel, y_floor):
    xs = [r[0] for r in rows]
    ys = [r[1] for r in rows]
    los = [r[2] for r in rows]
    his = [r[3] for r in rows]
    ax.set_axisbelow(True)
    ax.grid(True, color="#EEEEEE", linewidth=0.8)
    ax.errorbar(xs, ys, yerr=[np.array(ys) - np.array(los),
                              np.array(his) - np.array(ys)],
                color=colour, linewidth=2.0, marker="o", markersize=8,
                markeredgecolor="#333333", markeredgewidth=0.5, ecolor=colour,
                elinewidth=1.4, capsize=4, capthick=1.4, zorder=3)
    # n only. The bin token-RANGES are deliberately not drawn: the x-axis is
    # clamped to the plotted dots (which sit at bin medians), so a range label
    # like "4494-15691" names a value past the right-hand end of the axis and
    # reads as a contradictory tick. The axis now carries real token ticks, so
    # positions are readable directly; the bin extents belong in the caption.
    for (x, y, _, hi, n, rg) in rows:
        ax.annotate(f"n={n}", xy=(x, hi), xytext=(0, 6),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=FS_N, color="#555555")
    rho, rlo, rhi = rho_stats
    ax.text(0.97, 0.95, f"Spearman ρ = {rho:+.3f}\n[{rlo:+.3f}, {rhi:+.3f}]",
            transform=ax.transAxes, ha="right", va="top", fontsize=FS_BOX,
            color=colour,
            bbox=dict(boxstyle="round,pad=0.4", fc=box_fc, ec=box_ec))
    ax.set_title(title, fontsize=FS_TITLE)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_xlabel(xlabel, fontsize=FS_LABEL)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#999999")


def main():
    parser = argparse.ArgumentParser(
        description="Two-panel monitorability-vs-length figure (unhinted | "
                    "hinted).")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--data", default="test", choices=SPLITS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--bins", type=int, default=5)
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

    # one row per unique question: monitorability and BOTH length arms
    per_q = defaultdict(lambda: {"g": [], "Lh": [], "Lu": None})
    missing = []
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
                    lh = row.get("mean_cot_tokens_hinted", "")
                    lu = row.get("mean_cot_tokens_unhinted", "")
                    if any(v in ("", "None", "N/A") for v in (g, lh, lu)):
                        continue
                    key = (dataset, row["instance_id"])
                    per_q[key]["g"].append(float(g))
                    per_q[key]["Lh"].append(float(lh))
                    per_q[key]["Lu"] = float(lu)   # shared across hint slices
    if missing:
        print("WARNING: missing slices skipped: " + ", ".join(missing))
    if not per_q:
        print("ERROR: no entering questions found — nothing to plot.")
        sys.exit(1)

    G = np.array([float(np.mean(v["g"])) for v in per_q.values()])
    Lu = np.array([v["Lu"] for v in per_q.values()])
    Lh = np.array([float(np.mean(v["Lh"])) for v in per_q.values()])
    print(f"UNIQUE questions on both panels: {len(G)}")

    rows_u, rho_u = series(Lu, G, args.bins, args.resamples, rng,
                           "(a) UNHINTED length")
    rows_h, rho_h = series(Lh, G, args.bins, args.resamples, rng,
                           "(b) HINTED length")

    # ---------------- figure ----------------
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 4.6), sharex=True,
                                     sharey=True)
    all_lo = min(r[2] for r in rows_u + rows_h)
    all_hi = max(r[3] for r in rows_u + rows_h)
    y_floor = all_lo - 0.035
    ax_a.set_ylim(y_floor, all_hi + 0.03)

    xs_all = [r[0] for r in rows_u + rows_h]
    x_lo, x_hi = min(xs_all) / 1.4, max(xs_all) * 1.4
    for ax in (ax_a, ax_b):
        ax.set_xscale("log")
        ax.set_xlim(x_lo, x_hi)
        ticks = [t for t in TICKS if x_lo <= t <= x_hi]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(lambda v, _: f"{int(v):,}")
        ax.xaxis.set_minor_locator(NullLocator())

    # Short x-labels: at half width the full standalone captions collide across
    # the panel gap and the right one clips. The panel titles already name the
    # arm; "dots at bin median" and the CI method belong to the figure caption.
    draw(ax_a, rows_u, rho_u, DARK_BLUE, "#F4F6F9", "#CCD3DB",
         "(a) Unhinted arm",
         "Unhinted response length in tokens (log scale)\n"
         "per-question mean of 8 runs", y_floor)
    draw(ax_b, rows_h, rho_h, PLUM, "#F4F1F8", "#C9BEDD",
         "(b) Hinted arm",
         "Hinted response length in tokens (log scale)\n"
         "per-question mean over 8 runs × 3 hints", y_floor)
    ax_a.set_ylabel("Monitorability (g-mean²)", fontsize=FS_LABEL)

    out = args.out
    if out is None:
        plots_dir = ((root / config["paths"]["plots_dir"]).parent
                     / "extra-plots" / "cot-length-vs-monitorability")
        if args.data != "test":
            plots_dir = plots_dir / args.data
        plots_dir.mkdir(parents=True, exist_ok=True)
        out = plots_dir / f"length_gmean2_quantile_panel_{model_name}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
