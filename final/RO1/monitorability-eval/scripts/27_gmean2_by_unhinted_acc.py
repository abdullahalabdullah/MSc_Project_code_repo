#!/usr/bin/env python3
"""
Step 27: Level-wise plot of g-mean^2 vs unhinted task accuracy for ONE
model — pooled (default) or one panel per dataset (--per-dataset). The
statistically powered replacement for the per-question scatter (step 26).

Each ENTERING question (level-1 pass, gmean2_i defined) has an unhinted
accuracy acc_u_i = k/N over its N valid unhinted runs (N=8, so 9 levels;
questions with fewer valid runs are snapped to the nearest k/8 level).
Per level: the mean gmean2_i over every question at that level, with a
percentile-bootstrap 95% CI (resample questions within level) and the
level's n printed under the axis. Because per-question g-mean^2 is
near-binary, each level mean reads as "how monitorable is a question the
model gets right k times out of 8" — the conditional relationship the raw
scatter cannot show.

Default (pooled): one panel over all 8 datasets x 3 hint types, all 9
levels — the resolution the pooled n can support.
--per-dataset: per-dataset n CANNOT support 9 levels (middle levels hold
5-40 questions — pure noise), so accuracy is collapsed to the 3 groups the
pooled analysis showed actually matter (the effect is a threshold at
always-correct): never correct (0/8), sometimes correct (1-7/8), always
correct (8/8). One axes, x = the 8 datasets, three light -> dark blue dots
per dataset with CI whiskers — the same visual language as step 25's
terciles, with each dot averaging ~20-310 questions.

Stdout prints the per-level or per-group table (n / mean / CI) and
Pearson r + Spearman rho per plotted unit; the pooled run also prints two
headline contrasts with bootstrap CIs: always-correct (acc=1) vs the rest,
and acc >= 0.5 vs < 0.5.

Usage:
    python scripts/27_gmean2_by_unhinted_acc.py                  # pooled
    python scripts/27_gmean2_by_unhinted_acc.py --per-dataset    # 3-group

    Options:
      --config       path to config YAML (default: config/mvp_config.yaml)
      --data         split: test (default) | eval
      --model        model/adapter HF id (default: Qwen/Qwen3-4B)
      --per-dataset  3-group dot plot per dataset instead of the pooled panel
      --min-n        per-dataset mode: min questions per group to draw the
                     dot (default: 5; pooled mode ignores this)
      --resamples    bootstrap replicates per CI (default: 2000)
      --seed         bootstrap seed (default: 88)
      --out          output PNG path (defaults:
                     {plots_dir}/gmean2_by_unhinted_acc_{model}.png, or
                     …_by_dataset_{model}.png with --per-dataset)

Input (per dataset / hint type, produced by 04_compute_gmean2.py):
    {analysis_dir}/{model}/{dataset}/{hint}/results_per_instance.csv

Output:
    {plots_dir}/gmean2_by_unhinted_acc_{model}.png
    {plots_dir}/gmean2_by_unhinted_acc_by_dataset_{model}.png  (--per-dataset)
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
N_RUNS = 8  # unhinted runs per question -> 9 accuracy levels k/8

DATASETS = ["mmlu_pro", "gpqa_diamond", "lsat_ar", "lsat_lr",
            "medxpertqa", "lexam", "codemmlu", "wmdp"]

DARK_BLUE = "#1B3A6B"  # matches steps 22/25

# 3 accuracy groups for --per-dataset mode — the groups the pooled
# threshold effect motivates. Light -> dark shades of one blue (same ramp
# as step 25's terciles) so the ordering survives grayscale and CVD.
ACC_GROUPS = [
    ("Never correct (0/8)", "#9DC3E6", -0.25,
     lambda lev: lev == 0.0),
    ("Sometimes correct (1–7/8)", "#4A7BB7", 0.0,
     lambda lev: (lev > 0.0) & (lev < 1.0)),
    ("Always correct (8/8)", "#1B3A6B", 0.25,
     lambda lev: lev == 1.0),
]


def boot_mean_ci(values, n_resamples, rng):
    arr = np.asarray(values)
    idx = rng.integers(0, len(arr), size=(n_resamples, len(arr)))
    lo, hi = np.percentile(arr[idx].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def band_label(ks):
    ks = sorted(ks)
    core = str(ks[0]) if len(ks) == 1 else f"{ks[0]}–{ks[-1]}"
    return f"{core}/{N_RUNS}"


def parse_bands(spec):
    bands = []
    for chunk in spec.split("|"):
        ks = [int(x) for x in chunk.split(",") if x.strip() != ""]
        if ks:
            bands.append(ks)
    return bands


def boot_diff_ci(hi_vals, lo_vals, n_resamples, rng):
    h, l = np.asarray(hi_vals), np.asarray(lo_vals)
    ih = rng.integers(0, len(h), size=(n_resamples, len(h)))
    il = rng.integers(0, len(l), size=(n_resamples, len(l)))
    diffs = h[ih].mean(axis=1) - l[il].mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def level_table(name, lev, G, min_n, n_resamples, rng):
    """Per-level stats for one plotted unit; returns rows kept for drawing."""
    print(f"\n{name} — {len(G)} entering questions")
    print(f"{'acc level':>9} | {'n':>4} | {'g-mean2':>8} | 95% CI")
    print("-" * 48)
    rows = []
    for k in range(N_RUNS + 1):
        mask = lev == k / N_RUNS
        n = int(mask.sum())
        if n == 0:
            continue
        vals = G[mask]
        mean = float(vals.mean())
        lo, hi = boot_mean_ci(vals, n_resamples, rng)
        drop = " (dropped)" if n < min_n else ""
        print(f"{k:>7}/{N_RUNS} | {n:>4} | {mean:>8.3f} | "
              f"[{lo:.3f}, {hi:.3f}]{drop}")
        if n >= min_n:
            rows.append((k / N_RUNS, mean, lo, hi, n))
    return rows


def draw_levels(ax, rows, n_fontsize, y_floor, draw_line=True):
    """Dots + CI whiskers + (optionally) a line that BREAKS across gaps."""
    xs = [r[0] for r in rows]
    ys = [r[1] for r in rows]
    los = [r[2] for r in rows]
    his = [r[3] for r in rows]
    if draw_line:
        # NaN gaps wherever a level in between was dropped/empty.
        gx, gy = [xs[0]], [ys[0]]
        for (x0, _), (x1, y1) in zip(zip(xs, ys), list(zip(xs, ys))[1:]):
            if round((x1 - x0) * N_RUNS) > 1:
                gx.append(np.nan)
                gy.append(np.nan)
            gx.append(x1)
            gy.append(y1)
        ax.plot(gx, gy, color=DARK_BLUE, linewidth=1.6, zorder=2)
    ax.errorbar(xs, ys, yerr=[np.array(ys) - np.array(los),
                              np.array(his) - np.array(ys)],
                fmt="o", color=DARK_BLUE, markersize=6,
                markeredgecolor="#333333", markeredgewidth=0.5,
                ecolor=DARK_BLUE, elinewidth=1.3, capsize=2.5, capthick=1.3,
                zorder=3)
    for x, _, _, _, n in rows:
        ax.text(x, y_floor, f"{n}", ha="center", va="bottom",
                fontsize=n_fontsize, color="#666666")


def style_axis(ax):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#DDDDDD", linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#999999")
    ax.set_xticks([k / N_RUNS for k in range(N_RUNS + 1)])
    ax.set_xticklabels([f"{k}/{N_RUNS}" for k in range(N_RUNS + 1)])
    ax.set_xlim(-0.06, 1.06)


def main():
    parser = argparse.ArgumentParser(
        description="Level-wise g-mean^2 vs unhinted accuracy (pooled or "
                    "per dataset).")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--data", default="test", choices=SPLITS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--per-dataset", action="store_true")
    parser.add_argument("--no-line", action="store_true",
                        help="pooled mode: draw dots + CI only, no connecting "
                             "line between accuracy levels")
    parser.add_argument("--bands", default=None,
                        help="pooled mode: group accuracy levels into bands, "
                             "k-lists split by '|', e.g. '0|1,2,3|4,5,6,7|8' "
                             "(keep 8 in its own band); writes …_banded_…")
    parser.add_argument("--min-n", type=int, default=5)
    parser.add_argument("--resamples", type=int, default=2000)
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

    per_ds, missing = {ds: ([], []) for ds in DATASETS}, []
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
                    a = row.get("acc_u_i", "")
                    if g in ("", "None", "N/A") or a in ("", "None", "N/A"):
                        continue
                    per_ds[dataset][0].append(float(a))
                    per_ds[dataset][1].append(float(g))
    if missing:
        print("WARNING: missing slices skipped: " + ", ".join(missing))
    if not any(a for a, _ in per_ds.values()):
        print("ERROR: no data found — nothing to plot.")
        sys.exit(1)

    # RO1 bespoke analysis plots live under extra-plots/, not the main plots/ tree
    plots_dir = ((root / config["paths"]["plots_dir"]).parent
                 / "extra-plots" / "task-performance-vs-monitorability")
    if args.data != "test":
        plots_dir = plots_dir / args.data
    plots_dir.mkdir(parents=True, exist_ok=True)

    if args.per_dataset:
        from matplotlib.lines import Line2D

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color="#DDDDDD", linewidth=0.8)

        all_pts, plotted = [], []
        for d_idx, dataset in enumerate(DATASETS):
            A, G = (np.array(v) for v in per_ds[dataset])
            if len(G) < 3 * args.min_n:
                print(f"\n{dataset}: only {len(G)} entering questions "
                      f"— skipped")
                continue
            lev = np.clip(np.round(A * N_RUNS) / N_RUNS, 0, 1)
            r, r_p = stats.pearsonr(A, G)
            rho, rho_p = stats.spearmanr(A, G)
            print(f"\n{dataset} — {len(G)} entering questions, "
                  f"Pearson r = {r:+.3f} (p = {r_p:.3g}); "
                  f"Spearman rho = {rho:+.3f} (p = {rho_p:.3g})")
            print(f"{'group':>26} | {'n':>4} | {'g-mean2':>8} | 95% CI")
            print("-" * 64)
            for label, color, off, sel in ACC_GROUPS:
                mask = sel(lev)
                n = int(mask.sum())
                if n < args.min_n:
                    print(f"{label:>26} | {n:>4} | (dropped: n < "
                          f"{args.min_n})")
                    continue
                vals = G[mask]
                mean = float(vals.mean())
                lo, hi = boot_mean_ci(vals, args.resamples, rng)
                print(f"{label:>26} | {n:>4} | {mean:>8.3f} | "
                      f"[{lo:.3f}, {hi:.3f}]")
                x = d_idx + off
                ax.errorbar(x, mean, yerr=[[mean - lo], [hi - mean]],
                            fmt="o", color=color, markersize=7,
                            markeredgecolor="#333333", markeredgewidth=0.5,
                            ecolor=color, elinewidth=1.4, capsize=3,
                            capthick=1.4, zorder=3)
                all_pts.extend([lo, hi])
            plotted.append((d_idx, dataset))

        if not plotted:
            print("ERROR: no dataset had enough questions — nothing to plot.")
            sys.exit(1)
        ax.set_xticks([i for i, _ in plotted])
        ax.set_xticklabels([ds for _, ds in plotted])
        ax.set_xlim(-0.55, len(DATASETS) - 0.45)
        ax.set_ylim(min(all_pts) - 0.02, max(all_pts) + 0.02)
        for i in range(len(DATASETS) - 1):
            ax.axvline(i + 0.5, color="#EEEEEE", linewidth=0.8, zorder=1)
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Mean g-mean²")
        ax.set_title(f"Monitorability by unhinted task accuracy, per dataset "
                     f"— {model_name}\n(entering questions per dataset, hint "
                     f"types pooled, grouped by unhinted accuracy over the 8 "
                     f"runs; bootstrap 95% CI)")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#999999")
        handles = [Line2D([], [], marker="o", linestyle="none", color=color,
                          markeredgecolor="#333333", markeredgewidth=0.5,
                          markersize=7, label=label)
                   for label, color, _, _ in ACC_GROUPS]
        ax.legend(handles=handles, title="Unhinted accuracy",
                  loc="upper left", bbox_to_anchor=(1.005, 1.0),
                  frameon=True, edgecolor="#CCCCCC", fontsize=9)
        out = args.out or (plots_dir /
                           f"gmean2_by_unhinted_acc_by_dataset_{model_name}.png")
        fig.tight_layout()
    else:
        A = np.array([a for ds in DATASETS for a in per_ds[ds][0]])
        G = np.array([g for ds in DATASETS for g in per_ds[ds][1]])
        lev = np.clip(np.round(A * N_RUNS) / N_RUNS, 0, 1)
        rows = level_table(
            f"POOLED ({len(DATASETS)} datasets x {len(HINT_TYPES)} hints)",
            lev, G, 1, args.resamples, rng)
        r, r_p = stats.pearsonr(A, G)
        rho, rho_p = stats.spearmanr(A, G)
        print(f"\npooled per-question Pearson r = {r:+.3f} (p = {r_p:.3g}); "
              f"Spearman rho = {rho:+.3f} (p = {rho_p:.3g})")
        always = lev == 1.0
        d = float(G[always].mean() - G[~always].mean())
        lo, hi = boot_diff_ci(G[always], G[~always],
                              max(args.resamples, 5000), rng)
        print(f"always-correct (n={always.sum()}, mean {G[always].mean():.3f})"
              f" vs rest (n={(~always).sum()}, mean {G[~always].mean():.3f}): "
              f"Delta = {d:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}]")
        hi_m = lev >= 0.5
        d2 = float(G[hi_m].mean() - G[~hi_m].mean())
        lo2, hi2 = boot_diff_ci(G[hi_m], G[~hi_m],
                                max(args.resamples, 5000), rng)
        print(f"acc >= 0.5 (n={hi_m.sum()}, mean {G[hi_m].mean():.3f}) vs "
              f"acc < 0.5 (n={(~hi_m).sum()}, mean {G[~hi_m].mean():.3f}): "
              f"Delta = {d2:+.3f}, 95% CI [{lo2:+.3f}, {hi2:+.3f}]")

        if args.bands:
            bands = parse_bands(args.bands)
            kint = np.round(A * N_RUNS).astype(int)
            brows = []  # (x, mean, lo, hi, n, label)
            print()
            for i, ks in enumerate(bands):
                m = np.isin(kint, ks)
                n = int(m.sum())
                if n == 0:
                    continue
                v = G[m]
                blo, bhi = boot_mean_ci(v, args.resamples, rng)
                brows.append((i, float(v.mean()), blo, bhi, n,
                              band_label(ks)))
                print(f"  band {band_label(ks):>7}  n={n:5d}  "
                      f"mean={v.mean():.3f}  CI[{blo:.3f},{bhi:.3f}]")

            fig, ax = plt.subplots(figsize=(8.5, 5))
            ax.set_axisbelow(True)
            ax.yaxis.grid(True, color="#DDDDDD", linewidth=0.8)
            xs = [b[0] for b in brows]
            ys = [b[1] for b in brows]
            blos = [b[2] for b in brows]
            bhis = [b[3] for b in brows]
            y_lo = min(blos) - 0.03
            ax.set_ylim(y_lo, max(bhis) + 0.02)
            if not args.no_line:
                ax.plot(xs, ys, color=DARK_BLUE, linewidth=1.6, zorder=2)
            ax.errorbar(xs, ys,
                        yerr=[np.array(ys) - np.array(blos),
                              np.array(bhis) - np.array(ys)],
                        fmt="o", color=DARK_BLUE, markersize=7,
                        markeredgecolor="#333333", markeredgewidth=0.5,
                        ecolor=DARK_BLUE, elinewidth=1.4, capsize=3.5,
                        capthick=1.4, zorder=3)
            for x, _, _, _, n, _ in brows:
                ax.text(x, y_lo + 0.006, f"{n}", ha="center", va="bottom",
                        fontsize=8, color="#666666")
            ax.set_xticks(xs)
            ax.set_xticklabels([b[5] for b in brows])
            ax.set_xlim(min(xs) - 0.5, max(xs) + 0.5)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            for spine in ("left", "bottom"):
                ax.spines[spine].set_color("#999999")
            ax.set_xlabel("Unhinted accuracy, banded (correct runs / 8)")
            ax.set_ylabel("Mean g-mean²")
            ax.set_title(f"Monitorability by unhinted task accuracy — "
                         f"{model_name}\n(entering questions pooled over 8 "
                         f"datasets × 3 hint types; accuracy banded; "
                         f"bootstrap 95% CI)")
            out = args.out or (plots_dir /
                               f"gmean2_by_unhinted_acc_banded_{model_name}"
                               f".png")
            fig.tight_layout()
            fig.savefig(out, dpi=200, bbox_inches="tight")
            print(f"\nWrote {out}")
            return

        fig, ax = plt.subplots(figsize=(8.5, 5))
        style_axis(ax)
        los = [row[2] for row in rows]
        his = [row[3] for row in rows]
        y_lo = min(los) - 0.03
        ax.set_ylim(y_lo, max(his) + 0.02)
        draw_levels(ax, rows, n_fontsize=7.5, y_floor=y_lo + 0.006,
                    draw_line=not args.no_line)
        ax.set_xlabel("Unhinted accuracy (fraction of 8 unhinted runs correct)")
        ax.set_ylabel("Mean g-mean²")
        ax.set_title(f"Monitorability by unhinted task accuracy — "
                     f"{model_name}\n(entering questions pooled over 8 "
                     f"datasets × 3 hint types; bootstrap 95% CI)")
        suffix = "_noline" if args.no_line else ""
        out = args.out or (plots_dir /
                           f"gmean2_by_unhinted_acc{suffix}_{model_name}.png")
        fig.tight_layout()

    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
