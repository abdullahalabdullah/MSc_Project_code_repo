#!/usr/bin/env python3
"""
Step 47: Step 41's monitorability-vs-HINTED-length curve, drawn SEPARATELY for
each hint type — one figure, one panel per hint.

Step 41 pools the three hint types into a single curve by averaging, per
question, both the hinted length and `gmean2_i` over whichever hint types that
question qualified for (level-1 pass). That per-question collapse has two
costs this step avoids:

  1. NO AVERAGING AT ALL. Within one hint type a question appears exactly
     once: its `mean_cot_tokens_hinted` and its `gmean2_i` are the values
     measured under that hint and pair directly. The x-value is still a mean
     over the question's 8 runs (computed upstream by 04), but there is no
     second, cross-hint average stacked on top of it.
  2. NO ENTRY-COUNT CONFOUND. In step 41 a question contributes 1, 2, or 3
     hint conditions depending on where it passed level-1 (base: 541 / 317 /
     525 questions respectively), and that count correlates with length
     (Spearman +0.34) AND with monitorability (mean g-mean^2 0.946 / 0.850 /
     0.674). The pooled curve therefore mixes a between-strata composition
     contrast into what reads as a within-question length effect. Restricting
     to one hint type removes the strata entirely.

Consequently these three panels — not step 41's pooled curve — are the clean
estimate of the hinted-length effect, and they disagree with each other:
the relationship is hint-dependent, not a single pooled constant.

Unit = the QUESTION, within one hint type. Binning, axes, and statistics are
otherwise identical to step 41 (equal-count quantile bins, dots at the bin
median on a log x-axis, bin mean g-mean^2, bootstrap 95% CIs, Spearman rho
with a bootstrap CI). The panels share x and y limits so the hints' very
different monitorability bands are directly comparable.

Usage:
    python scripts/47_hinted_length_gmean2_quantile_by_hint.py         # base
    python scripts/47_hinted_length_gmean2_quantile_by_hint.py --model <hf-id>

    Options:
      --config     path to config YAML (default: config/mvp_config.yaml)
      --data       split: test (default) | eval
      --model      model/adapter HF id (default: Qwen/Qwen3-4B)
      --bins       equal-count quantile bins (default: 5)
      --linear-x   linear length axis (default: log10)
      --resamples  bootstrap replicates per CI / rho (default: 5000)
      --seed       bootstrap seed (default: 88)
      --outdir     output directory (default:
                   {plots_dir}/cot-length-vs-monitorability/)

Input (per dataset / hint type, produced by 04_compute_gmean2.py):
    {analysis_dir}/{model}/{dataset}/{hint}/results_per_instance.csv

Output (one figure, one panel per hint type):
    {plots_dir}/cot-length-vs-monitorability/
        gmean2_by_hinted_length_quantile_by_hint_{model}.png
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

# One CVD-safe hue per hint type, with a matching pale box fill.
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


def load_one_hint(root, config, model_name, hint, split):
    """One row per QUESTION for a single hint type — no averaging over hints."""
    L, G, missing = [], [], []
    for dataset in DATASETS:
        paths = resolve_paths(root, config, dataset, model_name, hint,
                              split=split)
        csv_path = paths["analysis_dir"] / "results_per_instance.csv"
        if not csv_path.exists():
            missing.append(dataset)
            continue
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
    if missing:
        print(f"  WARNING: {hint}: missing datasets skipped: "
              + ", ".join(missing))
    if not L:
        return None, None
    L, G = np.array(L), np.array(G)
    order = np.argsort(L)
    return L[order], G[order]


def analyse(L, G, nb, resamples, rng):
    """Quantile bins + per-bin mean g-mean^2 + Spearman rho with bootstrap CI."""
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
                          lo=lo, hi=hi, n=int(m.sum()),
                          rng_lab=f"{int(edges[i])}-{int(np.ceil(edges[i+1]))}"))
    rho = stats.spearmanr(L, G)[0]
    bs = np.empty(resamples)
    for j in range(resamples):
        idx = rng.integers(0, len(L), len(L))
        bs[j] = stats.spearmanr(L[idx], G[idx])[0]
    rlo, rhi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    return dict(drawn=drawn, rho=float(rho), rlo=float(rlo), rhi=float(rhi),
                n=len(L), edges=edges)


def main():
    parser = argparse.ArgumentParser(
        description="Monitorability vs HINTED length, one figure per hint "
                    "type (no cross-hint averaging).")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--data", default="test", choices=SPLITS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--linear-x", action="store_true")
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=88)
    parser.add_argument("--outdir", default=None)
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

    # ---- pass 1: compute every hint's curve so the three figures can share
    # x/y limits and be read side by side ----
    results = {}
    for hint in HINT_TYPES:
        L, G = load_one_hint(root, config, model_name, hint, args.data)
        if L is None:
            print(f"\n{hint}: no entering questions — skipped")
            results[hint] = None
            continue
        res = analyse(L, G, args.bins, args.resamples, rng)
        results[hint] = res
        print(f"\n=== {hint} — {res['n']} questions "
              f"(one row per question, no cross-hint averaging) ===")
        print(f"hinted length (tok): min={L.min():.0f} "
              f"median={np.median(L):.0f} max={L.max():.0f}")
        print(f"{'length range (tok)':>19} | {'n':>4} | {'med len':>7} | "
              f"{'monitorab.':>10} | 95% CI")
        print("-" * 74)
        for d in res["drawn"]:
            print(f"{d['rng_lab']:>19} | {d['n']:>4} | {d['med']:>7.0f} | "
                  f"{d['mean']:>10.3f} | [{d['lo']:.3f}, {d['hi']:.3f}]")
        sig = "" if (res["rlo"] < 0 < res["rhi"]) else "  (CI excludes 0)"
        print(f"Spearman(hinted length, monitorability) = {res['rho']:+.3f} "
              f"[{res['rlo']:+.3f}, {res['rhi']:+.3f}]{sig}")

    live = {h: r for h, r in results.items() if r}
    if not live:
        print("\nERROR: no populated hint types — nothing to plot.")
        sys.exit(1)

    bounds = [b for r in live.values() for d in r["drawn"]
              for b in (d["lo"], d["hi"])]
    ylo, yhi = min(bounds) - 0.04, max(bounds) + 0.04
    xlo = min(r["edges"][0] for r in live.values()) * 0.85
    xhi = max(r["edges"][-1] for r in live.values()) * 1.15

    outdir = args.outdir
    if outdir is None:
        outdir = ((root / config["paths"]["plots_dir"]).parent
                  / "extra-plots" / "cot-length-vs-monitorability")
        if args.data != "test":
            outdir = outdir / args.data
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- pass 2: one figure, one panel per hint ----
    hints = list(results)
    fig, axes = plt.subplots(1, len(hints), figsize=(5.4 * len(hints), 5.8),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, hint in zip(axes, hints):
        res = results[hint]
        colour, fc, ec = HINT_STYLE.get(hint, FALLBACK_STYLE)
        ax.set_axisbelow(True)
        if use_log:
            ax.set_xscale("log")
        ax.grid(True, color="#EEEEEE", linewidth=0.8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#999999")
        if not res:
            ax.set_title(hint, fontsize=12, fontweight="bold")
            ax.text(0.5, 0.5, "no entering questions", transform=ax.transAxes,
                    ha="center", va="center", color="#999999", fontsize=9)
            continue

        d = res["drawn"]
        xs = [b["med"] for b in d]
        ys = [b["mean"] for b in d]
        elo = [b["mean"] - b["lo"] for b in d]
        ehi = [b["hi"] - b["mean"] for b in d]
        ax.errorbar(xs, ys, yerr=[elo, ehi], color=colour, linewidth=2.0,
                    marker="o", markersize=7, markeredgecolor="#333333",
                    markeredgewidth=0.5, ecolor=colour, elinewidth=1.3,
                    capsize=3.5, capthick=1.3, zorder=3)
        for b, eh in zip(d, ehi):
            ax.annotate(f"n={b['n']}", xy=(b["med"], b["mean"] + eh),
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8,
                        color="#555555")

        verdict = ("CI excludes 0" if not (res["rlo"] < 0 < res["rhi"])
                   else "CI includes 0 — null")
        # Shared y-limits put each hint's curve in a different band, so anchor
        # the stats box wherever that panel's right-hand points are not.
        top_frac = max((b["hi"] - ylo) / (yhi - ylo) for b in d[-2:])
        box_y, box_va = (0.05, "bottom") if top_frac > 0.70 else (0.95, "top")
        ax.text(0.96, box_y,
                f"ρ = {res['rho']:+.3f}\n"
                f"[{res['rlo']:+.3f}, {res['rhi']:+.3f}]\n{verdict}",
                transform=ax.transAxes, ha="right", va=box_va, fontsize=9,
                color=colour,
                bbox=dict(boxstyle="round,pad=0.35", fc=fc, ec=ec))
        ax.set_title(f"{hint}  ·  n = {res['n']}", fontsize=12,
                     fontweight="bold")

    axes[0].set_xlim(xlo, xhi)
    axes[0].set_ylim(ylo, yhi)
    axes[0].set_ylabel("Monitorability (g-mean²)\nerror bars: bootstrap 95% CI",
                       fontsize=10)
    fig.supxlabel("HINTED response length — per-question mean over 8 runs "
                  f"(tokens{', log scale' if use_log else ''}; dots at bin "
                  "median)", fontsize=10, y=0.03)
    fig.suptitle("Monitorability vs HINTED length, within each hint condition "
                 f"— {model_name}\n"
                 "one row per question per hint — no cross-hint averaging and "
                 "no level-1 entry-count confound (contrast the pooled step 41)",
                 fontsize=13)

    out = (outdir /
           f"gmean2_by_hinted_length_quantile_by_hint_{model_name}.png")
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out}")

    print("\n--- cross-hint summary (compare with step 41's pooled "
          "rho, which averages over hints) ---")
    for hint, res in results.items():
        if not res:
            print(f"  {hint:<16} (no data)")
            continue
        flag = "" if (res["rlo"] < 0 < res["rhi"]) else "  *"
        print(f"  {hint:<16} n={res['n']:>4}  rho={res['rho']:+.3f} "
              f"[{res['rlo']:+.3f}, {res['rhi']:+.3f}]{flag}")


if __name__ == "__main__":
    main()
