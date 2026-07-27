#!/usr/bin/env python3
"""
Step 43: Does hint-SUSCEPTIBILITY (Treatment Effect) predict monitorability?
Pooled over all (dataset, hint) slices.

TE_i = p1_i - p0_i is the treatment effect — how much the hint raises the
probability of the hinted-target answer (the model's raw susceptibility to the
hint on that question). This asks: are questions where the hint sways the
model more (or less) also more monitorable?

Use TE, NOT R_effect. R_effect_i = TE_i / p1_i is embedded in the g-mean^2
formula itself (tpr_mc_i = z_x1_y1_i / r_effect_i, and gmean2_i = tpr_mc_i *
sqrt(tnr_x0_i * tnr_x1_i)), so correlating R_effect with g-mean^2 is circular
— it measures the metric's construction, not behaviour. TE does not appear in
the formula and is the clean, exogenous susceptibility measure.

Unit = the entering (question, hint) SLICE (level1_pass and gmean2_i defined —
the "Level-2" set on which monitorability is defined). TE is genuinely per
slice (a different hint has a different effect), so per-slice is the natural
unit (base: 2750 slices; mild within-question clustering).

Result (base model): NULL. Spearman(TE, gmean2) = -0.02 (n.s.) — a negligible,
non-significant relationship. Hint-susceptibility and monitorability are
ORTHOGONAL: how much the hint changes the model's behaviour (TE) is
independent of whether the model verbalises being swayed (monitorability).
Being easily nudged does not make a question more or less monitorable. Pairs
with the hinted-length null (steps 41/42): monitorability is about faithful
verbalisation, not susceptibility or verbosity.

Usage:
    python scripts/43_te_gmean2_quantile.py                        # base
    python scripts/43_te_gmean2_quantile.py --model <hf-id>

    Options:
      --config     path to config YAML (default: config/mvp_config.yaml)
      --data       split: test (default) | eval
      --model      model/adapter HF id (default: Qwen/Qwen3-4B)
      --bins       equal-count quantile bins (default: 5)
      --resamples  bootstrap replicates per CI / rho (default: 5000)
      --seed       bootstrap seed (default: 88)
      --out        output PNG path (default:
                   {plots_dir}/task-performance-vs-monitorability/
                   gmean2_by_te_quantile_{model}.png)

Input (per dataset / hint type, produced by 04_compute_gmean2.py):
    {analysis_dir}/{model}/{dataset}/{hint}/results_per_instance.csv

Output:
    {plots_dir}/task-performance-vs-monitorability/
        gmean2_by_te_quantile_{model}.png
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
STEEL = "#457B9D"


def boot_ci(v, n_resamples, rng):
    v = np.asarray(v, float)
    if len(v) < 2:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(v), (n_resamples, len(v)))
    m = v[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    parser = argparse.ArgumentParser(
        description="Monitorability vs hint-susceptibility (TE), pooled, "
                    "equal-count quantile bins.")
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

    TE, G, missing = [], [], []
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
                    t = row.get("te_i", "")
                    if g in ("", "None", "N/A") or t in ("", "None", "N/A"):
                        continue
                    TE.append(float(t))
                    G.append(float(g))
    if missing:
        print("WARNING: missing slices skipped: " + ", ".join(missing))
    if not TE:
        print("ERROR: no entering slices found — nothing to plot.")
        sys.exit(1)

    TE = np.array(TE)
    G = np.array(G)
    n = len(TE)
    order = np.argsort(TE)
    TE, G = TE[order], G[order]

    print(f"entering (question,hint) slices: {n}")
    print(f"TE: min={TE.min():.3f} median={np.median(TE):.3f} max={TE.max():.3f}\n")

    nb = args.bins
    edges = np.percentile(TE, np.linspace(0, 100, nb + 1))
    edges[-1] = np.nextafter(edges[-1], np.inf)

    xs, ys, elo, ehi, ns, ranges = [], [], [], [], [], []
    print(f"{'TE range':>15} | {'n':>4} | {'med TE':>7} | {'monitorab.':>10} | 95% CI")
    print("-" * 66)
    for i in range(nb):
        m = (TE >= edges[i]) & (TE < edges[i + 1])
        v = G[m]
        med = float(np.median(TE[m]))
        lo, hi = boot_ci(v, args.resamples, rng)
        rg = f"{edges[i]:.2f}-{edges[i+1]:.2f}"
        print(f"{rg:>15} | {int(m.sum()):>4} | {med:>7.3f} | {v.mean():>10.3f} "
              f"| [{lo:.3f}, {hi:.3f}]")
        xs.append(med)
        ys.append(float(v.mean()))
        elo.append(v.mean() - lo)
        ehi.append(hi - v.mean())
        ns.append(int(m.sum()))
        ranges.append(rg)

    rho = stats.spearmanr(TE, G)[0]
    bs = np.empty(args.resamples)
    for j in range(args.resamples):
        idx = rng.integers(0, n, n)
        bs[j] = stats.spearmanr(TE[idx], G[idx])[0]
    rlo, rhi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    print(f"\nSpearman(TE, monitorability) = {rho:+.3f} [{rlo:+.3f}, {rhi:+.3f}] "
          f"— negligible / null")

    # ---------------- figure ----------------
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.set_axisbelow(True)
    ax.grid(True, color="#EEEEEE", linewidth=0.8)
    ax.errorbar(xs, ys, yerr=[elo, ehi], color=STEEL, linewidth=2.0,
                marker="o", markersize=8, markeredgecolor="#333333",
                markeredgewidth=0.5, ecolor=STEEL, elinewidth=1.4,
                capsize=4, capthick=1.4, zorder=3)
    for x, y, eh, nn in zip(xs, ys, ehi, ns):
        ax.annotate(f"n={nn}", xy=(x, y + eh), xytext=(0, 6),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=8.5, color="#555555")

    ax.text(0.03, 0.95,
            f"Spearman ρ = {rho:+.3f}\n[{rlo:+.3f}, {rhi:+.3f}]\n"
            f"negligible (n.s.)",
            transform=ax.transAxes, ha="left", va="top", fontsize=10,
            color=STEEL,
            bbox=dict(boxstyle="round,pad=0.4", fc="#EEF3F7", ec="#B9CBD8"))

    ax.set_xlabel("Hint-susceptibility — Treatment Effect TE = p₁ − p₀ "
                  "(per (question,hint) slice; dots at bin median)")
    ax.set_ylabel("Monitorability (g-mean²)  ·  error bars: bootstrap 95% CI")
    ax.set_title("Monitorability vs hint-susceptibility (TE) — "
                 f"{model_name}\n"
                 f"{n} entering slices; NO relationship — being swayed by a "
                 "hint is independent of verbalising it")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#999999")

    out = args.out
    if out is None:
        plots_dir = ((root / config["paths"]["plots_dir"]).parent
                     / "extra-plots" / "task-performance-vs-monitorability")
        if args.data != "test":
            plots_dir = plots_dir / args.data
        plots_dir.mkdir(parents=True, exist_ok=True)
        out = plots_dir / f"gmean2_by_te_quantile_{model_name}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
