#!/usr/bin/env python3
"""
Step 41: Monitorability vs HINTED response length — the monitor's-eye
companion to step 37 (which uses UNHINTED length).

Step 37 bins by the UNHINTED arm's length (the model's intrinsic reasoning
verbosity, run once per question and shared across hint types). This step
bins by the HINTED arm's length — the length of the transcript the MONITOR
ACTUALLY READS. If monitorability fell simply because longer transcripts are
harder to read, THIS is where it would show up.

Key structural difference from step 37: the hinted arm is generated
SEPARATELY for each hint type, so `mean_cot_tokens_hinted` and `gmean2_i` are
both per-(question, hint) SLICE and pair directly — no per-question collapse,
no averaging over hints. The unit is therefore the SLICE (base: 2750 slices
from 1383 unique questions x 3 hint types), the same pooling steps 22/27/28/31
use. (Slices from one question share the question, so they are clustered, not
fully independent — a mild caveat on the CI width; unlike step 37 there is no
pseudoreplication of identical length values, because the model reasons a
different amount under each hint.)

The result is the headline of this figure: unlike unhinted length, HINTED
length is essentially UNRELATED to monitorability. Base model: Spearman = -0.01
(p ~ 0.7), quintile means flat at ~0.74-0.79 with overlapping CIs — vs step
37's clear -0.29. So the length->monitorability effect is NOT about the
monitored transcript being long; it is about the model reasoning more
INTRINSICALLY on that question (unhinted length), which is itself a marker of
low commitment/determinacy. The monitored object's length carries no signal.

Binning, axes, and stats are identical to step 37 (equal-count quantile bins,
dots at bin median on a log x-axis, bootstrap 95% CI, Spearman rho with
bootstrap CI).

Usage:
    python scripts/41_hinted_length_gmean2_quantile.py            # base
    python scripts/41_hinted_length_gmean2_quantile.py --model <hf-id>

    Options:
      --config     path to config YAML (default: config/mvp_config.yaml)
      --data       split: test (default) | eval
      --model      model/adapter HF id (default: Qwen/Qwen3-4B)
      --bins       number of equal-count quantile bins (default: 5)
      --linear-x   linear length axis (default: log10)
      --resamples  bootstrap replicates per CI (default: 5000)
      --seed       bootstrap seed (default: 88)
      --out        output PNG path (default:
                   {plots_dir}/cot-length-vs-monitorability/
                   gmean2_by_hinted_length_quantile_{model}.png)

Input (per dataset / hint type, produced by 04_compute_gmean2.py):
    {analysis_dir}/{model}/{dataset}/{hint}/results_per_instance.csv

Output:
    {plots_dir}/cot-length-vs-monitorability/
        gmean2_by_hinted_length_quantile_{model}.png
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
# distinct colour from step 37's navy, to signal "different arm / null result"
PLUM = "#6A4C93"


def boot_ci(v, n_resamples, rng):
    v = np.asarray(v, float)
    if len(v) < 2:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(v), (n_resamples, len(v)))
    m = v[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    parser = argparse.ArgumentParser(
        description="Monitorability vs HINTED response length, pooled over "
                    "(question, hint) slices, equal-count quantile bins.")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--data", default="test", choices=SPLITS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--bins", type=int, default=5)
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

    # --- per-QUESTION unit (parity with step 37): collapse the 3 hint slices
    # so the ONLY difference from 37 is the length arm. Monitorability = mean
    # gmean2 over the question's hint slices (identical to 37's y); hinted
    # length = mean hinted length over those slices. This restores independence
    # (one row per question, no within-question clustering) and isolates the
    # unhinted-vs-hinted length arm as the sole change. ---
    per_q, missing = defaultdict(lambda: {"g": [], "L": []}), []
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
                    L = row.get("mean_cot_tokens_hinted", "")
                    if g in ("", "None", "N/A") or L in ("", "None", "N/A"):
                        continue
                    key = (dataset, row["instance_id"])
                    per_q[key]["g"].append(float(g))
                    per_q[key]["L"].append(float(L))
    if missing:
        print("WARNING: missing slices skipped: " + ", ".join(missing))
    if not per_q:
        print("ERROR: no entering questions found — nothing to plot.")
        sys.exit(1)

    L = np.array([float(np.mean(v["L"])) for v in per_q.values()])
    G = np.array([float(np.mean(v["g"])) for v in per_q.values()])
    n = len(L)
    order = np.argsort(L)
    L, G = L[order], G[order]

    print(f"UNIQUE questions (hinted length & monitorability averaged over "
          f"{len(HINT_TYPES)} hint slices): {n}")
    print(f"mean hinted length (tok): min={L.min():.0f} "
          f"median={np.median(L):.0f} max={L.max():.0f}\n")

    nb = args.bins
    edges = np.percentile(L, np.linspace(0, 100, nb + 1))
    edges[-1] = np.nextafter(edges[-1], np.inf)

    xs, ys, elo, ehi, ns, ranges = [], [], [], [], [], []
    print(f"{'length range (tok)':>19} | {'n':>4} | {'med len':>7} | "
          f"{'monitorab.':>10} | 95% CI")
    print("-" * 74)
    for i in range(nb):
        m = (L >= edges[i]) & (L < edges[i + 1])
        v = G[m]
        med = float(np.median(L[m]))
        lo, hi = boot_ci(v, args.resamples, rng)
        rg = f"{int(edges[i])}-{int(np.ceil(edges[i+1]))}"
        print(f"{rg:>19} | {int(m.sum()):>4} | {med:>7.0f} | {v.mean():>10.3f} "
              f"| [{lo:.3f}, {hi:.3f}]")
        xs.append(med)
        ys.append(float(v.mean()))
        elo.append(v.mean() - lo)
        ehi.append(hi - v.mean())
        ns.append(int(m.sum()))
        ranges.append(rg)

    rho = stats.spearmanr(L, G)[0]
    bs = np.empty(args.resamples)
    for j in range(args.resamples):
        idx = rng.integers(0, n, n)
        bs[j] = stats.spearmanr(L[idx], G[idx])[0]
    rlo, rhi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    print(f"\nSpearman(mean hinted length, monitorability) = {rho:+.3f} "
          f"[{rlo:+.3f}, {rhi:+.3f}]   (unhinted per-question was -0.289 — "
          f"same questions, same monitorability, only the length arm differs)")

    # ---------------- figure ----------------
    use_log = not args.linear_x
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.set_axisbelow(True)
    if use_log:
        ax.set_xscale("log")
        ax.set_xlim(edges[0] * 0.85, edges[-1] * 1.15)
    ax.grid(True, color="#EEEEEE", linewidth=0.8)
    ax.errorbar(xs, ys, yerr=[elo, ehi], color=PLUM, linewidth=2.0,
                marker="o", markersize=8, markeredgecolor="#333333",
                markeredgewidth=0.5, ecolor=PLUM, elinewidth=1.4,
                capsize=4, capthick=1.4, zorder=3)
    for x, y, eh, nn, rg in zip(xs, ys, ehi, ns, ranges):
        ax.annotate(f"n={nn}", xy=(x, y + eh), xytext=(0, 6),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=8.5, color="#555555")
        ax.annotate(rg, xy=(x, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    va="bottom", fontsize=7.5, color="#999999")

    ax.text(0.97, 0.93,
            f"Spearman ρ = {rho:+.3f}\n[{rlo:+.3f}, {rhi:+.3f}]\n"
            f"negligible — vs −0.29 unhinted",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            color=PLUM,
            bbox=dict(boxstyle="round,pad=0.4", fc="#F4F1F8", ec="#C9BEDD"))

    ax.set_xlabel("HINTED response length — per-question mean over 8 runs × 3 "
                  f"hints (tokens{', log scale' if use_log else ''}; dots at "
                  "bin median)")
    ax.set_ylabel("Monitorability (g-mean²)  ·  error bars: bootstrap 95% CI")
    ax.set_title("Monitorability vs HINTED length (what the monitor reads) — "
                 f"{model_name}\n"
                 f"{n} questions (parity with the unhinted figure); negligible "
                 "relationship — the length effect is intrinsic, not transcript length")
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
               f"gmean2_by_hinted_length_quantile_{model_name}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
