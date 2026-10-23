#!/usr/bin/env python3
"""
Step 50: Level-wise plot of g-mean^2 vs answer CONSISTENCY — the consistency
analogue of step 27 (which does the same for unhinted task accuracy).

Step 27 asks "how monitorable is a question the model gets right k times out
of 8?" and finds a flat profile with a single step at 8/8. Step 31 then shows
that the step is about COMMITMENT, not correctness, by splitting the 8/8 group
into consistent-but-wrong and consistent-and-correct. This step asks the
question the other two leave open: is commitment a THRESHOLD like accuracy, or
a GRADIENT? It plots monitorability against the full consistency scale rather
than against the binary consistent/inconsistent split of steps 31/33/36.

Correctness plays no part here. The only variable is how much the model agreed
with itself.

    consistency_i = (count of the modal unhinted answer) / (valid unhinted runs)

so a question whose 8 unhinted runs all give the same answer scores 100%, one
that splits 4-4 between two answers scores 50%, and one that gives 8 different
answers scores 12.5%. Note the floor: with N=8 runs the modal count is at
least 1, so the scale starts at 1/8 = 12.5% and 0% is not attainable.

Unit and population are step 27's: every ENTERING observation (level-1 pass,
gmean2_i defined) over the 8 datasets x 3 hint types. Consistency is a
property of the QUESTION, read once per dataset from the shared unhinted arm
(no new inference) and joined to each of that question's entering hint slices,
exactly as steps 27/31 pool. Questions with fewer than 8 valid unhinted runs
are snapped to the nearest k/8 level, as in step 27.

Per level: mean gmean2_i with a percentile-bootstrap 95% CI (resampling
observations within the level) and n printed under the axis. Stdout adds
Pearson r and Spearman rho over the raw per-observation values, plus the
fully-consistent (100%) vs rest contrast for comparison with step 31's
consistency effect.

Usage:
    python scripts/50_gmean2_by_consistency_level.py            # base model
    python scripts/50_gmean2_by_consistency_level.py --model <hf-id>

    Options:
      --config     path to config YAML (default: config/mvp_config.yaml)
      --data       split: test (default) | eval
      --model      model/adapter HF id (default: Qwen/Qwen3-4B)
      --no-line    dots + CI only, no connecting line between levels
      --min-n      min observations per level to draw the dot (default: 5)
      --resamples  bootstrap replicates per CI (default: 5000)
      --seed       bootstrap seed (default: 88)
      --out        output PNG path (default:
                   {plots_dir}/extra-plots/task-performance-vs-monitorability/
                   gmean2_by_consistency_level_{model}.png)

Input:
    {outputs_dir}/{model}/{dataset}/{hint}/unhinted.jsonl[.gz]   (parsed_answer)
    {analysis_dir}/{model}/{dataset}/{hint}/results_per_instance.csv

Output:
    {plots_dir}/extra-plots/task-performance-vs-monitorability/
        gmean2_by_consistency_level_{model}.png
"""

import argparse
import csv
import gzip
import json
import sys
from collections import Counter, defaultdict
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
N_RUNS = 8            # unhinted runs per question
K_MIN = 1             # modal count is at least 1 -> consistency floor = 1/8
TEAL = "#2A9D8F"      # same colour step 31 gives the pooled consistency dot

DATASETS = ["mmlu_pro", "gpqa_diamond", "lsat_ar", "lsat_lr",
            "medxpertqa", "lexam", "codemmlu", "wmdp"]


def boot_mean_ci(values, n_resamples, rng):
    arr = np.asarray(values, float)
    if len(arr) < 2:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(arr), size=(n_resamples, len(arr)))
    lo, hi = np.percentile(arr[idx].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def boot_diff_ci(hi_vals, lo_vals, n_resamples, rng):
    h, l = np.asarray(hi_vals, float), np.asarray(lo_vals, float)
    ih = rng.integers(0, len(h), size=(n_resamples, len(h)))
    il = rng.integers(0, len(l), size=(n_resamples, len(l)))
    d = h[ih].mean(axis=1) - l[il].mean(axis=1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(lo), float(hi)


def open_jsonl(base, stem):
    plain, gz = base / f"{stem}.jsonl", base / f"{stem}.jsonl.gz"
    if plain.exists():
        return plain.open()
    if gz.exists():
        return gzip.open(gz, "rt")
    return None


def dataset_consistency(root, config, model_name, dataset, split):
    """{instance_id: modal-answer share} from the shared unhinted arm."""
    answers = defaultdict(list)
    for hint in HINT_TYPES:
        paths = resolve_paths(root, config, dataset, model_name, hint,
                              split=split)
        fh = open_jsonl(paths["outputs_dir"], "unhinted")
        if fh is None:
            continue
        with fh:
            for line in fh:
                r = json.loads(line)
                # Truncated runs are fragments, not answers — excluded to match
                # the pipeline's u_valid rule (see step 31 for the rationale).
                if r.get("truncated", False):
                    continue
                pa = r.get("parsed_answer")
                if pa in (None, "", "N/A"):
                    continue
                answers[str(r["instance_id"])].append(pa)
        break          # unhinted arm is shared across hint types — read once
    return {iid: Counter(a).most_common(1)[0][1] / len(a)
            for iid, a in answers.items() if a}


def main():
    parser = argparse.ArgumentParser(
        description="Level-wise g-mean^2 vs answer consistency, pooled.")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--data", default="test", choices=SPLITS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--no-line", action="store_true")
    parser.add_argument("--min-n", type=int, default=5)
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

    cons_vals, gm_vals, missing = [], [], []
    for dataset in DATASETS:
        consistency = dataset_consistency(root, config, model_name, dataset,
                                          args.data)
        if not consistency:
            missing.append(f"{dataset} (no unhinted arm)")
            continue
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
                    if g in ("", "None", "N/A"):
                        continue
                    iid = str(row["instance_id"])
                    if iid not in consistency:
                        continue
                    cons_vals.append(consistency[iid])
                    gm_vals.append(float(g))
    if missing:
        print("WARNING: skipped: " + ", ".join(missing))
    if not cons_vals:
        print("ERROR: no data found — nothing to plot.")
        sys.exit(1)

    C = np.array(cons_vals)
    G = np.array(gm_vals)
    lev = np.clip(np.round(C * N_RUNS) / N_RUNS, K_MIN / N_RUNS, 1.0)

    print(f"\nPOOLED ({len(DATASETS)} datasets x {len(HINT_TYPES)} hints) — "
          f"{len(G)} entering observations")
    print(f"{'consistency':>18} | {'n':>5} | {'g-mean2':>8} | 95% CI")
    print("-" * 58)
    rows = []
    for k in range(K_MIN, N_RUNS + 1):
        mask = lev == k / N_RUNS
        n = int(mask.sum())
        if n == 0:
            continue
        v = G[mask]
        mean = float(v.mean())
        lo, hi = boot_mean_ci(v, args.resamples, rng)
        drop = " (dropped)" if n < args.min_n else ""
        print(f"{k}/{N_RUNS} = {100*k/N_RUNS:5.1f}% | {n:>5} | {mean:>8.3f} | "
              f"[{lo:.3f}, {hi:.3f}]{drop}")
        if n >= args.min_n:
            rows.append((k / N_RUNS, mean, lo, hi, n))

    r, r_p = stats.pearsonr(C, G)
    rho, rho_p = stats.spearmanr(C, G)
    print(f"\nper-observation Pearson r = {r:+.3f} (p = {r_p:.3g}); "
          f"Spearman rho = {rho:+.3f} (p = {rho_p:.3g})")
    full = lev >= 1.0
    d = float(G[full].mean() - G[~full].mean())
    lo_d, hi_d = boot_diff_ci(G[full], G[~full], args.resamples, rng)
    print(f"fully consistent (100%, n={full.sum()}, mean {G[full].mean():.3f}) "
          f"vs rest (n={(~full).sum()}, mean {G[~full].mean():.3f}): "
          f"Delta = {d:+.3f}, 95% CI [{lo_d:+.3f}, {hi_d:+.3f}]")

    # ---------------- figure ----------------
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#DDDDDD", linewidth=0.8)
    xs = [r_[0] for r_ in rows]
    ys = [r_[1] for r_ in rows]
    los = [r_[2] for r_ in rows]
    his = [r_[3] for r_ in rows]
    y_lo = min(los) - 0.03
    ax.set_ylim(y_lo, max(his) + 0.02)
    if not args.no_line:
        # break the line across any level that was empty or dropped
        gx, gy = [xs[0]], [ys[0]]
        for (x0, _), (x1, y1) in zip(zip(xs, ys), list(zip(xs, ys))[1:]):
            if round((x1 - x0) * N_RUNS) > 1:
                gx.append(np.nan)
                gy.append(np.nan)
            gx.append(x1)
            gy.append(y1)
        ax.plot(gx, gy, color=TEAL, linewidth=1.6, zorder=2)
    ax.errorbar(xs, ys, yerr=[np.array(ys) - np.array(los),
                              np.array(his) - np.array(ys)],
                fmt="o", color=TEAL, markersize=6.5,
                markeredgecolor="#333333", markeredgewidth=0.5,
                ecolor=TEAL, elinewidth=1.3, capsize=2.5, capthick=1.3,
                zorder=3)
    for x, _, _, _, n in rows:
        ax.text(x, y_lo + 0.006, f"{n}", ha="center", va="bottom",
                fontsize=7.5, color="#666666")

    ticks = [k / N_RUNS for k in range(K_MIN, N_RUNS + 1)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{100*k/N_RUNS:.1f}%".replace(".0%", "%")
                        for k in range(K_MIN, N_RUNS + 1)])
    ax.set_xlim(K_MIN / N_RUNS - 0.06, 1.06)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#999999")
    ax.set_xlabel("Answer consistency — share of the 8 unhinted runs giving "
                  "the modal answer\n(12.5% = all 8 answers differ; 100% = the "
                  "same answer every run)")
    ax.set_ylabel("Mean g-mean²")
    ax.set_title(f"Monitorability by answer consistency — {model_name}\n"
                 f"(entering observations pooled over 8 datasets × 3 hint "
                 f"types; bootstrap 95% CI)")

    out = args.out
    if out is None:
        plots_dir = ((root / config["paths"]["plots_dir"]).parent
                     / "extra-plots" / "task-performance-vs-monitorability")
        if args.data != "test":
            plots_dir = plots_dir / args.data
        plots_dir.mkdir(parents=True, exist_ok=True)
        suffix = "_noline" if args.no_line else ""
        out = plots_dir / f"gmean2_by_consistency_level{suffix}_{model_name}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
