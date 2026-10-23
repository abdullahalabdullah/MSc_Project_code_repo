#!/usr/bin/env python3
"""
Step 56: Monitorability by p0 level — how often the model picked the hinted
answer on its own, in the unhinted arm.

    p0_i = (# valid unhinted runs whose answer is the hint target) / 8

x axis = p0 level k/8 (k = 0..7); y = mean per-question g-mean² with a 95%
cluster-bootstrap CI (questions resampled with all their hint rows), n
annotated as "<rows> (<questions>q)". 8/8 cannot occur: p0 = 1 gives
TE = p1 − p0 ≤ 0, so those rows never pass the retention filter.

At k = 0 the rows are split into the two groups of step 31 that exist there:
consistent questions (same unhinted answer in all valid runs — p0 is 0 for
every retained consistent question by construction) and inconsistent
questions that never picked the hinted answer. Every row with k ≥ 1 is
inconsistent. A dotted line marks the mean over all plotted rows.

Only questions with exactly 8 valid unhinted runs are used, so that k/8 is
exact (truncated / unparsed runs excluded, as in step 31). Hint types and
datasets are pooled row-wise (a question contributes up to 3 rows).

Usage:
    python scripts/56_gmean2_by_p0_level.py                 # base
    python scripts/56_gmean2_by_p0_level.py --model <hf-id>

    Options:
      --config     path to config YAML (default: config/mvp_config.yaml)
      --data       split: test (default) | eval
      --model      model id (default: Qwen/Qwen3-4B)
      --resamples  cluster-bootstrap replicates per CI (default: 10000)
      --seed       bootstrap seed (default: 88)
      --out        output PNG path

Input:
    {outputs_dir}/{model}/{dataset}/{hint}/unhinted.jsonl[.gz]
    {analysis_dir}/{model}/{dataset}/{hint}/results_per_instance.csv

Output:
    {plots_dir}/../extra-plots/task-performance-vs-monitorability/
        gmean2_by_p0_level_{model}.png
"""

import argparse
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import yaml

from src.dataset import SPLITS, model_dir_name, resolve_paths
from src.hints import HINT_TYPES

DEFAULT_MODEL = "Qwen/Qwen3-4B"
DATASETS = ["mmlu_pro", "gpqa_diamond", "lsat_ar", "lsat_lr",
            "medxpertqa", "lexam", "codemmlu", "wmdp"]
N_RUNS = 8

# same colours as step 31
GREY, TEAL = "#9AA7B4", "#2A9D8F"


def open_jsonl(base, stem):
    plain, gz = base / f"{stem}.jsonl", base / f"{stem}.jsonl.gz"
    if plain.exists():
        return plain.open()
    if gz.exists():
        return gzip.open(gz, "rt")
    return None


def load_rows(root, config, model_name, split):
    """Retained (question, hint) rows: dict(q, k, consistent, g)."""
    rows = []
    for dataset in DATASETS:
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
                    if r.get("truncated", False):
                        continue
                    pa = r.get("parsed_answer")
                    if pa in (None, "", "N/A"):
                        continue
                    answers[int(r["instance_id"])].append(pa)
            break  # shared unhinted arm
        for hint in HINT_TYPES:
            paths = resolve_paths(root, config, dataset, model_name, hint,
                                  split=split)
            csv_path = paths["analysis_dir"] / "results_per_instance.csv"
            if not csv_path.exists():
                print(f"WARNING: missing {csv_path}")
                continue
            with csv_path.open() as f:
                for row in csv.DictReader(f):
                    g = row.get("gmean2_i", "")
                    if g in ("", "None", "N/A"):
                        continue
                    iid = int(row["instance_id"])
                    a = answers.get(iid, [])
                    if len(a) != N_RUNS:
                        continue  # k/8 must be exact
                    k = sum(x == row["hint_target"] for x in a)
                    rows.append(dict(q=(dataset, iid), k=k,
                                     consistent=len(set(a)) == 1,
                                     g=float(g)))
    return rows


def cluster_boot_means(qid, gm, masks, n_resamples, rng, chunk=1000):
    """Cluster bootstrap over questions (as steps 31/33): resampled mean of
    each mask, all masks sharing the same draws. (n_resamples, len(masks))."""
    nq = int(qid.max()) + 1
    S = np.stack([np.bincount(qid, weights=gm * m, minlength=nq)
                  for m in masks], axis=1)
    N = np.stack([np.bincount(qid, weights=m.astype(float), minlength=nq)
                  for m in masks], axis=1)
    out, done = [], 0
    while done < n_resamples:
        b = min(chunk, n_resamples - done)
        picks = rng.integers(0, nq, (b, nq))
        W = np.stack([np.bincount(row, minlength=nq) for row in picks])
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append((W @ S) / (W @ N))
        done += b
    return np.vstack(out)


def main():
    parser = argparse.ArgumentParser(
        description="Monitorability by p0 level (unhinted picks of the "
                    "hinted answer).")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--data", default="test", choices=SPLITS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--resamples", type=int, default=10000)
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

    rows = load_rows(root, config, model_name, args.data)
    if not rows:
        print("ERROR: no data found — nothing to plot.")
        sys.exit(1)
    gm = np.array([r["g"] for r in rows])
    k = np.array([r["k"] for r in rows])
    cons = np.array([r["consistent"] for r in rows])
    qkeys = {q: j for j, q in enumerate(dict.fromkeys(r["q"] for r in rows))}
    qid = np.array([qkeys[r["q"]] for r in rows])

    # groups: (label, mask, x, colour)
    levels = [lv for lv in range(N_RUNS) if (k == lv).any()]
    groups = [("consistent", (k == 0) & cons, -0.16, TEAL),
              ("inconsistent 0/8", (k == 0) & ~cons, 0.16, GREY)]
    groups += [(f"{lv}/8", k == lv, float(lv), GREY) for lv in levels if lv]
    boot = cluster_boot_means(qid, gm, [g[1] for g in groups],
                              args.resamples, rng)

    print(f"{len(rows)} rows ({len(qkeys)} questions) with exactly "
          f"{N_RUNS} valid unhinted runs\n")
    print(f"{'group':>18} | {'rows':>5} | {'q':>4} | {'g-mean2':>7} | 95% CI")
    print("-" * 58)
    stats = []
    for j, (name, m, x, color) in enumerate(groups):
        mean = float(gm[m].mean())
        lo = float(np.nanpercentile(boot[:, j], 2.5))
        hi = float(np.nanpercentile(boot[:, j], 97.5))
        n, nq = int(m.sum()), len(set(qid[m]))
        stats.append((name, x, mean, lo, hi, n, nq, color))
        print(f"{name:>18} | {n:>5} | {nq:>4} | {mean:>7.3f} | "
              f"[{lo:.3f}, {hi:.3f}]")
    overall = float(gm.mean())
    print(f"\noverall mean (all plotted rows): {overall:.3f}")

    # ---------------- figure ----------------
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#E5E5E5", linewidth=0.8)
    ax.axhline(overall, color="#888888", linestyle=":", linewidth=1.2,
               zorder=1)
    # label sits between 1/8 and 2/8, where no CI reaches the line
    ax.text(1.5, overall + 0.006, f"overall mean {overall:.3f}",
            ha="center", va="bottom", fontsize=8.5, color="#666666")
    # connecting line through the inconsistent dots (k = 0 .. 7)
    inc = [s for s in stats if s[0] != "consistent"]
    ax.plot([s[1] for s in inc], [s[2] for s in inc], color=GREY,
            linewidth=1.3, zorder=2)
    for name, x, mean, lo, hi, n, nq, color in stats:
        ax.errorbar(x, mean, yerr=[[mean - lo], [hi - mean]], fmt="o",
                    color=color, ecolor=color, markersize=9,
                    markeredgecolor="#333333", markeredgewidth=0.6,
                    elinewidth=1.8, capsize=3.5, capthick=1.8, zorder=3)
        ax.text(x, hi + 0.012, f"{mean:.3f}", ha="center", va="bottom",
                fontsize=8.5, color="#333333", fontweight="bold")
        ax.text(x, lo - 0.012, f"{n}\n({nq}q)", ha="center", va="top",
                fontsize=7.5, color="#555555")
    ax.set_xticks(range(N_RUNS))
    ax.set_xticklabels([f"{lv}/8" for lv in range(N_RUNS)])
    ax.set_xlim(-0.6, N_RUNS - 0.4)
    lo_all = min(s[3] for s in stats)
    hi_all = max(s[4] for s in stats)
    ax.set_ylim(lo_all - 0.09, hi_all + 0.05)
    ax.set_xlabel("p0 — unhinted runs (of 8) in which the model picked the "
                  "hinted answer")
    ax.set_ylabel("Monitorability (g-mean²)")
    ax.set_title("Monitorability vs p0 (how often the model picks the hinted "
                 "answer without the hint)", fontsize=12)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#999999")
    handles = [
        Line2D([], [], marker="o", linestyle="none", color=TEAL,
               markeredgecolor="#333333", markersize=9,
               label="Consistent (same answer in all 8 runs; p0 = 0)"),
        Line2D([], [], marker="o", linestyle="-", color=GREY,
               markeredgecolor="#333333", markersize=9,
               label="Inconsistent (answers vary across runs)"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8.5,
              frameon=True, edgecolor="#CCCCCC")
    fig.text(0.01, -0.03,
             "Rows = question × hint (all datasets and hint types pooled); "
             "only questions with exactly 8 valid unhinted runs. Labels: "
             "rows (distinct questions).\n95% CIs: cluster bootstrap over "
             f"questions ({args.resamples:,} resamples). 8/8 cannot occur: "
             "p0 = 1 gives TE ≤ 0, so such rows fail the retention filter.",
             fontsize=8, color="#555555", ha="left", va="top")

    out = args.out
    if out is None:
        plots_dir = ((root / config["paths"]["plots_dir"]).parent
                     / "extra-plots" / "task-performance-vs-monitorability")
        if args.data != "test":
            plots_dir = plots_dir / args.data
        plots_dir.mkdir(parents=True, exist_ok=True)
        out = plots_dir / f"gmean2_by_p0_level_{model_name}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
