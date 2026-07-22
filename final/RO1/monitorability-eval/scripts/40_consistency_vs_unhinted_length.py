#!/usr/bin/env python3
"""
Step 40: Does answer CONSISTENCY correlate with unhinted response LENGTH?
Pooled over all datasets.

This is the confound behind the length->monitorability result (steps 37-39):
monitorability rises with consistency (step 31) AND falls with length
(steps 37-39), so before reading length as its own driver we must show that
consistency and length are themselves related. They are, strongly and
negatively: questions the model answers consistently are markedly SHORTER.

Unit = the QUESTION. From the shared unhinted arm (8 runs/question, read once
per dataset), consistency_i = modal-answer share over the valid runs; length =
`mean_cot_tokens_unhinted` (the mean of those 8 runs). Consistency is
quasi-discrete (modal count k out of 8 -> k/8), so we group questions by that
level and plot the per-level MEDIAN length (length is right-skewed, so median
not mean) with a bootstrap 95% CI, on a log y-axis. The pooled Spearman on the
raw per-question values is shown in a corner box.

Base model (test): median unhinted length falls monotonically as consistency
rises — 3/8: ~3260 tok ... 7/8: ~2390 ... 8/8: ~1120 — Spearman(consistency,
length) = -0.445 (p ~ 1e-78); consistent (mode=1) questions are ~2.4x shorter
than inconsistent ones (median 1122 vs 2690 tok). So the shortest, most-
monitorable questions are disproportionately the consistent ones: unhinted
length is an indirect symptom of commitment, not an independent axis.

Usage:
    python scripts/40_consistency_vs_unhinted_length.py            # base
    python scripts/40_consistency_vs_unhinted_length.py --model <hf-id>

    Options:
      --config     path to config YAML (default: config/mvp_config.yaml)
      --data       split: test (default) | eval
      --model      model/adapter HF id (default: Qwen/Qwen3-4B)
      --min-n      min questions per consistency level to draw it (default: 5)
      --resamples  bootstrap replicates per CI / rho (default: 5000)
      --seed       bootstrap seed (default: 88)
      --out        output PNG path (default:
                   {plots_dir}/cot-length-vs-monitorability/
                   consistency_vs_unhinted_length_{model}.png)

Input:
    {outputs_dir}/{model}/{dataset}/{hint}/unhinted.jsonl[.gz]   (parsed_answer)
    {analysis_dir}/{model}/{dataset}/{hint}/results_per_instance.csv

Output:
    {plots_dir}/cot-length-vs-monitorability/
        consistency_vs_unhinted_length_{model}.png
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
N_RUNS = 8
DATASETS = ["mmlu_pro", "gpqa_diamond", "lsat_ar", "lsat_lr",
            "medxpertqa", "lexam", "codemmlu", "wmdp"]
TEAL = "#2A9D8F"


def open_jsonl(base, stem):
    plain, gz = base / f"{stem}.jsonl", base / f"{stem}.jsonl.gz"
    if plain.exists():
        return plain.open()
    if gz.exists():
        return gzip.open(gz, "rt")
    return None


def boot_ci_mean(v, n_resamples, rng):
    v = np.asarray(v, float)
    if len(v) < 2:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(v), (n_resamples, len(v)))
    m = v[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    parser = argparse.ArgumentParser(
        description="Consistency vs unhinted response length, pooled.")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--data", default="test", choices=SPLITS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--bins", type=int, default=5,
                        help="equal-count length quantile bins (default: 5)")
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

    cons, length, missing = [], [], []
    for dataset in DATASETS:
        # consistency from the shared unhinted arm (once per dataset)
        answers = defaultdict(list)
        for hint in HINT_TYPES:
            base = resolve_paths(root, config, dataset, model_name, hint,
                                 split=args.data)["outputs_dir"]
            fh = open_jsonl(base, "unhinted")
            if fh is None:
                continue
            with fh:
                for line in fh:
                    r = json.loads(line)
                    pa = r.get("parsed_answer")
                    if pa in (None, "", "N/A"):
                        continue
                    answers[str(r["instance_id"])].append(pa)
            break
        if not answers:
            missing.append(f"{dataset} (no unhinted arm)")
            continue
        consistency = {iid: Counter(a).most_common(1)[0][1] / len(a)
                       for iid, a in answers.items()}
        # per-question unhinted length
        for hint in HINT_TYPES:
            cp = resolve_paths(root, config, dataset, model_name, hint,
                               split=args.data)["analysis_dir"] / \
                "results_per_instance.csv"
            if not cp.exists():
                continue
            with cp.open() as f:
                for row in csv.DictReader(f):
                    L = row.get("mean_cot_tokens_unhinted", "")
                    if L in ("", "None", "N/A"):
                        continue
                    iid = str(row["instance_id"])
                    if iid in consistency:
                        cons.append(consistency[iid])
                        length.append(float(L))
            break
    if missing:
        print("WARNING: skipped: " + ", ".join(missing))
    if not cons:
        print("ERROR: no data found — nothing to plot.")
        sys.exit(1)

    c = np.array(cons)
    L = np.array(length)
    n = len(c)

    rho = stats.spearmanr(c, L)[0]
    bs = np.empty(args.resamples)
    for j in range(args.resamples):
        idx = rng.integers(0, n, n)
        bs[j] = stats.spearmanr(c[idx], L[idx])[0]
    rlo, rhi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)

    print(f"n = {n}")
    print(f"Spearman(consistency, unhinted length) = {rho:+.3f} "
          f"[{rlo:+.3f}, {rhi:+.3f}]")

    # equal-count length quantile bins on x; mean consistency on y
    nb = args.bins
    edges = np.percentile(L, np.linspace(0, 100, nb + 1))
    edges[-1] = np.nextafter(edges[-1], np.inf)
    xs, ys, elo, ehi, ns, ranges = [], [], [], [], [], []
    print(f"\n{'length range (tok)':>19} | {'n':>4} | {'med len':>7} | "
          f"{'mean cons':>9} | 95% CI")
    print("-" * 66)
    for i in range(nb):
        m = (L >= edges[i]) & (L < edges[i + 1])
        if m.sum() == 0:
            continue
        v = c[m]
        med = float(np.median(L[m]))
        lo, hi = boot_ci_mean(v, args.resamples, rng)
        rg = f"{int(edges[i])}-{int(np.ceil(edges[i+1]))}"
        print(f"{rg:>19} | {int(m.sum()):>4} | {med:>7.0f} | "
              f"{v.mean():>9.3f} | [{lo:.3f}, {hi:.3f}]")
        xs.append(med)
        ys.append(float(v.mean()))
        elo.append(v.mean() - lo)
        ehi.append(hi - v.mean())
        ns.append(int(m.sum()))
        ranges.append(rg)

    # ---------------- figure ----------------
    use_log = not args.linear_x
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.set_axisbelow(True)
    if use_log:
        ax.set_xscale("log")
        ax.set_xlim(edges[0] * 0.85, edges[-1] * 1.15)
    ax.grid(True, color="#EEEEEE", linewidth=0.8)
    ax.errorbar(xs, ys, yerr=[elo, ehi], color=TEAL, linewidth=2.0,
                marker="o", markersize=9, markeredgecolor="#333333",
                markeredgewidth=0.5, ecolor=TEAL, elinewidth=1.4,
                capsize=4, capthick=1.4, zorder=3)
    for x, y, eh, nn, rg in zip(xs, ys, ehi, ns, ranges):
        ax.annotate(f"n={nn}", xy=(x, y + eh), xytext=(0, 6),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=8.5, color="#555555")
        ax.annotate(rg, xy=(x, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    va="bottom", fontsize=7.5, color="#999999")

    ax.text(0.03, 0.06,
            f"n = {n}\nSpearman ρ = {rho:+.3f} [{rlo:+.3f}, {rhi:+.3f}]",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9.5,
            color="#333333",
            bbox=dict(boxstyle="round,pad=0.45", fc="#F4F6F9", ec="#CCD3DB"))

    ax.set_xlabel("Unhinted response length — per-question mean of 8 runs "
                  f"(tokens{', log scale' if use_log else ''}; dots at bin "
                  "median)")
    ax.set_ylabel("Answer consistency — modal-answer share over 8 runs\n"
                  "mean per bin · bars: bootstrap 95% CI")
    ax.set_title("Longer reasoning is LESS consistent — "
                 f"{model_name}\n"
                 "pooled over 8 datasets; the length↔monitorability effect is "
                 "confounded with consistency")
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
        out = plots_dir / f"consistency_vs_unhinted_length_{model_name}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
