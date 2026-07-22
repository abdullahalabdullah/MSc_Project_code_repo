#!/usr/bin/env python3
"""
Step 34: With the committed (consistent) questions removed, does
monitorability track task accuracy? Pooled over all datasets × hint types
for ONE model.

Steps 27/31 show the accuracy "threshold" is really a determinism effect:
the only questions that are more monitorable are the ones the model answers
the SAME way in all N runs (consistent). This step removes those and asks
whether, among the questions where the model is genuinely uncertain, higher
accuracy predicts monitorability. It does not — the residual association is
negligible, which is the complement of the consistency result.

Filter (--filter):
  inconsistent (default) — keep questions with consistency < 1 (the model
      does NOT give the same answer in all N runs). This drops both
      "always correct" (8/8, consistent) and "always the same wrong answer"
      (consistent-wrong); a handful of acc=1 anomalies (valid-run mismatch,
      see step 31) are also dropped. Matches "remove the consistent cases".
  partial-acc — keep questions with unhinted accuracy strictly between 0/8
      and 8/8 (drop BOTH accuracy endpoints regardless of consistency).

consistency_i = modal unhinted-answer share over the valid unhinted runs,
read once per dataset from the shared unhinted arm (no new inference), as in
steps 31/33.

Figure: a level plot (dots + bootstrap 95% CI, n under each dot, connecting
line) of mean gmean2_i at each retained unhinted-accuracy level k/8, with a
dotted line at the retained overall mean and a Pearson/Spearman annotation
over the retained per-question points.

Usage:
    python scripts/34_acc_monitorability_inconsistent.py                 # base
    python scripts/34_acc_monitorability_inconsistent.py --filter partial-acc

    Options:
      --config     path to config YAML (default: config/mvp_config.yaml)
      --data       split: test (default) | eval
      --model      model/adapter HF id (default: Qwen/Qwen3-4B)
      --filter     inconsistent (default) | partial-acc
      --min-n      min questions per level to draw the dot (default: 15)
      --resamples  bootstrap replicates per CI (default: 5000)
      --seed       bootstrap seed (default: 88)
      --out        output PNG path (default:
                   {plots_dir}/task-performance-vs-monitorability/
                   acc_vs_monitorability_{filter}_{model}.png)

Input:
    {outputs_dir}/{model}/{dataset}/{hint}/unhinted.jsonl[.gz]
    {analysis_dir}/{model}/{dataset}/{hint}/results_per_instance.csv

Output:
    {plots_dir}/task-performance-vs-monitorability/
        acc_vs_monitorability_{filter}_{model}.png
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

DARK_BLUE = "#1B3A6B"
REF_COLOR = "#C1462F"


def boot_ci(v, n_resamples, rng):
    v = np.asarray(v, float)
    if len(v) < 2:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(v), (n_resamples, len(v)))
    m = v[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_diff_ci(a, b, n_resamples, rng):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    ia = rng.integers(0, len(a), (n_resamples, len(a)))
    ib = rng.integers(0, len(b), (n_resamples, len(b)))
    d = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def open_jsonl(base, stem):
    plain, gz = base / f"{stem}.jsonl", base / f"{stem}.jsonl.gz"
    if plain.exists():
        return plain.open()
    if gz.exists():
        return gzip.open(gz, "rt")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Accuracy vs monitorability with committed questions "
                    "removed.")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--data", default="test", choices=SPLITS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--filter", default="inconsistent",
                        choices=["inconsistent", "partial-acc"])
    parser.add_argument("--style", default="level",
                        choices=["level", "two-dot", "levels-h"],
                        help="level = vertical per-accuracy-level dot plot; "
                             "two-dot = horizontal mostly-incorrect vs "
                             "mostly-correct contrast (step-31 style); "
                             "levels-h = the step-31 horizontal style but one "
                             "row per accuracy level 0/8..7/8")
    parser.add_argument("--acc-split", type=int, default=4,
                        help="two-dot: k>=this many correct runs is the "
                             "'mostly correct' group (default 4, i.e. >=0.5)")
    parser.add_argument("--min-n", type=int, default=15)
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

    # ---- pooled per-question (consistency, acc, gmean2) ----
    recs, missing = [], []
    for dataset in DATASETS:
        answers = defaultdict(list)
        for hint in HINT_TYPES:
            paths = resolve_paths(root, config, dataset, model_name, hint,
                                  split=args.data)
            fh = open_jsonl(paths["outputs_dir"], "unhinted")
            if fh is None:
                continue
            with fh:
                for line in fh:
                    r = json.loads(line)
                    pa = r.get("parsed_answer")
                    if pa in (None, "", "N/A"):
                        continue
                    answers[r["instance_id"]].append(pa)
            break
        if not answers:
            missing.append(f"{dataset} (no unhinted arm)")
            continue
        consistency = {iid: Counter(a).most_common(1)[0][1] / len(a)
                       for iid, a in answers.items()}
        for hint in HINT_TYPES:
            paths = resolve_paths(root, config, dataset, model_name, hint,
                                  split=args.data)
            csv_path = paths["analysis_dir"] / "results_per_instance.csv"
            if not csv_path.exists():
                missing.append(f"{dataset}/{hint}")
                continue
            with csv_path.open() as f:
                for row in csv.DictReader(f):
                    g, a = row.get("gmean2_i", ""), row.get("acc_u_i", "")
                    if g in ("", "None", "N/A") or a in ("", "None", "N/A"):
                        continue
                    iid = int(row["instance_id"])
                    if iid not in consistency:
                        continue
                    recs.append((consistency[iid], float(a), float(g)))
    if missing:
        print("WARNING: missing inputs skipped: " + ", ".join(missing))
    if not recs:
        print("ERROR: no data found — nothing to plot.")
        sys.exit(1)

    cons = np.array([r[0] for r in recs])
    acc = np.array([r[1] for r in recs])
    gm = np.array([r[2] for r in recs])
    lev = np.round(acc * N_RUNS).astype(int)

    if args.filter == "inconsistent":
        keep = (cons < 0.999) & (acc < 0.999)
        removed_label = ("consistent questions (same answer in all "
                         f"{N_RUNS} runs)")
        keep_label = "inconsistent questions"
    else:  # partial-acc
        keep = (lev > 0) & (lev < N_RUNS)
        removed_label = f"questions answered 0/{N_RUNS} or {N_RUNS}/{N_RUNS}"
        keep_label = f"questions with 1–{N_RUNS - 1} of {N_RUNS} correct"

    a_k, g_k, l_k = acc[keep], gm[keep], lev[keep]
    n_keep = int(keep.sum())
    r, r_p = stats.pearsonr(a_k, g_k)
    rho, rho_p = stats.spearmanr(a_k, g_k)

    print(f"pooled entering: {len(recs)}   removed: {removed_label}")
    print(f"retained ({keep_label}): n = {n_keep}")
    print(f"  Pearson r(acc, g-mean²) = {r:+.3f} (p = {r_p:.3g})")
    print(f"  Spearman rho            = {rho:+.3f} (p = {rho_p:.3g})")

    # ================= two-dot style (step-31 look) =================
    if args.style == "two-dot":
        s = args.acc_split
        lo_mask = l_k < s
        hi_mask = l_k >= s
        v_lo, v_hi = g_k[lo_mask], g_k[hi_mask]
        lo_ci = boot_ci(v_lo, args.resamples, rng)
        hi_ci = boot_ci(v_hi, args.resamples, rng)
        d = float(v_hi.mean() - v_lo.mean())
        dlo, dhi = boot_diff_ci(v_hi, v_lo, args.resamples, rng)
        lo_lab = f"Mostly incorrect\n(0–{s - 1} of {N_RUNS} correct)"
        hi_lab = f"Mostly correct\n({s}–{N_RUNS - 1} of {N_RUNS} correct)"
        print(f"\ntwo-dot split at k={s} (among {keep_label}):")
        print(f"  mostly incorrect  n={lo_mask.sum():5d}  mean {v_lo.mean():.3f}"
              f"  CI [{lo_ci[0]:.3f}, {lo_ci[1]:.3f}]")
        print(f"  mostly correct    n={hi_mask.sum():5d}  mean {v_hi.mean():.3f}"
              f"  CI [{hi_ci[0]:.3f}, {hi_ci[1]:.3f}]")
        print(f"  accuracy effect (correct − incorrect): {d:+.3f} "
              f"[{dlo:+.3f}, {dhi:+.3f}]"
              f"  {'(CI excludes 0)' if not (dlo < 0 < dhi) else '(n.s.)'}")

        rows = [(lo_lab, v_lo.mean(), lo_ci[0], lo_ci[1], int(lo_mask.sum()),
                 "#9AA7B4", 2),
                (hi_lab, v_hi.mean(), hi_ci[0], hi_ci[1], int(hi_mask.sum()),
                 "#1B3A6B", 1)]
        fig, ax = plt.subplots(figsize=(9.5, 4.6))
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color="#DDDDDD", linewidth=0.8)
        for lab, mean, clo, chi, n, color, y in rows:
            ax.errorbar(mean, y, xerr=[[mean - clo], [chi - mean]], fmt="o",
                        color=color, ecolor=color, markersize=10,
                        markeredgecolor="#333333", markeredgewidth=0.6,
                        elinewidth=2.0, capsize=4, capthick=2.0, zorder=3)
            ax.text(mean, y + 0.20, f"{mean:.3f}", ha="center", va="bottom",
                    fontsize=9.5, color=color, fontweight="bold")
            ax.text(mean, y - 0.22, f"n = {n}", ha="center", va="top",
                    fontsize=8.5, color="#777777")
        x_br = max(r[3] for r in rows) + 0.006
        ax.plot([x_br, x_br + 0.004, x_br + 0.004, x_br], [2, 2, 1, 1],
                color="#1B3A6B", linewidth=1.2, clip_on=False, zorder=4)
        sig = "  ✓" if not (dlo < 0 < dhi) else ""
        ax.text(x_br + 0.010, 1.5,
                f"accuracy effect\nΔ = {d:+.3f} [{dlo:+.3f}, {dhi:+.3f}]{sig}",
                va="center", ha="left", fontsize=9, color="#1B3A6B",
                clip_on=False)
        ax.set_yticks([2, 1])
        ax.set_yticklabels([rows[0][0], rows[1][0]], fontsize=9.5)
        ax.set_ylim(0.45, 2.6)
        lo_all = min(r[2] for r in rows)
        ax.set_xlim(lo_all - 0.02, x_br + 0.004)
        ax.set_xlabel("Mean per-question g-mean²  (bootstrap 95% CI)")
        ax.set_title("Among inconsistent questions, accuracy barely moves "
                     f"monitorability — {model_name}\n"
                     "(committed questions removed; the accuracy gap is small "
                     "and largely residual commitment)", fontsize=11)
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
            tag = args.filter.replace("-", "_")
            out = (plots_dir /
                   f"acc_vs_monitorability_{tag}_twodot_{model_name}.png")
        fig.tight_layout()
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"\nWrote {out}")
        return

    # ============ horizontal one row per accuracy level ============
    if args.style == "levels-h":
        levels = []  # (k, mean, lo, hi, n)
        for k in range(N_RUNS + 1):
            m = l_k == k
            n = int(m.sum())
            if n < args.min_n:
                continue
            v = g_k[m]
            clo, chi = boot_ci(v, args.resamples, rng)
            levels.append((k, float(v.mean()), clo, chi, n))
        if len(levels) < 2:
            print("ERROR: fewer than 2 populated levels — nothing to plot.")
            sys.exit(1)

        cmap = plt.get_cmap("Blues")
        nlev = len(levels)
        colors = [cmap(0.30 + 0.62 * i / max(nlev - 1, 1))
                  for i in range(nlev)]
        # top row = lowest accuracy (0/8); descend to highest
        ys = list(range(nlev, 0, -1))

        fig, ax = plt.subplots(figsize=(9.5, 6.2))
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color="#DDDDDD", linewidth=0.8)
        overall = float(g_k.mean())
        ax.axvline(overall, color="#C1462F", linestyle=":", linewidth=1.3,
                   zorder=1)
        ax.text(overall, nlev + 0.55, f"overall mean {overall:.3f}",
                color="#C1462F", fontsize=8.5, ha="center", va="bottom")
        for (k, mean, clo, chi, n), color, y in zip(levels, colors, ys):
            ax.errorbar(mean, y, xerr=[[mean - clo], [chi - mean]], fmt="o",
                        color=color, ecolor=color, markersize=9,
                        markeredgecolor="#333333", markeredgewidth=0.6,
                        elinewidth=1.8, capsize=4, capthick=1.8, zorder=3)
            ax.text(mean, y + 0.22, f"{mean:.3f}", ha="center", va="bottom",
                    fontsize=9, color=color, fontweight="bold")
            ax.text(mean, y - 0.24, f"n = {n}", ha="center", va="top",
                    fontsize=8, color="#888888")
        ax.set_yticks(ys)
        ax.set_yticklabels([f"{k}/{N_RUNS} correct" for k, *_ in levels],
                           fontsize=9.5)
        ax.set_ylim(0.4, nlev + 1.7)
        all_b = [b for _, _, lo, hi, _ in levels for b in (lo, hi)]
        ax.set_xlim(min(all_b) - 0.02, max(all_b) + 0.02)
        p_r = "p < 0.001" if r_p < 0.001 else f"p = {r_p:.3f}"
        p_rho = "p < 0.001" if rho_p < 0.001 else f"p = {rho_p:.3f}"
        ax.text(0.015, 0.985,
                f"{keep_label}, n = {n_keep}\n"
                f"Pearson r = {r:+.2f} ({p_r});  "
                f"Spearman ρ = {rho:+.2f} ({p_rho})",
                transform=ax.transAxes, fontsize=9, va="top", ha="left",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="#CCCCCC",
                          pad=4))
        ax.set_xlabel("Mean per-question g-mean²  (bootstrap 95% CI)")
        ax.set_ylabel("Unhinted task accuracy of the question")
        ax.set_title("Monitorability by accuracy level among inconsistent "
                     f"questions — {model_name}\n"
                     f"(committed questions removed; {removed_label} not "
                     "shown; hint types pooled)", fontsize=11)
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
            tag = args.filter.replace("-", "_")
            out = (plots_dir /
                   f"acc_vs_monitorability_{tag}_levels_h_{model_name}.png")
        fig.tight_layout()
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"\nWrote {out}")
        return
    # ================= level style (default) =================
    print(f"\n{'acc level':>9} | {'n':>5} | {'g-mean2':>8} | 95% CI")
    print("-" * 48)
    xs, ys, los, his, ns = [], [], [], [], []
    for k in range(N_RUNS + 1):
        m = l_k == k
        n = int(m.sum())
        if n == 0:
            continue
        v = g_k[m]
        lo, hi = boot_ci(v, args.resamples, rng)
        drop = " (dropped)" if n < args.min_n else ""
        print(f"{k:>7}/8 | {n:>5} | {v.mean():>8.3f} | "
              f"[{lo:.3f}, {hi:.3f}]{drop}")
        if n >= args.min_n:
            xs.append(k / N_RUNS)
            ys.append(float(v.mean()))
            los.append(lo)
            his.append(hi)
            ns.append(n)
    if len(xs) < 2:
        print("ERROR: fewer than 2 populated levels — nothing to plot.")
        sys.exit(1)

    overall = float(g_k.mean())

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#DDDDDD", linewidth=0.8)

    xs = np.array(xs)
    ax.plot(xs, ys, color=DARK_BLUE, linewidth=1.6, zorder=2)
    ax.errorbar(xs, ys, yerr=[np.array(ys) - np.array(los),
                              np.array(his) - np.array(ys)],
                fmt="o", color=DARK_BLUE, markersize=7,
                markeredgecolor="#333333", markeredgewidth=0.5,
                ecolor=DARK_BLUE, elinewidth=1.4, capsize=3, capthick=1.4,
                zorder=3)
    ax.axhline(overall, color=REF_COLOR, linestyle=":", linewidth=1.4,
               zorder=1)
    ax.text(xs.max(), overall, f"  overall mean = {overall:.3f}",
            color=REF_COLOR, fontsize=9, va="bottom", ha="left")
    for x, n in zip(xs, ns):
        ax.annotate(f"{n}", xy=(x, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    va="bottom", fontsize=8, color="#777777")

    p_r = "p < 0.001" if r_p < 0.001 else f"p = {r_p:.3f}"
    p_rho = "p < 0.001" if rho_p < 0.001 else f"p = {rho_p:.3f}"
    ax.text(0.03, 0.97,
            f"{keep_label}, n = {n_keep}\n"
            f"Pearson r = {r:+.2f} ({p_r})\n"
            f"Spearman ρ = {rho:+.2f} ({p_rho})",
            transform=ax.transAxes, fontsize=9.5, va="top", ha="left",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="#CCCCCC",
                      pad=4))

    ax.set_xticks([k / N_RUNS for k in range(N_RUNS + 1)])
    ax.set_xticklabels([f"{k}/8" for k in range(N_RUNS + 1)])
    ax.set_xlim(-0.06, 1.06)
    ax.set_xlabel("Unhinted task accuracy of the question (correct runs / 8)")
    ax.set_ylabel("Mean per-question g-mean²")
    ax.set_title("With committed questions removed, accuracy still does not "
                 f"predict monitorability — {model_name}\n"
                 f"(removed: {removed_label}; hint types pooled; "
                 f"bootstrap 95% CI)", fontsize=11)
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
        tag = args.filter.replace("-", "_")
        out = plots_dir / f"acc_vs_monitorability_{tag}_{model_name}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
