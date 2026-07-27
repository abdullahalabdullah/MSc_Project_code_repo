#!/usr/bin/env python3
"""
Step 31: Is monitorability driven by being RIGHT or by being CONSISTENT?

Steps 27/28 established a threshold: a question the model answers correctly
in all N unhinted runs is more monitorable (+0.112 g-mean^2) than one it
does not. But "always correct" logically implies "always the same answer",
so that variable conflates KNOWING the answer with COMMITTING to one.

This step separates them using data already on disk — the per-sample
`parsed_answer` of the unhinted arm — with no new inference:

    consistency_i = (count of the modal unhinted answer) / (valid runs)
    accuracy_i    = acc_u_i  (fraction of valid unhinted runs that are correct)

and splits the entering questions into three groups:

    inconsistent           consistency < 1
    consistent but WRONG   consistency = 1, accuracy = 0
    consistent and CORRECT consistency = 1, accuracy = 1

The base model's answer: consistency does the work. Consistently-WRONG
questions score 0.807 vs 0.710 for inconsistent ones (Delta +0.096,
CI excluding zero), while consistent-correct beats consistent-wrong by
only +0.037 (CI includes zero). Task correctness per se adds little once
commitment is controlled; the step-27 accuracy threshold is largely a
proxy for determinism.

The figure is one horizontal dot-and-CI panel (three rows, n annotated),
with the two decisive contrasts bracketed alongside.

Usage:
    python scripts/31_consistency_vs_correctness.py                  # base
    python scripts/31_consistency_vs_correctness.py --model <hf-id>

    Options:
      --config     path to config YAML (default: config/mvp_config.yaml)
      --data       split: test (default) | eval
      --model      model/adapter HF id (default: Qwen/Qwen3-4B)
      --resamples  bootstrap replicates per CI (default: 5000)
      --seed       bootstrap seed (default: 88)
      --out        output PNG path
                   (default: {plots_dir}/consistency_vs_correctness_{model}.png)

Input:
    {outputs_dir}/{model}/{dataset}/{hint}/unhinted.jsonl[.gz]   (per-sample answers)
    {analysis_dir}/{model}/{dataset}/{hint}/results_per_instance.csv

Output:
    {plots_dir}/consistency_vs_correctness_{model}.png

NOTE: the unhinted arm is shared across hint types, so consistency is read
once per dataset (from the first hint dir that has the file) and joined to
all three hint slices — matching how steps 27/28 pool. A question therefore
contributes up to 3 rows, one per hint type, as in those steps.
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

DATASETS = ["mmlu_pro", "gpqa_diamond", "lsat_ar", "lsat_lr",
            "medxpertqa", "lexam", "codemmlu", "wmdp"]

GROUPS_3 = [
    ("inconsistent", "Inconsistent\n(answers vary across runs)", "#9AA7B4"),
    ("cons_wrong", "Consistent but WRONG\n(same answer 8/8, never correct)",
     "#C1462F"),
    ("cons_right", "Consistent and CORRECT\n(same answer 8/8, always correct)",
     "#1B3A6B"),
]

# two-group merge: the consistency split alone (consistent = wrong + correct)
GROUPS_2 = [
    ("inconsistent", "Inconsistent\n(answers vary across runs)", "#9AA7B4"),
    ("consistent", "Consistent\n(same answer in all 8 runs)", "#1B3A6B"),
]


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
    """unhinted.jsonl or unhinted.jsonl.gz, whichever exists."""
    plain, gz = base / f"{stem}.jsonl", base / f"{stem}.jsonl.gz"
    if plain.exists():
        return plain.open()
    if gz.exists():
        return gzip.open(gz, "rt")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Consistency vs correctness as drivers of monitorability.")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--data", default="test", choices=SPLITS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--merge-consistent", action="store_true",
                        help="two dots — inconsistent vs consistent — instead "
                             "of the three-group breakdown; writes "
                             "…_2group_… .png")
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

    recs, missing, anomalies = [], [], 0
    for dataset in DATASETS:
        # --- consistency from the shared unhinted arm (once per dataset) ---
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
        consistency = {}
        for iid, a in answers.items():
            consistency[iid] = Counter(a).most_common(1)[0][1] / len(a)

        # --- join to each hint slice's per-question g-mean^2 ---
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
                    recs.append((dataset, hint, consistency[iid],
                                 float(a), float(g)))
    if missing:
        print("WARNING: missing inputs skipped: " + ", ".join(missing))
    if not recs:
        print("ERROR: no data found — nothing to plot.")
        sys.exit(1)

    cons = np.array([r[2] for r in recs])
    acc = np.array([r[3] for r in recs])
    gm = np.array([r[4] for r in recs])

    determ = cons >= 0.999
    correct = acc >= 0.999
    # acc = 1 implies consistency = 1; any violation is a validity-set mismatch
    anomalies = int((correct & ~determ).sum())

    GROUPS = GROUPS_2 if args.merge_consistent else GROUPS_3
    if args.merge_consistent:
        masks = {"inconsistent": ~determ, "consistent": determ}
    else:
        masks = {
            "inconsistent": ~determ,
            "cons_wrong": determ & (acc <= 0.001),
            "cons_right": determ & correct,
        }

    print(f"{len(recs)} question-slices "
          f"({len(DATASETS)} datasets x {len(HINT_TYPES)} hint types)\n")
    print(f"{'group':>24} | {'n':>5} | {'g-mean2':>8} | 95% CI")
    print("-" * 60)
    stats_by_group = {}
    for key, label, _ in GROUPS:
        v = gm[masks[key]]
        lo, hi = boot_ci(v, args.resamples, rng)
        stats_by_group[key] = (v, float(v.mean()), lo, hi)
        flat = label.replace("\n", " ")
        print(f"{flat[:24]:>24} | {len(v):>5} | {v.mean():>8.3f} | "
              f"[{lo:.3f}, {hi:.3f}]")

    v_inc = stats_by_group["inconsistent"][0]
    if args.merge_consistent:
        v_con = stats_by_group["consistent"][0]
        d1 = v_con.mean() - v_inc.mean()
        lo1, hi1 = boot_diff_ci(v_con, v_inc, args.resamples, rng)
        # brackets list: one contrast (consistent − inconsistent)
        brackets = [(2, 1, d1, lo1, hi1, "consistency effect", GROUPS[1][2])]
        print(f"\nCONSISTENCY effect (consistent − inconsistent): "
              f"{d1:+.3f} [{lo1:+.3f}, {hi1:+.3f}]")
    else:
        v_cw = stats_by_group["cons_wrong"][0]
        v_cr = stats_by_group["cons_right"][0]
        d1 = v_cw.mean() - v_inc.mean()
        lo1, hi1 = boot_diff_ci(v_cw, v_inc, args.resamples, rng)
        d2 = v_cr.mean() - v_cw.mean()
        lo2, hi2 = boot_diff_ci(v_cr, v_cw, args.resamples, rng)
        brackets = [
            (3, 2, d1, lo1, hi1, "consistency effect", GROUPS[1][2]),
            (2, 1, d2, lo2, hi2, "correctness effect", GROUPS[2][2]),
        ]
        print(f"\nCONSISTENCY effect (consistent-wrong − inconsistent): "
              f"{d1:+.3f} [{lo1:+.3f}, {hi1:+.3f}]")
        print(f"CORRECTNESS effect (consistent-correct − consistent-wrong): "
              f"{d2:+.3f} [{lo2:+.3f}, {hi2:+.3f}]")

    rho_c = stats.spearmanr(cons, gm)[0]
    rho_a = stats.spearmanr(acc, gm)[0]
    print(f"\nSpearman(consistency, g-mean²) = {rho_c:+.3f}")
    print(f"Spearman(accuracy,    g-mean²) = {rho_a:+.3f}")
    if anomalies:
        print(f"\nNOTE: {anomalies} question-slices are 'always correct' but "
              f"not fully consistent ({100 * anomalies / len(recs):.1f}%) — "
              f"acc_u_i's valid-run set differs slightly from the runs with a "
              f"parseable answer. They fall in the 'inconsistent' group.")

    # ---------------- figure ----------------
    GREY, RED, NAVY, TEAL = "#9AA7B4", "#C1462F", "#1B3A6B", "#2A9D8F"

    def dot(mean, y, lo, hi, n, color):
        ax.errorbar(mean, y, xerr=[[mean - lo], [hi - mean]], fmt="o",
                    color=color, ecolor=color, markersize=10,
                    markeredgecolor="#333333", markeredgewidth=0.6,
                    elinewidth=2.0, capsize=4, capthick=2.0, zorder=3)
        ax.text(mean, y + 0.20, f"{mean:.3f}", ha="center", va="bottom",
                fontsize=9.5, color=color, fontweight="bold")
        ax.text(mean, y - 0.22, f"n = {n}", ha="center", va="top",
                fontsize=8.5, color="#777777")

    def bracket(y_lo, y_hi, d, lo, hi, lab, color, x_br):
        ax.plot([x_br, x_br + 0.012, x_br + 0.012, x_br],
                [y_lo, y_lo, y_hi, y_hi], color=color, linewidth=1.2,
                clip_on=False, zorder=4)
        sig = "" if (lo < 0 < hi) else "  ✓"
        ax.text(x_br + 0.022, (y_lo + y_hi) / 2,
                f"{lab}\nΔ = {d:+.3f} [{lo:+.3f}, {hi:+.3f}]{sig}",
                va="center", ha="left", fontsize=9, color=color,
                clip_on=False)

    if args.merge_consistent:
        fig, ax = plt.subplots(figsize=(9.5, 4.6))
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color="#DDDDDD", linewidth=0.8)
        ys = [2, 1]
        for (key, label, color), y in zip(GROUPS, ys):
            v, mean, lo, hi = stats_by_group[key]
            dot(mean, y, lo, hi, len(v), color)
        x_br = max(stats_by_group[k][3] for k, _, _ in GROUPS) + 0.035
        for (y_lo, y_hi, d, lo, hi, lab, color) in brackets:
            bracket(y_lo, y_hi, d, lo, hi, lab, color, x_br)
        ax.set_yticks(ys)
        ax.set_yticklabels([g[1] for g in GROUPS], fontsize=9.5)
        ax.set_ylim(0.45, 2.6)
        lo_all = min(stats_by_group[k][2] for k, _, _ in GROUPS)
        title = ("What predicts monitorability: committing to an answer "
                 f"— {model_name}\nquestions the model answers consistently "
                 "are more monitorable than ones it does not")
    else:
        # ABOVE the line: the three groups — Inconsistent, Consistent-WRONG,
        # Consistent-CORRECT — with the consistency and correctness brackets.
        # BELOW the line: the merged Consistency dot (wrong + correct pooled),
        # showing where the whole committed group lands.
        fig, ax = plt.subplots(figsize=(9.5, 5.4))
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color="#DDDDDD", linewidth=0.8)
        v_con = gm[determ]
        con_m = float(v_con.mean())
        con_lo, con_hi = boot_ci(v_con, args.resamples, rng)
        inc = stats_by_group["inconsistent"]
        cw = stats_by_group["cons_wrong"]
        cr = stats_by_group["cons_right"]
        rows = [
            (4, inc[1], inc[2], inc[3], len(inc[0]), GREY,
             "Inconsistent\n(answers vary across runs)"),
            (3, cw[1], cw[2], cw[3], len(cw[0]), RED,
             "Consistent but WRONG\n(same answer 8/8, never correct)"),
            (2, cr[1], cr[2], cr[3], len(cr[0]), NAVY,
             "Consistent and CORRECT\n(same answer 8/8, always correct)"),
            (1, con_m, con_lo, con_hi, len(v_con), TEAL,
             "Consistency\n(consistent, wrong + correct pooled)"),
        ]
        for y, mean, lo, hi, n, color, _ in rows:
            dot(mean, y, lo, hi, n, color)
        # dashed separator under the three groups; the pooled dot sits below it
        ax.axhline(1.5, color="#CCCCCC", linestyle="--", linewidth=1.2,
                   zorder=1)
        ax.text(0.015, 1.5 - 0.03, "↓ Consistent-WRONG and Consistent-CORRECT "
                "pooled", transform=ax.get_yaxis_transform(),
                ha="left", va="top", fontsize=8.5, color="#999999",
                fontstyle="italic")
        x_br = max(r[3] for r in rows) + 0.035
        # consistency effect: Consistent-WRONG − Inconsistent
        bracket(4, 3, d1, lo1, hi1, "consistency effect", GROUPS[1][2], x_br)
        # correctness effect: Consistent-CORRECT − Consistent-WRONG
        bracket(3, 2, d2, lo2, hi2, "correctness effect", GROUPS[2][2], x_br)
        ax.set_yticks([r[0] for r in rows])
        ax.set_yticklabels([r[6] for r in rows], fontsize=9.5)
        ax.set_ylim(0.4, 4.7)
        lo_all = min(r[2] for r in rows)
        title = ("What predicts monitorability: committing to an answer, "
                 f"not getting it right — {model_name}\n"
                 "consistently-WRONG already beats inconsistent; adding "
                 "correctness barely moves it")

    ax.set_xlim(lo_all - 0.03, x_br + 0.012)
    ax.set_xlabel("Mean per-question g-mean²  (bootstrap 95% CI)")
    ax.set_title(title, fontsize=11)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#999999")

    out = args.out
    if out is None:
        # RO1 bespoke analysis plots live under extra-plots/, not the main plots/ tree
        plots_dir = ((root / config["paths"]["plots_dir"]).parent
                     / "extra-plots" / "task-performance-vs-monitorability")
        if args.data != "test":
            plots_dir = plots_dir / args.data
        plots_dir.mkdir(parents=True, exist_ok=True)
        tag = "_2group" if args.merge_consistent else ""
        out = plots_dir / f"consistency_vs_correctness{tag}_{model_name}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
