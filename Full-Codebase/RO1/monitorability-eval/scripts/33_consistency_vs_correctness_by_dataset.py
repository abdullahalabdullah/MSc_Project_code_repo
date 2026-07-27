#!/usr/bin/env python3
"""
Step 33: Per-dataset version of step 31 — is monitorability driven by being
RIGHT or by being CONSISTENT, shown separately for each dataset.

Step 31 pools all 8 datasets into one three-group dot plot (inconsistent /
consistent-but-WRONG / consistent-and-CORRECT). This step draws the SAME
three groups as an 8-panel small-multiple (2x4, domain-table order, shared
x-axis), so the reader can see whether the "commitment, not correctness"
pattern holds dataset by dataset or is carried by a few.

Grouping (identical to step 31, per dataset):
    consistency_i = modal unhinted answer share over the valid unhinted runs
    inconsistent           consistency < 1
    consistent but WRONG   consistency = 1, accuracy = 0
    consistent and CORRECT consistency = 1, accuracy = 1

Per (dataset, group): mean gmean2_i with a percentile-bootstrap 95% CI, n
annotated; groups with n < --min-n are dropped from that panel (the
consistent-WRONG group is the thin one — pooled n=281, so per dataset it is
often small; the guard keeps noisy dots out). Stdout prints, per dataset,
the group table plus the two step-31 contrasts (consistency effect =
consistent-wrong − inconsistent; correctness effect = consistent-correct −
consistent-wrong) with bootstrap CIs.

No new inference: the unhinted arm is shared across hint types, so
consistency is read once per dataset and joined to all 3 hint slices,
matching how steps 27/28/31 pool.

Usage:
    python scripts/33_consistency_vs_correctness_by_dataset.py            # base
    python scripts/33_consistency_vs_correctness_by_dataset.py --model <hf-id>

    Options:
      --config     path to config YAML (default: config/mvp_config.yaml)
      --data       split: test (default) | eval
      --model      model/adapter HF id (default: Qwen/Qwen3-4B)
      --min-n      min questions per (dataset, group) to draw the dot
                   (default: 8)
      --resamples  bootstrap replicates per CI (default: 5000)
      --seed       bootstrap seed (default: 88)
      --out        output PNG path (default:
                   {plots_dir}/task-performance-vs-monitorability/
                   consistency_vs_correctness_by_dataset_{model}.png)

Input:
    {outputs_dir}/{model}/{dataset}/{hint}/unhinted.jsonl[.gz]
    {analysis_dir}/{model}/{dataset}/{hint}/results_per_instance.csv

Output:
    {plots_dir}/task-performance-vs-monitorability/
        consistency_vs_correctness_by_dataset_{model}.png
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
from matplotlib.lines import Line2D
import numpy as np
import yaml

from src.dataset import SPLITS, model_dir_name, resolve_paths
from src.hints import HINT_TYPES

DEFAULT_MODEL = "Qwen/Qwen3-4B"

DATASETS = ["mmlu_pro", "gpqa_diamond", "lsat_ar", "lsat_lr",
            "medxpertqa", "lexam", "codemmlu", "wmdp"]

# same three groups / colours / vertical order as step 31
GROUPS_3 = [
    ("inconsistent", "Inconsistent", "#9AA7B4", 3),
    ("cons_wrong", "Consistent but WRONG", "#C1462F", 2),
    ("cons_right", "Consistent and CORRECT", "#1B3A6B", 1),
]

# two-group merge: the well-powered per-dataset split (step-31 consistency
# effect). "consistent" = consistent-wrong + consistent-correct.
GROUPS_2 = [
    ("inconsistent", "Inconsistent (answers vary)", "#9AA7B4", 2),
    ("consistent", "Consistent (same answer 8/8)", "#1B3A6B", 1),
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
    plain, gz = base / f"{stem}.jsonl", base / f"{stem}.jsonl.gz"
    if plain.exists():
        return plain.open()
    if gz.exists():
        return gzip.open(gz, "rt")
    return None


def load_dataset(root, config, model_name, dataset, split):
    """Per-question (consistency, accuracy, gmean2) for one dataset."""
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
                pa = r.get("parsed_answer")
                if pa in (None, "", "N/A"):
                    continue
                answers[r["instance_id"]].append(pa)
        break
    if not answers:
        return None
    consistency = {iid: Counter(a).most_common(1)[0][1] / len(a)
                   for iid, a in answers.items()}

    recs = []
    for hint in HINT_TYPES:
        paths = resolve_paths(root, config, dataset, model_name, hint,
                              split=split)
        csv_path = paths["analysis_dir"] / "results_per_instance.csv"
        if not csv_path.exists():
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
    return recs


def main():
    parser = argparse.ArgumentParser(
        description="Per-dataset consistency-vs-correctness small multiples.")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--data", default="test", choices=SPLITS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--merge-consistent", action="store_true",
                        help="two dots per panel — inconsistent vs consistent "
                             "(the well-powered per-dataset split) — instead "
                             "of the three-group breakdown; writes "
                             "…_2group_… .png")
    parser.add_argument("--min-n", type=int, default=8)
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

    GROUPS = GROUPS_2 if args.merge_consistent else GROUPS_3

    def group_masks(determ, correct, acc):
        if args.merge_consistent:
            return {"inconsistent": ~determ, "consistent": determ}
        return {"inconsistent": ~determ,
                "cons_wrong": determ & (acc <= 0.001),
                "cons_right": determ & correct}

    # contrasts to print (hi_key − lo_key), per mode
    if args.merge_consistent:
        CONTRASTS = [("consistency effect (consistent − inconsistent)",
                      "consistent", "inconsistent")]
    else:
        CONTRASTS = [
            ("consistency effect (cons-wrong − inconsistent)",
             "cons_wrong", "inconsistent"),
            ("correctness effect (cons-correct − cons-wrong)",
             "cons_right", "cons_wrong"),
        ]

    # ---- gather per-dataset group stats ----
    panels = {}   # dataset -> {group_key: (mean, lo, hi, n) or None}
    all_bounds = []
    for dataset in DATASETS:
        recs = load_dataset(root, config, model_name, dataset, args.data)
        if not recs:
            print(f"{dataset}: no data — skipped")
            panels[dataset] = None
            continue
        cons = np.array([r[0] for r in recs])
        acc = np.array([r[1] for r in recs])
        gm = np.array([r[2] for r in recs])
        determ = cons >= 0.999
        correct = acc >= 0.999
        masks = group_masks(determ, correct, acc)
        stats_g = {}
        print(f"\n{dataset} — {len(recs)} entering questions")
        print(f"{'group':>24} | {'n':>4} | {'g-mean2':>8} | 95% CI")
        print("-" * 56)
        for key, label, _, _ in GROUPS:
            v = gm[masks[key]]
            n = len(v)
            if n < args.min_n:
                print(f"{label:>24} | {n:>4} | (dropped: n < {args.min_n})")
                stats_g[key] = None
                continue
            mean = float(v.mean())
            lo, hi = boot_ci(v, args.resamples, rng)
            print(f"{label:>24} | {n:>4} | {mean:>8.3f} | [{lo:.3f}, {hi:.3f}]")
            stats_g[key] = (mean, lo, hi, n, v)
            all_bounds.extend([lo, hi])
        # contrasts (when both groups survive)
        for eff, hi_k, lo_k in CONTRASTS:
            a, b = stats_g.get(hi_k), stats_g.get(lo_k)
            if a and b:
                d = a[0] - b[0]
                lo, hi = boot_diff_ci(a[4], b[4], args.resamples, rng)
                print(f"    {eff}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        panels[dataset] = {k: (v[:4] if v else None)
                           for k, v in stats_g.items()}

    if not all_bounds:
        print("\nERROR: no dataset had enough data — nothing to plot.")
        sys.exit(1)

    # ---- figure: 2x4 small multiples, shared x ----
    xlo, xhi = min(all_bounds) - 0.03, max(all_bounds) + 0.03
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.4), sharex=True,
                             sharey=True)
    color_of = {k: c for k, _, c, _ in GROUPS}
    y_of = {k: y for k, _, _, y in GROUPS}

    for ax, dataset in zip(axes.flat, DATASETS):
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color="#EEEEEE", linewidth=0.8)
        ax.set_title(dataset, fontsize=10, fontweight="bold")
        p = panels.get(dataset)
        if not p:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", color="#999999", fontsize=9)
        else:
            for key, _, color, y in GROUPS:
                cell = p.get(key)
                if cell is None:
                    continue
                mean, lo, hi, n = cell
                ax.errorbar(mean, y, xerr=[[mean - lo], [hi - mean]],
                            fmt="o", color=color, ecolor=color,
                            markersize=8, markeredgecolor="#333333",
                            markeredgewidth=0.5, elinewidth=1.8, capsize=3,
                            capthick=1.8, zorder=3)
                ax.text(mean, y + 0.28, f"{mean:.3f}", ha="center",
                        va="bottom", fontsize=8.5, color=color,
                        fontweight="bold")
                ax.text(mean, y - 0.30, f"n={n}", ha="center", va="top",
                        fontsize=7.5, color="#888888")
        ax.set_yticks([g[3] for g in GROUPS])
        ax.set_yticklabels([])
        ax.set_ylim(0.4, max(g[3] for g in GROUPS) + 0.7)
        ax.set_xlim(xlo, xhi)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#999999")

    for ax in axes[-1]:
        ax.set_xlabel("Mean g-mean²", fontsize=9)

    handles = [Line2D([], [], marker="o", linestyle="none", color=c,
                      markeredgecolor="#333333", markeredgewidth=0.5,
                      markersize=9, label=lab)
               for _, lab, c, _ in GROUPS]
    fig.legend(handles=handles, loc="lower center", ncol=len(GROUPS),
               frameon=True, edgecolor="#CCCCCC", fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    if args.merge_consistent:
        title = ("Consistent answers are more monitorable in every dataset "
                 f"— {model_name}\n(entering questions, hint types pooled; "
                 "consistent = same answer in all 8 unhinted runs; "
                 "bootstrap 95% CI)")
    else:
        title = ("Consistency vs correctness as drivers of monitorability, "
                 f"per dataset — {model_name}\n(entering questions, hint "
                 "types pooled; bootstrap 95% CI; groups with n < "
                 f"{args.min_n} omitted)")
    fig.suptitle(title, fontsize=12)

    out = args.out
    if out is None:
        # RO1 bespoke analysis plots live under extra-plots/, not the main plots/ tree
        plots_dir = ((root / config["paths"]["plots_dir"]).parent
                     / "extra-plots" / "task-performance-vs-monitorability")
        if args.data != "test":
            plots_dir = plots_dir / args.data
        plots_dir.mkdir(parents=True, exist_ok=True)
        tag = "_2group" if args.merge_consistent else ""
        out = (plots_dir /
               f"consistency_vs_correctness_by_dataset{tag}_{model_name}.png")
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
