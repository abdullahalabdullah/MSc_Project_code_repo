#!/usr/bin/env python3
"""
Step 33: Per-dataset version of step 31 — is monitorability driven by being
RIGHT or by being CONSISTENT, shown separately for each dataset.

Step 31 pools all 8 datasets into one dot plot: three contrast groups
(inconsistent / consistent-but-WRONG / consistent-and-CORRECT) above a dashed
divider, and the pooled Consistency dot (wrong + correct together) below it.
This step draws the SAME four rows as an 8-panel small-multiple (2x4,
domain-table order, shared x-axis), so the reader can see whether the
"commitment, not correctness" pattern holds dataset by dataset or is carried
by a few.

The pooled row is not a fourth contrast group — it is rows 2 and 3 combined,
so it deliberately double-counts them. It is drawn below the divider for the
same reason as in step 31: it answers "where does the committed group as a
whole land?", which is a different question from the two contrasts above it.

Grouping (identical to step 31, per dataset):
    consistency_i = modal unhinted answer share over the valid unhinted runs
    inconsistent           consistency < 1
    consistent but WRONG   consistency = 1, accuracy = 0
    consistent and CORRECT consistency = 1, accuracy = 1

Per (dataset, group): mean gmean2_i with a 95% percentile CI from a cluster
bootstrap over questions (a question contributes up to 3 rows, one per hint
type, sharing the same unhinted runs, so its rows are resampled together;
all groups and contrasts in a dataset share the same draws). Each dot is
annotated "n=<rows> (<questions>q)". Groups with fewer than --min-n distinct
QUESTIONS are dropped from that panel (the consistent-WRONG group is the thin
one — pooled n=285 rows / 146 questions, so per dataset it is often small;
the guard keeps noisy dots out). Stdout prints, per dataset, the group table
plus the two step-31 contrasts (consistency effect = consistent-wrong −
inconsistent; correctness effect = consistent-correct − consistent-wrong)
with cluster-bootstrap CIs. Group means are raw (not adjusted for the
within-dataset hint-type mix), so the figure is descriptive.

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
      --by         dataset (default, 2x4 panels) | hint (1x3 panels, all
                   datasets pooled; --adjusted then adjusts for the dataset
                   mix); hint writes consistency_vs_correctness_by_hint… .png
      --min-n      min distinct questions per (dataset, group) to draw the
                   dot (default: 8)
      --adjusted   dots and contrasts adjusted for the hint-type mix within
                   each dataset (OLS hint fixed effects, adjusted means);
                   writes …_adjusted_… .png
      --resamples  cluster-bootstrap replicates per CI (default: 10000)
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

# Type sizes set for the PRINTED size (canvas ~11in -> ~6.3in of
# \textwidth, so ~0.57 scaling in the PDF).
FS_PANEL, FS_LABEL, FS_TICK, FS_VAL, FS_N, FS_LEG = 12, 12, 10, 9, 8.5, 10

DEFAULT_MODEL = "Qwen/Qwen3-4B"

DATASETS = ["mmlu_pro", "gpqa_diamond", "lsat_ar", "lsat_lr",
            "medxpertqa", "lexam", "codemmlu", "wmdp"]

# same groups / colours / vertical order as step 31: the three contrast groups
# above a dashed divider, and below it the pooled Consistency dot (consistent-
# WRONG + consistent-CORRECT together), which is where the whole committed
# group lands.
GROUPS_3 = [
    ("inconsistent", "Inconsistent", "#9AA7B4", 4),
    ("cons_wrong", "Consistent but WRONG", "#C1462F", 3),
    ("cons_right", "Consistent and CORRECT", "#1B3A6B", 2),
    ("consistent", "Consistent (wrong + correct pooled)", "#2A9D8F", 1),
]
HINT_LABELS = {"professor": "Professor", "grader_hacking": "Grader-hacking",
               "unethical": "Unethical"}
POOLED_KEY = "consistent"      # drawn below the divider in 3-group mode
DIVIDER_Y = 1.5

# two-group merge: the well-powered per-dataset split (step-31 consistency
# effect). "consistent" = consistent-wrong + consistent-correct.
GROUPS_2 = [
    ("inconsistent", "Inconsistent (answers vary)", "#9AA7B4", 2),
    ("consistent", "Consistent (same answer in all valid runs)",
     "#1B3A6B", 1),
]


def cluster_boot_means(qid, gm, masks, n_resamples, rng, chunk=1000):
    """Cluster bootstrap over questions (same scheme as step 31): resampled
    mean g-mean² of each mask, all masks sharing the same draws, so column
    differences are valid bootstrap samples of group differences. A draw in
    which a group has no rows gives NaN for that group (ignored by
    nanpercentile). Returns (n_resamples, len(masks))."""
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


def hint_adjusted_means(labels, levels, hint_code, gm, idx):
    """Group means adjusted for hint type, over rows `idx` of one dataset.

    OLS  g-mean² ~ group + hint-type FE  (first level of each factor is the
    reference); each group is then evaluated on the rows' own hint mix
    (adjusted means), so differences between the returned means equal the
    group coefficients. A level with no rows in `idx` returns NaN."""
    lab, h, y = labels[idx], hint_code[idx], gm[idx]
    present = [lv for lv in levels if (lab == lv).any()]
    if not present:
        return np.full(len(levels), np.nan)
    ref = present[0]
    others = present[1:]
    cols = [np.ones(len(idx))] + [(lab == lv).astype(float) for lv in others]
    cols += [(h == k).astype(float) for k in range(1, hint_code.max() + 1)]
    X = np.column_stack(cols)
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    k = len(others)
    base = (np.delete(X, np.s_[1:1 + k], axis=1)
            @ np.delete(b, np.s_[1:1 + k])).mean()
    eff = {ref: 0.0, **{lv: b[1 + j] for j, lv in enumerate(others)}}
    return np.array([base + eff[lv] if lv in eff else np.nan
                     for lv in levels])


def adjusted_boot(qid, labels, levels, hint_code, gm, n_resamples, rng):
    """Hint-adjusted means for `levels` (rows whose label is in `levels`),
    plus cluster-bootstrap samples over questions. Returns (full, boots)."""
    keep = np.isin(labels, levels)
    rows = np.where(keep)[0]
    full = hint_adjusted_means(labels, levels, hint_code, gm, rows)
    by_q = defaultdict(list)
    for i in rows:
        by_q[qid[i]].append(i)
    groups = [np.array(v) for v in by_q.values()]
    boots = np.empty((n_resamples, len(levels)))
    for r in range(n_resamples):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[j] for j in pick])
        boots[r] = hint_adjusted_means(labels, levels, hint_code, gm, idx)
    return full, boots


def pct_ci(samples):
    return (float(np.nanpercentile(samples, 2.5)),
            float(np.nanpercentile(samples, 97.5)))


def open_jsonl(base, stem):
    plain, gz = base / f"{stem}.jsonl", base / f"{stem}.jsonl.gz"
    if plain.exists():
        return plain.open()
    if gz.exists():
        return gzip.open(gz, "rt")
    return None


def load_dataset(root, config, model_name, dataset, split):
    """Per (question, hint) row: (consistency, accuracy, gmean2, iid, hint)."""
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
                recs.append((consistency[iid], float(a), float(g), iid,
                             hint))
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
    parser.add_argument("--adjusted", action="store_true",
                        help="dots and contrasts adjusted for the hint-type mix "
                             "within each dataset (OLS hint fixed effects, "
                             "adjusted means); writes …_adjusted_… .png")
    parser.add_argument("--by", default="dataset",
                        choices=["dataset", "hint"],
                        help="one panel per dataset (default, 2x4) or per hint "
                             "type (1x3, all datasets pooled; with --adjusted "
                             "the adjustment is for the dataset mix); 'hint' "
                             "writes consistency_vs_correctness_by_hint… .png")
    parser.add_argument("--min-n", type=int, default=8)
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

    GROUPS = GROUPS_2 if args.merge_consistent else GROUPS_3

    def group_masks(determ, correct, acc):
        if args.merge_consistent:
            return {"inconsistent": ~determ, "consistent": determ}
        return {"inconsistent": ~determ,
                "cons_wrong": determ & (acc <= 0.001),
                "cons_right": determ & correct,
                # pooled row: every consistent question, wrong and correct
                # together. Not a contrast group — it is the two above it
                # combined, so it deliberately double-counts them.
                "consistent": determ}

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

    # ---- load every dataset once; rows carry their dataset name ----
    all_recs = []
    for dataset in DATASETS:
        recs = load_dataset(root, config, model_name, dataset, args.data)
        if not recs:
            print(f"{dataset}: no data — skipped")
            continue
        all_recs += [(dataset,) + r for r in recs]
    # panel axis and the factor adjusted for within a panel
    if args.by == "dataset":
        PANELS = DATASETS
        panel_of = lambda r: r[0]
        fe_of = lambda r: HINT_TYPES.index(r[5])
        fe_name = "hint type"
        panel_title = lambda k: k
    else:
        PANELS = list(HINT_TYPES)
        panel_of = lambda r: r[5]
        fe_of = lambda r: DATASETS.index(r[0])
        fe_name = "dataset"
        panel_title = lambda k: HINT_LABELS.get(k, k)

    # ---- gather per-panel group stats ----
    panels = {}   # panel -> {group_key: (mean, lo, hi, n, nq) or None}
    all_bounds = []
    for dataset in PANELS:   # `dataset` = panel key (a dataset or a hint)
        prow = [r for r in all_recs if panel_of(r) == dataset]
        if not prow:
            print(f"{dataset}: no data — skipped")
            panels[dataset] = None
            continue
        recs = [r[1:] for r in prow]
        cons = np.array([r[0] for r in recs])
        acc = np.array([r[1] for r in recs])
        gm = np.array([r[2] for r in recs])
        determ = cons >= 0.999
        correct = acc >= 0.999
        masks = group_masks(determ, correct, acc)
        # question = (dataset, instance_id): the cluster-bootstrap unit
        iids = [(r[0], r[4]) for r in prow]
        uniq = {iid: j for j, iid in enumerate(dict.fromkeys(iids))}
        qid = np.array([uniq[i] for i in iids])
        keys = [g[0] for g in GROUPS]
        if args.adjusted:
            # one joint model over the contrast groups (so contrasts =
            # differences of the dots), a second two-group model for the
            # pooled consistent row (3-group mode only)
            hint_code = np.array([fe_of(r) for r in prow])
            contrast_keys = [k for k in keys
                             if args.merge_consistent or k != POOLED_KEY]
            labels = np.full(len(recs), "", dtype=object)
            for k in contrast_keys:
                labels[masks[k]] = k
            full, b1 = adjusted_boot(qid, labels, contrast_keys, hint_code,
                                     gm, args.resamples, rng)
            means_of = dict(zip(contrast_keys, full))
            boot_of = {k: b1[:, j] for j, k in enumerate(contrast_keys)}
            if not args.merge_consistent:
                pooled = np.where(determ, "consistent",
                                  "inconsistent").astype(object)
                full2, b2 = adjusted_boot(
                    qid, pooled, ["inconsistent", "consistent"], hint_code,
                    gm, args.resamples, rng)
                means_of[POOLED_KEY] = full2[1]
                boot_of[POOLED_KEY] = b2[:, 1]
            boot = np.column_stack([boot_of[k] for k in keys])
        else:
            boot = cluster_boot_means(qid, gm, [masks[k] for k in keys],
                                      args.resamples, rng)
            means_of = {k: float(gm[masks[k]].mean()) if masks[k].any()
                        else float("nan") for k in keys}
        stats_g = {}
        print(f"\n{dataset} — {len(recs)} entering rows, "
              f"{len(uniq)} questions")
        print(f"{'group':>24} | {'rows':>4} | {'q':>4} | {'g-mean2':>8} | 95% CI")
        print("-" * 64)
        for j, (key, label, _, _) in enumerate(GROUPS):
            m = masks[key]
            v = gm[m]
            n, nq = int(m.sum()), len(set(qid[m]))
            if nq < args.min_n:
                print(f"{label:>24} | {n:>4} | {nq:>4} | "
                      f"(dropped: < {args.min_n} questions)")
                stats_g[key] = None
                continue
            mean = float(means_of[key])
            lo, hi = pct_ci(boot[:, j])
            print(f"{label:>24} | {n:>4} | {nq:>4} | {mean:>8.3f} | "
                  f"[{lo:.3f}, {hi:.3f}]")
            stats_g[key] = (mean, lo, hi, n, nq, j)
            all_bounds.extend([lo, hi])
        # contrasts (when both groups survive)
        for eff, hi_k, lo_k in CONTRASTS:
            a, b = stats_g.get(hi_k), stats_g.get(lo_k)
            if a and b:
                d = a[0] - b[0]
                lo, hi = pct_ci(boot[:, a[5]] - boot[:, b[5]])
                print(f"    {eff}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        panels[dataset] = {k: (v[:5] if v else None)
                           for k, v in stats_g.items()}

    if not all_bounds:
        print("\nERROR: no dataset had enough data — nothing to plot.")
        sys.exit(1)

    # ---- figure: small multiples, shared x (2x4 per dataset, 1x3 per hint) ----
    xlo, xhi = min(all_bounds) - 0.03, max(all_bounds) + 0.03
    # taller in 3-group mode: four rows per panel need the extra room, or the
    # n= labels collide with the divider
    if args.by == "dataset":
        fig_h = 5.0 if args.merge_consistent else 7.4
        fig, axes = plt.subplots(2, 4, figsize=(11, fig_h), sharex=True,
                                 sharey=True)
    else:
        fig_h = 3.2 if args.merge_consistent else 4.6
        fig, axes = plt.subplots(1, 3, figsize=(11, fig_h), sharex=True,
                                 sharey=True)
    color_of = {k: c for k, _, c, _ in GROUPS}
    y_of = {k: y for k, _, _, y in GROUPS}

    for ax, dataset in zip(axes.flat, PANELS):
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color="#EEEEEE", linewidth=0.8)
        ax.set_title(panel_title(dataset), fontsize=FS_PANEL,
                     fontweight="bold")
        ax.tick_params(labelsize=FS_TICK)
        p = panels.get(dataset)
        if not p:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", color="#999999", fontsize=9)
        else:
            for key, _, color, y in GROUPS:
                cell = p.get(key)
                if cell is None:
                    continue
                mean, lo, hi, n, nq = cell
                ax.errorbar(mean, y, xerr=[[mean - lo], [hi - mean]],
                            fmt="o", color=color, ecolor=color,
                            markersize=8, markeredgecolor="#333333",
                            markeredgewidth=0.5, elinewidth=1.8, capsize=3,
                            capthick=1.8, zorder=3)
                ax.text(mean, y + 0.24, f"{mean:.3f}", ha="center",
                        va="bottom", fontsize=FS_VAL, color=color,
                        fontweight="bold")
                ax.text(mean, y - 0.26, f"n={n} ({nq}q)", ha="center",
                        va="top",
                        fontsize=FS_N, color="#333333")
        if not args.merge_consistent:
            # separator: the three contrast groups sit above it, the pooled
            # Consistency dot below — same device as step 31
            ax.axhline(DIVIDER_Y, color="#CCCCCC", linestyle="--",
                       linewidth=1.1, zorder=1)
        ax.set_yticks([g[3] for g in GROUPS])
        ax.set_yticklabels([])
        ax.set_ylim(0.4, max(g[3] for g in GROUPS) + 0.7)
        ax.set_xlim(xlo, xhi)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#999999")

    fig.supxlabel("Mean Monitorability per-question (g-mean²)"
                  + (f", {fe_name}-adjusted" if args.adjusted else ""),
                  fontsize=FS_LABEL, y=0.08 if args.by == "dataset" else 0.10)

    handles = [Line2D([], [], marker="o", linestyle="none", color=c,
                      markeredgecolor="#333333", markeredgewidth=0.5,
                      markersize=9, label=lab)
               for _, lab, c, _ in GROUPS]
    fig.legend(handles=handles, loc="lower center", ncol=len(GROUPS),
               frameon=True, edgecolor="#CCCCCC", fontsize=FS_LEG,
               bbox_to_anchor=(0.5, -0.02))
    title = ("Monitorability vs Consistency, per "
             + ("dataset" if args.by == "dataset" else "hint type"))
    if args.adjusted:
        title += f" — adjusted for {fe_name}"
    fig.suptitle(title, fontsize=13)
    note = (("n = question × hint rows (q = distinct questions). "
             if args.by == "dataset" else
             "n = questions (one row per question within a hint type). ")
            + "95% CIs: "
            f"cluster bootstrap over questions ({args.resamples:,} "
            f"resamples). Groups with < {args.min_n} questions not drawn.")
    if args.adjusted:
        unit = "dataset" if args.by == "dataset" else "hint type"
        note += (f"\nDots: {fe_name}-adjusted means — per {unit}, OLS "
                 f"g-mean² ~ group + {fe_name}, each group evaluated on that "
                 f"{unit}'s {fe_name} mix. Thin groups (< ~20 questions) are "
                 "unstable.")
    fig.text(0.5, -0.07, note, ha="center", va="top", fontsize=9,
             color="#555555")

    out = args.out
    if out is None:
        # RO1 bespoke analysis plots live under extra-plots/, not the main plots/ tree
        plots_dir = ((root / config["paths"]["plots_dir"]).parent
                     / "extra-plots" / "task-performance-vs-monitorability")
        if args.data != "test":
            plots_dir = plots_dir / args.data
        plots_dir.mkdir(parents=True, exist_ok=True)
        tag = "_2group" if args.merge_consistent else ""
        if args.adjusted:
            tag += "_adjusted"
        out = (plots_dir / f"consistency_vs_correctness_by_{args.by}"
                           f"{tag}_{model_name}.png")
    fig.tight_layout(rect=[0, 0.04, 1, 0.94] if args.by == "dataset"
                     else [0, 0.08, 1, 0.92])
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
