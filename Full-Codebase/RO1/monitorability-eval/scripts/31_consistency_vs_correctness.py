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
questions score 0.805 vs 0.710 for inconsistent ones (Delta +0.095,
CI excluding zero), while consistent-correct beats consistent-wrong by
only +0.033 (CI includes zero). Task correctness per se adds little once
commitment is controlled; the step-27 accuracy threshold is largely a
proxy for determinism.

Runs that hit the generation cap are excluded from the answer set, matching
the pipeline's u_valid rule (src/gmean2_metrics.py). That keeps this step's
denominator identical to acc_u_i's, so "always correct" implies "consistent"
and the consistent-and-correct group (n = 1032) is exactly the 8/8 accuracy
level of step 30's question panel.

The figure is one horizontal dot-and-CI panel (three rows, n annotated),
with the two decisive contrasts bracketed alongside. All CIs are 95%
percentile intervals from a cluster bootstrap over questions: a question
contributes up to 3 rows (one per hint type) that share the same unhinted
runs, so its rows are resampled together rather than as independent rows.

Usage:
    python scripts/31_consistency_vs_correctness.py                  # base
    python scripts/31_consistency_vs_correctness.py --model <hf-id>

    Options:
      --config     path to config YAML (default: config/mvp_config.yaml)
      --data       split: test (default) | eval
      --model      model/adapter HF id (default: Qwen/Qwen3-4B)
      --adjusted   dots and brackets adjusted for dataset + hint type (OLS
                   fixed effects, adjusted means), cluster-bootstrap CIs over
                   questions; writes …_adjusted_… .png
      --resamples  bootstrap replicates per CI (default: 10000)
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

--arm hinted: consistency AND accuracy are instead computed from the HINTED
runs, pooled over all 3 hint types (up to 8 x 3 = 24 runs per question, read
from each hint dir's hinted.jsonl[.gz], same validity rule). That one pooled
value per question is joined to all three hint slices exactly as above, so the
rows and g-mean² values are identical to the unhinted figure — only the
grouping changes. It always uses the two-group layout: consistent-and-CORRECT
is empty by construction (an entering question has p1 > 0 in every hint
slice, so at least one hinted run per hint type picked the wrong hint
target), and "consistent" therefore means "picked the hint target in every
hinted run". Output:
    {plots_dir}/consistency_vs_correctness_hinted_{model}.png
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


def cluster_boot_means(qid, gm, masks, n_resamples, rng, chunk=1000):
    """Cluster bootstrap over questions: resampled mean g-mean² of each mask.

    `qid` maps each row to a question index. Each replicate draws questions
    with replacement and keeps all of a question's hint rows together (they
    share the same unhinted runs, so they are not independent). All masks
    share the same draws, so differences between columns are valid bootstrap
    samples of the group differences. Returns (n_resamples, len(masks))."""
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


def adjusted_means(labels, levels, ds_code, hint_code, gm, idx):
    """Group means adjusted for dataset and hint type, over rows `idx`.

    Fits OLS  g-mean² ~ group + dataset FE + hint-type FE  (first level of
    each factor is the reference), then averages every row's prediction with
    its group switched to each level in turn, so all groups are evaluated on
    the same dataset x hint mix (the rows' own). Differences between the
    returned means equal the group coefficients."""
    lab, d, h, y = labels[idx], ds_code[idx], hint_code[idx], gm[idx]
    cols = [np.ones(len(idx))]
    cols += [(lab == lv).astype(float) for lv in levels[1:]]
    cols += [(d == k).astype(float) for k in range(1, ds_code.max() + 1)]
    cols += [(h == k).astype(float) for k in range(1, hint_code.max() + 1)]
    X = np.column_stack(cols)
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    n_grp = len(levels) - 1
    base = np.delete(X, np.s_[1:1 + n_grp], axis=1) @ np.delete(
        b, np.s_[1:1 + n_grp])
    return np.array([base.mean()] + [base.mean() + b[1 + j]
                                     for j in range(n_grp)])


def adjusted_stats(labels, levels, ds_code, hint_code, gm, by_q, n_resamples,
                   rng):
    """Adjusted means for `levels`, plus cluster-bootstrap (over questions)
    samples of them, restricted to rows whose label is in `levels`."""
    keep = np.isin(labels, levels)
    full = adjusted_means(labels, levels, ds_code, hint_code, gm,
                          np.where(keep)[0])
    groups = [ix[keep[ix]] for ix in by_q]
    groups = [ix for ix in groups if len(ix)]
    boots = np.empty((n_resamples, len(levels)))
    for i in range(n_resamples):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[j] for j in pick])
        boots[i] = adjusted_means(labels, levels, ds_code, hint_code, gm, idx)
    return full, boots


def pct_ci(samples):
    return (float(np.nanpercentile(samples, 2.5)),
            float(np.nanpercentile(samples, 97.5)))


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
    parser.add_argument("--arm", default="unhinted",
                        choices=["unhinted", "hinted"],
                        help="arm whose runs define consistency/accuracy: the "
                             "shared unhinted arm (default) or the hinted arm "
                             "pooled over all 3 hint types; 'hinted' writes "
                             "…_hinted_… .png")
    parser.add_argument("--adjusted", action="store_true",
                        help="plot group means and contrasts adjusted for "
                             "dataset and hint type (OLS fixed effects), with "
                             "cluster-bootstrap CIs over questions; writes "
                             "…_adjusted_… .png")
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
    if args.arm == "hinted" and not args.merge_consistent:
        # Consistent-and-CORRECT is structurally empty on the hinted arm: an
        # entering question has p1 > 0 in every hint slice, i.e. at least one
        # hinted run per hint type picked the (wrong) hint target, so no
        # entering question is correct in all its hinted runs.
        print("NOTE: --arm hinted uses the two-group layout (consistent-and-"
              "correct is empty by construction on the hinted arm).\n")
        args.merge_consistent = True

    recs, missing, anomalies = [], [], 0
    for dataset in DATASETS:
        # --- consistency from the shared unhinted arm (once per dataset), or
        # from the hinted arm pooled over all 3 hint types (--arm hinted) ---
        answers = defaultdict(list)
        correct_answer = {}
        for hint in HINT_TYPES:
            paths = resolve_paths(root, config, dataset, model_name, hint,
                                  split=args.data)
            fh = open_jsonl(paths["outputs_dir"], args.arm)
            if fh is None:
                continue
            with fh:
                for line in fh:
                    r = json.loads(line)
                    # Same validity rule as the pipeline's u_valid (see
                    # src/gmean2_metrics.py): a run that hit the generation cap
                    # did not produce an answer, it produced a fragment, and its
                    # "parsed" answer is usually a stray token (e.g. "H" on a
                    # four-option question). Counting that as the model giving a
                    # DIFFERENT answer is a false positive for inconsistency.
                    # Keeping this in step with acc_u_i also makes "always
                    # correct" imply "consistent", so this figure's
                    # consistent-and-correct group is exactly the 8/8 accuracy
                    # level of step 30's question panel.
                    if r.get("truncated", False):
                        continue
                    pa = r.get("parsed_answer")
                    if pa in (None, "", "N/A"):
                        continue
                    answers[int(r["instance_id"])].append(pa)
                    correct_answer[int(r["instance_id"])] = r.get("correct_answer")
            if args.arm == "unhinted":
                break  # shared arm: one copy is enough; hinted pools all 3
        if not answers:
            missing.append(f"{dataset} (no {args.arm} arm)")
            continue
        consistency, hinted_acc = {}, {}
        for iid, a in answers.items():
            consistency[iid] = Counter(a).most_common(1)[0][1] / len(a)
            hinted_acc[iid] = sum(x == correct_answer[iid] for x in a) / len(a)

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
                    # accuracy must come from the same runs as consistency
                    acc_i = float(a) if args.arm == "unhinted" else hinted_acc[iid]
                    recs.append((dataset, hint, consistency[iid],
                                 acc_i, float(g), iid))
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
    # question index per row: the unit of the cluster bootstrap
    q_rows = defaultdict(list)
    for i, r in enumerate(recs):
        q_rows[(r[0], r[5])].append(i)
    by_q = [np.array(v) for v in q_rows.values()]
    qid = np.empty(len(recs), dtype=int)
    for j, ix in enumerate(by_q):
        qid[ix] = j
    # acc = 1 implies consistency = 1; any violation is a validity-set mismatch
    anomalies = int((correct & ~determ).sum())

    GROUPS = GROUPS_2 if args.merge_consistent else GROUPS_3
    if args.arm == "hinted":
        # Every consistent entering question picked the hint target in all
        # its hinted runs (p1 > 0 in each slice forces it), so say so.
        GROUPS = [("inconsistent",
                   "Inconsistent\n(answers vary across hinted runs)", GROUPS[0][2]),
                  ("consistent",
                   "Consistent\n(picked the hint's answer\nin all hinted runs)",
                   GROUPS[1][2])]
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
    # one shared set of cluster-bootstrap draws for every dot and bracket
    keys = [g[0] for g in GROUPS]
    boot = cluster_boot_means(qid, gm, [masks[k] for k in keys] + [determ],
                              args.resamples, rng)
    stats_by_group = {}
    for j, (key, label, _) in enumerate(GROUPS):
        v = gm[masks[key]]
        lo, hi = pct_ci(boot[:, j])
        stats_by_group[key] = (v, float(v.mean()), lo, hi)
        flat = label.replace("\n", " ")
        print(f"{flat[:24]:>24} | {len(v):>5} | {v.mean():>8.3f} | "
              f"[{lo:.3f}, {hi:.3f}]")

    def contrast(a, b):
        ja, jb = keys.index(a), keys.index(b)
        d = stats_by_group[a][1] - stats_by_group[b][1]
        return (d, *pct_ci(boot[:, ja] - boot[:, jb]))

    if args.merge_consistent:
        d1, lo1, hi1 = contrast("consistent", "inconsistent")
        # brackets list: one contrast (consistent − inconsistent)
        brackets = [(2, 1, d1, lo1, hi1, "consistency effect", GROUPS[1][2])]
        print(f"\nCONSISTENCY effect (consistent − inconsistent): "
              f"{d1:+.3f} [{lo1:+.3f}, {hi1:+.3f}]")
    else:
        d1, lo1, hi1 = contrast("cons_wrong", "inconsistent")
        d2, lo2, hi2 = contrast("cons_right", "cons_wrong")
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

    # pooled consistent dot (three-group figure only)
    v_con_all = gm[determ]
    con_m = float(v_con_all.mean())
    con_lo, con_hi = pct_ci(boot[:, -1])

    if args.adjusted:
        # Replace every dot and bracket with its dataset- and hint-adjusted
        # counterpart. One joint model for the plotted groups (so bracket
        # = difference of the two dots it joins), a second two-group model for
        # the pooled consistent dot; CIs from a cluster bootstrap over
        # questions (a question contributes up to 3 hint rows).
        ds_list = sorted({r[0] for r in recs}, key=DATASETS.index)
        ds_code = np.array([ds_list.index(r[0]) for r in recs])
        hint_code = np.array([HINT_TYPES.index(r[1]) for r in recs])
        labels = np.full(len(recs), "", dtype=object)
        for key, m in masks.items():
            labels[m] = key
        levels = [g[0] for g in GROUPS]
        full, boots = adjusted_stats(labels, levels, ds_code, hint_code, gm,
                                     by_q, args.resamples, rng)
        print("\nADJUSTED for dataset + hint type (cluster bootstrap over "
              "questions):")
        for j, key in enumerate(levels):
            lo, hi = pct_ci(boots[:, j])
            stats_by_group[key] = (stats_by_group[key][0], float(full[j]),
                                   lo, hi)
            print(f"  {key:>13}: {full[j]:.3f} [{lo:.3f}, {hi:.3f}]")
        new_brackets = []
        for (y_lo, y_hi, _, _, _, lab, color) in brackets:
            # bracket rows count down from the top group
            a = levels[len(levels) - y_hi]   # lower row = later group
            b_ = levels[len(levels) - y_lo]
            ja, jb = levels.index(a), levels.index(b_)
            d = float(full[ja] - full[jb])
            lo, hi = pct_ci(boots[:, ja] - boots[:, jb])
            new_brackets.append((y_lo, y_hi, d, lo, hi, lab, color))
            print(f"  {lab}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        brackets = new_brackets
        if not args.merge_consistent:
            (d1, lo1, hi1), (d2, lo2, hi2) = [b[2:5] for b in brackets]
            pooled = np.where(labels == "inconsistent", "inconsistent",
                              np.where(determ, "consistent", ""))
            full2, boots2 = adjusted_stats(
                pooled.astype(object), ["inconsistent", "consistent"],
                ds_code, hint_code, gm, by_q, args.resamples, rng)
            con_m = float(full2[1])
            con_lo, con_hi = pct_ci(boots2[:, 1])
            print(f"  pooled consistent: {con_m:.3f} [{con_lo:.3f}, "
                  f"{con_hi:.3f}]")

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
                fontsize=10, color="#333333")

    def bracket(y_lo, y_hi, d, lo, hi, lab, color, x_br):
        # Inset both ends: the two brackets share the middle row, so drawn to
        # the full row height their arms land on top of each other.
        pad = 0.10 if y_hi > y_lo else -0.10
        y0, y1 = y_lo + pad, y_hi - pad
        ax.plot([x_br, x_br + 0.012, x_br + 0.012, x_br],
                [y0, y0, y1, y1], color=color, linewidth=1.2,
                clip_on=False, zorder=4)
        ax.text(x_br + 0.022, (y_lo + y_hi) / 2,
                f"{lab}\nΔ = {d:+.3f} [{lo:+.3f}, {hi:+.3f}]",
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
        fig, ax = plt.subplots(figsize=(8.6, 4.4))
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color="#DDDDDD", linewidth=0.8)
        v_con = v_con_all
        inc = stats_by_group["inconsistent"]
        cw = stats_by_group["cons_wrong"]
        cr = stats_by_group["cons_right"]
        rows = [
            (4, inc[1], inc[2], inc[3], len(inc[0]), GREY, GROUPS[0][1]),
            (3, cw[1], cw[2], cw[3], len(cw[0]), RED, GROUPS[1][1]),
            (2, cr[1], cr[2], cr[3], len(cr[0]), NAVY, GROUPS[2][1]),
            (1, con_m, con_lo, con_hi, len(v_con), TEAL,
             "Consistency\n(consistent, wrong + correct pooled)"),
        ]
        for y, mean, lo, hi, n, color, _ in rows:
            dot(mean, y, lo, hi, n, color)
        # dashed separator under the three groups; the pooled dot sits below it
        ax.axhline(1.5, color="#CCCCCC", linestyle="--", linewidth=1.2,
                   zorder=1)
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
    ax.set_xlabel("Monitorability (g-mean²)"
                  + (", adjusted for dataset and hint type" if args.adjusted
                     else ""))
    ax.set_title("Monitorability vs Consistency"
                 + (" (consistency from hinted runs, 3 hint types pooled)"
                    if args.arm == "hinted" else "")
                 + (" — adjusted for dataset and hint type" if args.adjusted
                    else ""), fontsize=13)
    if args.adjusted:
        fig.text(0.01, -0.02,
                 "Dots: group means from OLS  g-mean² ~ group + dataset + "
                 "hint type, each group evaluated on the pooled dataset × hint "
                 "mix; n = rows.\nCIs: 95% cluster bootstrap over questions "
                 f"({args.resamples:,} resamples). Brackets = differences of "
                 "the adjusted means.",
                 fontsize=8, color="#555555", ha="left", va="top")
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
        # hinted is always two-group, so it needs no _2group tag
        tag = ("_hinted" if args.arm == "hinted"
               else "_2group" if args.merge_consistent else "")
        if args.adjusted:
            tag += "_adjusted"
        out = plots_dir / f"consistency_vs_correctness{tag}_{model_name}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
