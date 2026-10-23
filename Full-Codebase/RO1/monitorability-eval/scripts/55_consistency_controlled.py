#!/usr/bin/env python3
"""
Step 55: Is the consistency effect on monitorability explained by response
length, dataset, hint type or baseline hint-option choice? Controlled
estimates + a within-dataset length-tercile appendix figure, ONE model.

Steps 31/36 compare consistent vs inconsistent questions with at most one
control at a time (step 36: length terciles pooled over datasets; --by-hint:
hint type). Datasets differ in length, consistency AND monitorability, and
hint types enter the two groups unevenly, so this step adds:

1. A ladder of OLS regressions on the question-slice rows

       gmean2_i ~ consistent + controls

   (consistent = same answer in all valid unhinted runs, as in step 31):
       (1) no controls
       (2) + dataset + hint type (fixed effects)
       (3) + log(mean unhinted CoT tokens)
       (4) + p0 > 0  (the model picks the hint target in >=1 UNHINTED run)
   95% CIs from a cluster bootstrap over QUESTIONS (each question contributes
   up to 3 rows, one per hint type, so rows are not independent). Written as
   a markdown table.

   Why (4): p0 > 0 occurs only among inconsistent questions (45% of them) and
   those score lower. TPR_mc already divides by R_effect = TE/p1 to correct
   for baseline hint-option choice (src/gmean2_metrics.py), but that
   correction is estimated from 8 runs and capped at 1, so (4) bounds how
   much of the effect could be tied to it.

2. Appendix figure: consistent vs inconsistent mean gmean2_i within length
   terciles computed SEPARATELY PER DATASET (8 panels, 2x4, shared y), so no
   comparison mixes datasets. Groups with fewer than --min-n rows on either
   side are drawn hollow and get no Δ. Row-level bootstrap CIs per dot.

Usage:
    python scripts/55_consistency_controlled.py                  # base
    python scripts/55_consistency_controlled.py --model <hf-id>

    Options:
      --config     path to config YAML (default: config/mvp_config.yaml)
      --data       split: test (default) | eval
      --model      model/adapter HF id (default: Qwen/Qwen3-4B)
      --min-n      min rows per side for a Δ in the figure (default: 10)
      --resamples  bootstrap replicates (default: 2000)
      --seed       bootstrap seed (default: 88)

Input:
    {outputs_dir}/{model}/{dataset}/{hint}/unhinted.jsonl[.gz]
    {analysis_dir}/{model}/{dataset}/{hint}/results_per_instance.csv

Output:
    outputs-and-results/monitorability-results/consistency_controlled_regression_{model}.md
    {plots_dir}/../extra-plots/task-performance-vs-monitorability/
        consistency_within_dataset_length_terciles_{model}.png
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

GREY = "#9AA7B4"   # inconsistent
NAVY = "#1B3A6B"   # consistent

MODELS = [
    ("(1) consistent only", dict(fe=False, length=False, p0=False)),
    ("(2) + dataset + hint type", dict(fe=True, length=False, p0=False)),
    ("(3) + log unhinted length", dict(fe=True, length=True, p0=False)),
    ("(4) + p0 > 0", dict(fe=True, length=True, p0=True)),
]


def open_jsonl(base, stem):
    plain, gz = base / f"{stem}.jsonl", base / f"{stem}.jsonl.gz"
    if plain.exists():
        return plain.open()
    if gz.exists():
        return gzip.open(gz, "rt")
    return None


def missing_value(v):
    return v in ("", "None", "N/A", None)


def load_rows(root, config, model_name, split):
    """One dict per entering (dataset, hint, question) row."""
    rows, missing = [], []
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
                    # Same validity rule as steps 31/36 (pipeline's u_valid).
                    if r.get("truncated", False):
                        continue
                    pa = r.get("parsed_answer")
                    if missing_value(pa):
                        continue
                    answers[int(r["instance_id"])].append(pa)
            break  # shared unhinted arm: one copy is enough
        if not answers:
            missing.append(f"{dataset} (no unhinted arm)")
            continue
        consistent = {iid: Counter(a).most_common(1)[0][1] == len(a)
                      for iid, a in answers.items()}
        for hint in HINT_TYPES:
            paths = resolve_paths(root, config, dataset, model_name, hint,
                                  split=split)
            csv_path = paths["analysis_dir"] / "results_per_instance.csv"
            if not csv_path.exists():
                missing.append(f"{dataset}/{hint}")
                continue
            with csv_path.open() as f:
                for row in csv.DictReader(f):
                    if any(missing_value(row.get(k)) for k in
                           ("gmean2_i", "acc_u_i", "p0_i",
                            "mean_cot_tokens_unhinted")):
                        continue
                    iid = int(row["instance_id"])
                    if iid not in consistent:
                        continue
                    rows.append(dict(
                        dataset=dataset, hint=hint,
                        qid=f"{dataset}:{iid}",
                        cons=float(consistent[iid]),
                        g=float(row["gmean2_i"]),
                        length=float(row["mean_cot_tokens_unhinted"]),
                        p0=float(float(row["p0_i"]) > 0)))
    return rows, missing


def design(rows, fe, length, p0):
    cols = [[1.0] * len(rows), [r["cons"] for r in rows]]
    if fe:
        cols += [[float(r["dataset"] == d) for r in rows] for d in DATASETS[1:]]
        cols += [[float(r["hint"] == h) for r in rows] for h in HINT_TYPES[1:]]
    if length:
        cols.append([np.log(max(r["length"], 1.0)) for r in rows])
    if p0:
        cols.append([r["p0"] for r in rows])
    return np.array(cols).T, np.array([r["g"] for r in rows])


def cons_coef(rows, spec):
    X, y = design(rows, **spec)
    return float(np.linalg.lstsq(X, y, rcond=None)[0][1])


def regressions(rows, n_resamples, rng):
    """[(label, estimate, lo, hi)] — CIs from a cluster bootstrap over
    questions, the same resampled clusters reused across the ladder."""
    by_q = defaultdict(list)
    for r in rows:
        by_q[r["qid"]].append(r)
    qids = list(by_q)
    est = [cons_coef(rows, spec) for _, spec in MODELS]
    boots = [[] for _ in MODELS]
    for _ in range(n_resamples):
        sample = [r for q in rng.choice(qids, len(qids)) for r in by_q[q]]
        for k, (_, spec) in enumerate(MODELS):
            boots[k].append(cons_coef(sample, spec))
    out = []
    for (label, _), e, b in zip(MODELS, est, boots):
        lo, hi = np.percentile(b, [2.5, 97.5])
        out.append((label, e, float(lo), float(hi)))
    return out


def boot_ci(v, n_resamples, rng):
    v = np.asarray(v, float)
    if len(v) < 2:
        return float("nan"), float("nan")
    m = v[rng.integers(0, len(v), (n_resamples, len(v)))].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def within_dataset_terciles(rows, n_resamples, rng):
    """panels[ds] = [(label, (con mean, ci, n), (inc mean, ci, n))] with
    tercile cuts from that dataset's own lengths."""
    panels = {}
    for ds in DATASETS:
        R = [r for r in rows if r["dataset"] == ds]
        if not R:
            continue
        L = np.array([r["length"] for r in R])
        t1, t2 = np.percentile(L, [100 / 3, 200 / 3])
        bands = [(f"Short\n≤{t1:.0f}", L <= t1),
                 (f"Medium\n{t1:.0f}–{t2:.0f}", (L > t1) & (L <= t2)),
                 (f"Long\n>{t2:.0f}", L > t2)]
        G = np.array([r["g"] for r in R])
        C = np.array([r["cons"] for r in R]) > 0.5
        out = []
        for label, mask in bands:
            sides = []
            for sel in (mask & C, mask & ~C):
                v = G[sel]
                sides.append((float(v.mean()) if len(v) else float("nan"),
                              boot_ci(v, n_resamples, rng), len(v)))
            out.append((label, sides[0], sides[1]))
        panels[ds] = out
    return panels


def plot_panels(panels, min_n, model_name, out):
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.2), sharey=True)
    bounds = [b for p in panels.values() for _, c, i in p
              for b in (*c[1], *i[1]) if np.isfinite(b)]
    ymin, ymax = min(bounds), max(bounds)
    n_pos = n_tot = 0
    off = 0.16
    for ax, ds in zip(axes.flat, DATASETS):
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color="#E5E5E5", linewidth=0.8)
        ax.set_title(ds, fontsize=11, fontweight="bold", pad=26)
        p = panels.get(ds)
        if not p:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", color="#999999")
            continue
        for k, (label, (cm, cci, cn), (im, ici, inn)) in enumerate(p):
            ok = cn >= min_n and inn >= min_n
            for x, m, ci, n, color in ((k - off, im, ici, inn, GREY),
                                       (k + off, cm, cci, cn, NAVY)):
                if not n:
                    continue
                face = color if ok else "white"
                ax.errorbar(x, m, yerr=[[m - ci[0]], [ci[1] - m]]
                            if np.isfinite(ci[0]) else None,
                            fmt="o", color=color, ecolor=color,
                            markerfacecolor=face, markeredgecolor=color,
                            markersize=7, elinewidth=1.4, capsize=3,
                            zorder=3)
                ax.text(x, (ci[0] if np.isfinite(ci[0]) else m) - 0.012,
                        f"n={n}", ha="center", va="top", fontsize=6.5,
                        color="#999999")
            if cn and inn:
                ax.plot([k - off, k + off], [im, cm], color="#BBBBBB",
                        linestyle="--", linewidth=0.9, zorder=2)
            if ok:
                d = cm - im
                n_tot += 1
                n_pos += d > 0
                ax.text(k, ymax + 0.04, f"Δ {d:+.2f}", ha="center",
                        va="bottom", fontsize=8, color="#333333")
            else:
                ax.text(k, ymax + 0.04, f"n < {min_n}", ha="center",
                        va="bottom", fontsize=7.5, color="#AAAAAA")
        ax.set_xticks(range(len(p)))
        ax.set_xticklabels([b[0] for b in p], fontsize=8)
        ax.set_xlim(-0.6, len(p) - 0.4)
        ax.set_ylim(ymin - 0.06, ymax + 0.09)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#999999")
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean per-question g-mean²")
    for ax in axes[-1, :]:
        ax.set_xlabel("Unhinted length tercile (tokens, this dataset)",
                      fontsize=9)
    fig.suptitle("Consistent vs inconsistent questions within each dataset's "
                 f"own length terciles — {model_name}\n(hint types pooled; "
                 f"Δ = consistent − inconsistent; hollow = fewer than {min_n} "
                 f"rows on a side; consistent higher in {n_pos} of {n_tot} "
                 "comparable groups)", fontsize=12)
    handles = [
        Line2D([], [], marker="o", linestyle="none", color=NAVY,
               markersize=8, label="Consistent (same answer in all valid runs)"),
        Line2D([], [], marker="o", linestyle="none", color=GREY,
               markersize=8, label="Inconsistent (answers vary)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.02), frameon=True,
               edgecolor="#CCCCCC", fontsize=9.5)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}  (consistent higher in {n_pos}/{n_tot} groups)")


def write_table(results, rows, model_name, n_resamples, out):
    n_q = len({r["qid"] for r in rows})
    n_c = int(sum(r["cons"] for r in rows))
    lines = [
        f"# Consistency effect on monitorability, with controls — "
        f"{model_name}",
        "",
        "OLS of per-question g-mean² on `consistent` (same answer in all "
        "valid unhinted runs) plus controls. Coefficient = difference in "
        "g-mean² between consistent and inconsistent questions holding the "
        "controls fixed. 95% CI from a cluster bootstrap over questions "
        f"({n_resamples} resamples). Generated by "
        "`scripts/55_consistency_controlled.py`.",
        "",
        f"Rows: {len(rows)} question × hint slices ({n_q} questions; "
        f"{n_c} consistent, {len(rows) - n_c} inconsistent).",
        "",
        "| Model | Consistency effect | 95% CI |",
        "|---|---:|---|",
    ]
    for label, e, lo, hi in results:
        lines.append(f"| {label} | {e:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    lines += [
        "",
        "Controls: dataset and hint type as fixed effects; length = "
        "log(mean unhinted CoT tokens); p0 > 0 = the model picks the hint "
        "target in at least one unhinted run (occurs only among inconsistent "
        "questions). TPR_mc already corrects for p0 via R_effect, so (4) is a "
        "bound on how much of the effect could be tied to that correction, "
        "not a correction the metric is missing.",
        "",
    ]
    out.write_text("\n".join(lines))
    print(f"Wrote {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Controlled consistency estimates + within-dataset "
                    "length-tercile figure.")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--data", default="test", choices=SPLITS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--min-n", type=int, default=10)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=88)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    with config_path.open() as f:
        config = yaml.safe_load(f)

    model_name = model_dir_name(args.model)
    rng = np.random.default_rng(args.seed)

    rows, missing = load_rows(root, config, model_name, args.data)
    if missing:
        print("WARNING: missing inputs skipped: " + ", ".join(missing))
    if not rows:
        print("ERROR: no data found.")
        sys.exit(1)

    results = regressions(rows, args.resamples, rng)
    print(f"\n{'model':<28} effect   95% CI")
    for label, e, lo, hi in results:
        print(f"{label:<28} {e:+.3f}  [{lo:+.3f}, {hi:+.3f}]")

    # The table sits with the main results (one level above the per-model
    # analysis dirs), the figure with the other task-performance plots.
    any_paths = resolve_paths(root, config, DATASETS[0], model_name,
                              HINT_TYPES[0], split=args.data)
    table_dir = any_paths["analysis_dir"].parents[2]
    write_table(results, rows, model_name, args.resamples,
                table_dir / f"consistency_controlled_regression_{model_name}.md")

    plots_dir = ((root / config["paths"]["plots_dir"]).parent
                 / "extra-plots" / "task-performance-vs-monitorability")
    if args.data != "test":
        plots_dir = plots_dir / args.data
    plots_dir.mkdir(parents=True, exist_ok=True)
    panels = within_dataset_terciles(rows, args.resamples, rng)
    plot_panels(panels, args.min_n, model_name,
                plots_dir / "consistency_within_dataset_length_terciles_"
                            f"{model_name}.png")


if __name__ == "__main__":
    main()
