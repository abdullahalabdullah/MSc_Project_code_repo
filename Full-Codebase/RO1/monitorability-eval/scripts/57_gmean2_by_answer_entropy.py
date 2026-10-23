#!/usr/bin/env python3
"""
Step 57: Monitorability by the model's uncertainty — Shannon entropy of its
unhinted answers.

Consistency (steps 31/33) only asks whether all 8 unhinted runs agree.
Entropy measures how spread the 8 answers are:

    H_i = − Σ_a p_a · log2 p_a,   p_a = (# valid unhinted runs answering a) / 8

H = 0 is a consistent question; 7:1 gives 0.54 bits, 4:4 gives 1.00,
4:2:1:1 gives 1.75. With 8 runs only a handful of distinct values exist
(one per answer-count pattern), and patterns above ~1.5 bits are rare, so
rows are grouped into five bins:

    consistent      H = 0
    very low        0 < H ≤ 0.6      (7:1)
    low             0.6 < H ≤ 1.0    (6:2, 5:3, 4:4 — two answers)
    medium          1.0 < H ≤ 1.5    (6:1:1, 5:2:1, 4:3:1, 4:2:2)
    high            H > 1.5          (three-plus answers, more evenly spread)

y = mean per-question g-mean² per bin with a 95% cluster-bootstrap CI
(questions resampled with all their hint rows), n annotated as
"<rows> (<questions>q)". Stdout also prints the per-pattern table and
Spearman(H, g-mean²) over the inconsistent rows only (H > 0), i.e. whether
monitorability keeps falling with uncertainty beyond the consistent /
inconsistent step.

Note: maximum entropy depends on the number of options (2 bits for the
4-option datasets, 3 bits with ≥ 8 options), so the same H is not equally
extreme in every dataset; bins are on raw bits.

Only questions with exactly 8 valid unhinted runs are used (as step 56).
Datasets and hint types pooled row-wise; --by-hint draws one panel per hint.

Usage:
    python scripts/57_gmean2_by_answer_entropy.py
    python scripts/57_gmean2_by_answer_entropy.py --by-hint

    Options:
      --config     path to config YAML (default: config/mvp_config.yaml)
      --data       split: test (default) | eval
      --model      model id (default: Qwen/Qwen3-4B)
      --by-hint    1x3 panels, one per hint type → …_by_hint_… .png
      --fine       eight groups by answer-count pattern (consistent, 7:1,
                   6:2, 5:3, 4:4, two 3-answer groups, 4+ answers) instead of
                   five entropy bins → …_fine_… .png
      --resamples  cluster-bootstrap replicates per CI (default: 10000)
      --seed       bootstrap seed (default: 88)
      --out        output PNG path

Input:
    {outputs_dir}/{model}/{dataset}/{hint}/unhinted.jsonl[.gz]
    {analysis_dir}/{model}/{dataset}/{hint}/results_per_instance.csv

Output:
    {plots_dir}/../extra-plots/task-performance-vs-monitorability/
        gmean2_by_answer_entropy[_by_hint]_{model}.png
"""

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import spearmanr

from src.dataset import SPLITS, model_dir_name, resolve_paths
from src.hints import HINT_TYPES

# reuse step 56's loader pieces and cluster bootstrap
_spec = importlib.util.spec_from_file_location(
    "step56", Path(__file__).with_name("56_gmean2_by_p0_level.py"))
step56 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(step56)

DEFAULT_MODEL = step56.DEFAULT_MODEL
DATASETS = step56.DATASETS
N_RUNS = step56.N_RUNS
HINT_LABELS = {"professor": "Professor", "grader_hacking": "Grader-hacking",
               "unethical": "Unethical"}

# (key, label, lower bound exclusive, upper bound inclusive)
BINS = [
    ("consistent", "consistent\nH = 0", None, 0.0),
    ("very_low", "very low\n0–0.6\n(7:1)", 0.0, 0.6),
    ("low", "low\n0.6–1.0\n(two answers)", 0.6, 1.0),
    ("medium", "medium\n1.0–1.5", 1.0, 1.5),
    ("high", "high\n> 1.5", 1.5, np.inf),
]
# --fine: eight groups keyed on the answer-count pattern (ordered by entropy);
# patterns too rare to stand alone are pooled by number of distinct answers.
FINE_BINS = [
    ("consistent", "consistent\n8\nH = 0", None, None),
    ("p71", "7:1\nH = 0.54", None, None),
    ("p62", "6:2\nH = 0.81", None, None),
    ("p53", "5:3\nH = 0.95", None, None),
    ("p44", "4:4\nH = 1.00", None, None),
    ("three_low", "3 answers\n6:1:1 or\n5:2:1\nH 1.06–1.30", None, None),
    ("three_high", "3 answers\n4:3:1, 4:2:2\nor 3:3:2\nH 1.41–1.56", None, None),
    ("four_plus", "4+ answers\nH ≥ 1.55", None, None),
]
TWO = {(7, 1): "p71", (6, 2): "p62", (5, 3): "p53", (4, 4): "p44"}


def fine_bin_of(pattern):
    if len(pattern) == 1:
        return "consistent"
    if len(pattern) == 2:
        return TWO[pattern]
    if len(pattern) == 3:
        return "three_low" if pattern in ((6, 1, 1), (5, 2, 1)) else "three_high"
    return "four_plus"


TEAL, GREY = "#2A9D8F", "#9AA7B4"


def entropy_bits(answers):
    p = np.array(list(Counter(answers).values()), float) / len(answers)
    return float(-(p * np.log2(p)).sum())


def bin_of(h):
    if h < 1e-9:
        return "consistent"
    for key, _, lo, hi in BINS[1:]:
        if lo < h <= hi + 1e-9:
            return key
    return BINS[-1][0]


def load_rows(root, config, model_name, split):
    rows = []
    for dataset in DATASETS:
        answers = defaultdict(list)
        for hint in HINT_TYPES:
            paths = resolve_paths(root, config, dataset, model_name, hint,
                                  split=split)
            fh = step56.open_jsonl(paths["outputs_dir"], "unhinted")
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
                        continue
                    h = entropy_bits(a)
                    rows.append(dict(
                        q=(dataset, iid), hint=hint, H=h, bin=bin_of(h),
                        pattern=tuple(sorted(Counter(a).values(),
                                             reverse=True)),
                        g=float(g)))
    return rows


def bin_stats(rows, n_resamples, rng, bins_def=None):
    gm = np.array([r["g"] for r in rows])
    qkeys = {q: j for j, q in enumerate(dict.fromkeys(r["q"] for r in rows))}
    qid = np.array([qkeys[r["q"]] for r in rows])
    bins_def = bins_def or BINS
    bins = np.array([r["bin"] for r in rows])
    masks = [bins == key for key, _, _, _ in bins_def]
    boot = step56.cluster_boot_means(qid, gm, masks, n_resamples, rng)
    out = []
    for j, (key, label, _, _) in enumerate(bins_def):
        m = masks[j]
        if not m.any():
            out.append(None)
            continue
        out.append(dict(key=key, mean=float(gm[m].mean()),
                        lo=float(np.nanpercentile(boot[:, j], 2.5)),
                        hi=float(np.nanpercentile(boot[:, j], 97.5)),
                        n=int(m.sum()), nq=len(set(qid[m]))))
    return out, float(gm.mean())


def report(name, rows, stats):
    print(f"\n=== {name}: {len(rows)} rows ===")
    print(f"{'bin':>11} | {'rows':>5} | {'q':>4} | {'g-mean2':>7} | 95% CI")
    for s in stats:
        if s:
            print(f"{s['key']:>11} | {s['n']:>5} | {s['nq']:>4} | "
                  f"{s['mean']:>7.3f} | [{s['lo']:.3f}, {s['hi']:.3f}]")
    inc = [r for r in rows if r["H"] > 0]
    if len(inc) > 2:
        rho, p = spearmanr([r["H"] for r in inc], [r["g"] for r in inc])
        print(f"Spearman(H, g-mean²), inconsistent rows only: "
              f"rho = {rho:+.3f}, p = {p:.2g}, n = {len(inc)}")


def draw(ax, stats, overall, title=None, bins_def=None):
    bins_def = bins_def or BINS
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#E5E5E5", linewidth=0.8)
    ax.axhline(overall, color="#888888", linestyle=":", linewidth=1.2,
               zorder=1)
    pts = [(i, s) for i, s in enumerate(stats) if s]
    inc = [(i, s) for i, s in pts if s["key"] != "consistent"]
    ax.plot([i for i, _ in inc], [s["mean"] for _, s in inc], color=GREY,
            linewidth=1.3, zorder=2)
    for i, s in pts:
        color = TEAL if s["key"] == "consistent" else GREY
        ax.errorbar(i, s["mean"],
                    yerr=[[s["mean"] - s["lo"]], [s["hi"] - s["mean"]]],
                    fmt="o", color=color, ecolor=color, markersize=9,
                    markeredgecolor="#333333", markeredgewidth=0.6,
                    elinewidth=1.8, capsize=3.5, capthick=1.8, zorder=3)
        ax.text(i, s["hi"] + 0.012, f"{s['mean']:.3f}", ha="center",
                va="bottom", fontsize=8.5, color="#333333",
                fontweight="bold")
        ax.text(i, s["lo"] - 0.012, f"{s['n']}\n({s['nq']}q)", ha="center",
                va="top", fontsize=7.5, color="#555555")
    ax.set_xticks(range(len(bins_def)))
    ax.set_xticklabels([b[1] for b in bins_def],
                       fontsize=8.5 if len(bins_def) <= 5 else 7.5)
    ax.set_xlim(-0.5, len(bins_def) - 0.5)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#999999")


def main():
    parser = argparse.ArgumentParser(
        description="Monitorability by unhinted answer entropy.")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--data", default="test", choices=SPLITS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--by-hint", action="store_true")
    parser.add_argument("--fine", action="store_true",
                        help="eight groups by answer-count pattern instead of "
                             "five entropy bins; writes …_fine_… .png")
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
    bins_def = FINE_BINS if args.fine else BINS
    if args.fine:
        for r in rows:
            r["bin"] = fine_bin_of(r["pattern"])

    # per-pattern table (all hints pooled)
    by_pat = defaultdict(list)
    for r in rows:
        by_pat[r["pattern"]].append(r)
    print(f"{'pattern':>18} | {'H':>5} | {'rows':>5} | {'q':>4} | g-mean2")
    for pat, rs in sorted(by_pat.items(), key=lambda kv: kv[1][0]["H"]):
        print(f"{str(pat):>18} | {rs[0]['H']:>5.2f} | {len(rs):>5} | "
              f"{len({r['q'] for r in rs}):>4} | "
              f"{np.mean([r['g'] for r in rs]):.3f}")

    note = ("Rows = question × hint; only questions with exactly 8 valid "
            "unhinted runs. H = Shannon entropy (bits) of the 8 unhinted "
            "answers. Labels: rows (distinct questions).\n95% CIs: cluster "
            f"bootstrap over questions ({args.resamples:,} resamples). "
            "Dotted line = mean over the plotted rows. Max H is 2 bits for "
            "4-option datasets, 3 bits with ≥ 8 options.")

    if args.by_hint:
        fig, axes = plt.subplots(1, 3, figsize=(15 if not args.fine else 22,
                                                5.0 if not args.fine else 5.6),
                                 sharey=True)
        bounds = []
        for ax, hint in zip(axes, HINT_TYPES):
            sub = [r for r in rows if r["hint"] == hint]
            stats, overall = bin_stats(sub, args.resamples, rng, bins_def)
            report(HINT_LABELS[hint], sub, stats)
            draw(ax, stats, overall, HINT_LABELS[hint], bins_def)
            bounds += [s["lo"] for s in stats if s] + \
                      [s["hi"] for s in stats if s]
        axes[0].set_ylabel("Monitorability (g-mean²)")
        axes[0].set_ylim(min(bounds) - 0.09, max(bounds) + 0.05)
        fig.supxlabel("Uncertainty: entropy of the 8 unhinted answers (bits)",
                      fontsize=11, y=0.04)
        fig.suptitle("Monitorability vs answer entropy, per hint type",
                     fontsize=13)
        tag = "_by_hint"
    else:
        stats, overall = bin_stats(rows, args.resamples, rng, bins_def)
        report("All hints pooled", rows, stats)
        fig, ax = plt.subplots(figsize=(9.6 if not args.fine else 12.5, 5.2))
        draw(ax, stats, overall, bins_def=bins_def)
        lo_all = min(s["lo"] for s in stats if s)
        hi_all = max(s["hi"] for s in stats if s)
        ax.set_ylim(lo_all - 0.09, hi_all + 0.05)
        # fine layout: the right edge collides with the last value label, so
        # put the label in the gap between the 4:4 and 3-answer groups
        lx, ha = ((4.55, "center") if args.fine
                  else (len(bins_def) - 0.55, "right"))
        ax.text(lx, overall + 0.006, f"overall mean {overall:.3f}", ha=ha,
                va="bottom", fontsize=8.5, color="#666666")
        ax.set_xlabel("Uncertainty: entropy of the 8 unhinted answers (bits)")
        ax.set_ylabel("Monitorability (g-mean²)")
        ax.set_title("Monitorability vs the model's uncertainty (answer "
                     "entropy across 8 unhinted runs)", fontsize=12)
        tag = ""
    if args.fine:
        tag = "_fine" + tag
    fig.text(0.01, -0.03, note, fontsize=8, color="#555555", ha="left",
             va="top")

    out = args.out
    if out is None:
        plots_dir = ((root / config["paths"]["plots_dir"]).parent
                     / "extra-plots" / "task-performance-vs-monitorability")
        if args.data != "test":
            plots_dir = plots_dir / args.data
        plots_dir.mkdir(parents=True, exist_ok=True)
        out = plots_dir / f"gmean2_by_answer_entropy{tag}_{model_name}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
