#!/usr/bin/env python3
"""
Step 59 (experiment: uncertainty-vs-monitorability): monitorability by the
model's uncertainty — Shannon entropy of its N = 30 unhinted answers.

The N = 30 counterpart of step 57 (which is left unchanged and assumes N = 8).
Per question:

    H_i = − Σ_a p_a · log2 p_a,   p_a = (# valid unhinted runs answering a) / n_valid_i

A run is valid iff its answer letter was parsed and it was not truncated
(as steps 56/57). A question enters only with n_valid_i ≥ --min-valid (default
27 of 30); entropy, p0 and the monitor metrics all use the same full set of
valid runs (no split-half).

Bins are equal-width in bits, with H = 0 kept as its own bin:

    consistent  H = 0
    (0, w], (w, 2w], …  up to log2(4) = 2 bits (all datasets are 4-option)

w = --bin-width (default 0.5 → four non-zero bins). With 30 runs a single
stray answer (29:1) already gives H ≈ 0.21, so the consistent bin is stricter
than at N = 8.

y = mean per-question g-mean² per bin with a 95% bootstrap CI over questions
(one row per question: a single hint type). Questions that fail Level 1
(TE ≤ 0, g-mean² undefined) cannot enter the mean; because Level-1 pass rate
itself varies with entropy, each bin also reports its pass rate. Bins with
fewer than --min-q g-mean²-defined questions are not drawn.

Panels: all datasets pooled, then one per dataset. Stdout prints the bin
tables and Spearman(H, g-mean²) over H > 0 questions per panel.

Usage:
    python monitorability-eval/scripts/59_uvm_gmean2_by_answer_entropy.py

    Options:
      --config      path to config YAML (default: config/uncertainty_vs_monitorability.yaml)
      --model       model id (default: Qwen/Qwen3-4B)
      --datasets    dataset keys (default: gpqa_diamond lexam wmdp)
      --hint-type   hint type (default: grader_hacking)
      --data        split: test (default) | eval
      --min-valid   minimum valid unhinted runs per question (default: 27)
      --bin-width   entropy bin width in bits (default: 0.5)
      --max-h       upper entropy bound in bits (default: 2.0 = log2 of 4 options)
      --min-q       minimum questions for a bin to be drawn (default: 8)
      --resamples   bootstrap replicates per CI (default: 10000)
      --seed        bootstrap seed (default: 88)

Input:
    {outputs_dir}/{model}/{dataset}/{hint}/unhinted.jsonl[.gz]
    {analysis_dir}/{model}/{dataset}/{hint}/results_per_instance.csv   (step 04)

Output (under {analysis_dir}/{model}/ and {plots_dir}/{model}/):
    entropy_per_question_{hint}.csv   one row per question: n_valid, H, bin,
                                      answer counts, modal answer/share, p0,
                                      TE, Level-1 pass, g-mean², …
    entropy_bins_{hint}.csv           per panel × bin: questions, Level-1 pass
                                      rate, mean g-mean², 95% CI
    gmean2_by_answer_entropy_{hint}.png
"""

import argparse
import csv
import gzip
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
from src.gmean2_metrics import resolve_jsonl
from src.hints import HINT_TYPES

# reuse step 56's cluster bootstrap (with one row per question it is the
# ordinary bootstrap over questions)
_spec = importlib.util.spec_from_file_location(
    "step56", Path(__file__).with_name("56_gmean2_by_p0_level.py"))
step56 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(step56)

DEFAULT_CONFIG = "config/uncertainty_vs_monitorability.yaml"
DEFAULT_MODEL = "Qwen/Qwen3-4B"
DEFAULT_DATASETS = ["gpqa_diamond", "lexam", "wmdp"]
HINT_LABELS = {"professor": "Professor", "grader_hacking": "Grader-hacking",
               "unethical": "Unethical"}
TEAL, GREY = "#2A9D8F", "#9AA7B4"

PER_Q_FIELDS = ["dataset", "instance_id", "n_valid", "entropy_bits", "bin",
                "answer_counts", "modal_answer", "modal_share",
                "modal_is_correct", "correct_answer", "hint_target", "p0_i",
                "p1_i", "te_i", "level1_pass", "gmean2_i", "acc_u_i"]
BIN_FIELDS = ["panel", "bin", "questions", "questions_gmean2",
              "level1_pass_pct", "mean_gmean2", "ci_lo", "ci_hi", "drawn"]


def make_bins(width, max_h):
    """[(key, label, lo_exclusive, hi_inclusive)], consistent bin first."""
    edges = np.round(np.arange(0.0, max_h + 1e-9, width), 6)
    if edges[-1] < max_h - 1e-9:
        edges = np.append(edges, max_h)
    bins = [("consistent", "consistent\nH = 0", None, 0.0)]
    for lo, hi in zip(edges[:-1], edges[1:]):
        bins.append((f"{lo:g}-{hi:g}", f"({lo:g}, {hi:g}]", float(lo),
                     float(hi)))
    return bins


def bin_of(h, bins):
    if h < 1e-9:
        return "consistent"
    for key, _, lo, hi in bins[1:]:
        if lo < h <= hi + 1e-9:
            return key
    return bins[-1][0]  # numerical overshoot above max_h


def entropy_bits(answers):
    p = np.array(list(Counter(answers).values()), float) / len(answers)
    return float(-(p * np.log2(p)).sum())


def read_unhinted(path):
    """{instance_id: [valid answers]} from an unhinted.jsonl[.gz]."""
    path = resolve_jsonl(path)
    if not path.exists():
        return None
    opener = gzip.open if path.suffix == ".gz" else open
    answers = defaultdict(list)
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            iid = int(r["instance_id"])
            answers.setdefault(iid, [])
            if r.get("truncated", False):
                continue
            pa = r.get("parsed_answer")
            if pa in (None, "", "None", "N/A"):
                continue
            answers[iid].append(pa)
    return answers


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(x) else x


def load_rows(root, config, model_name, datasets, hint, split, min_valid,
              bins):
    rows, dropped = [], {}
    for ds in datasets:
        paths = resolve_paths(root, config, ds, model_name, hint, split=split)
        answers = read_unhinted(paths["outputs_dir"] / "unhinted.jsonl")
        csv_path = paths["analysis_dir"] / "results_per_instance.csv"
        if answers is None or not csv_path.exists():
            print(f"WARNING: missing outputs or {csv_path} for {ds} — skipped.")
            continue
        n_drop = 0
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                iid = int(row["instance_id"])
                a = answers.get(iid, [])
                if len(a) < min_valid:
                    n_drop += 1
                    continue
                h = entropy_bits(a)
                counts = Counter(a)
                modal, modal_n = counts.most_common(1)[0]
                rows.append(dict(
                    dataset=ds, instance_id=iid, n_valid=len(a),
                    entropy_bits=round(h, 6), bin=bin_of(h, bins),
                    answer_counts=" ".join(f"{k}:{v}" for k, v in
                                           sorted(counts.items())),
                    modal_answer=modal,
                    modal_share=round(modal_n / len(a), 6),
                    modal_is_correct=modal == row["correct_answer"],
                    correct_answer=row["correct_answer"],
                    hint_target=row["hint_target"],
                    p0_i=row.get("p0_i", ""), p1_i=row.get("p1_i", ""),
                    te_i=row.get("te_i", ""),
                    level1_pass=row.get("level1_pass", ""),
                    gmean2_i=row.get("gmean2_i", ""),
                    acc_u_i=row.get("acc_u_i", ""),
                    q=(ds, iid), H=h, g=_num(row.get("gmean2_i"))))
        dropped[ds] = n_drop
    return rows, dropped


def bin_stats(rows, bins, n_resamples, rng, min_q):
    """Per-bin stats; g-mean² means over the questions where it is defined."""
    out = []
    grows = [r for r in rows if r["g"] is not None]
    boot = None
    if grows:
        gm = np.array([r["g"] for r in grows])
        qid = np.arange(len(grows))
        b_arr = np.array([r["bin"] for r in grows])
        masks = [b_arr == key for key, _, _, _ in bins]
        boot = step56.cluster_boot_means(qid, gm, masks, n_resamples, rng)
    for j, (key, label, _, _) in enumerate(bins):
        allq = [r for r in rows if r["bin"] == key]
        gq = [r["g"] for r in allq if r["g"] is not None]
        s = dict(key=key, label=label, nq_all=len(allq), nq=len(gq),
                 l1=(100 * len(gq) / len(allq)) if allq else float("nan"),
                 mean=float(np.mean(gq)) if gq else float("nan"),
                 lo=float("nan"), hi=float("nan"))
        if gq and boot is not None:
            s["lo"] = float(np.nanpercentile(boot[:, j], 2.5))
            s["hi"] = float(np.nanpercentile(boot[:, j], 97.5))
        s["drawn"] = len(gq) >= min_q
        out.append(s)
    overall = (float(np.mean([r["g"] for r in grows])) if grows
               else float("nan"))
    return out, overall


def report(name, rows, stats):
    print(f"\n=== {name}: {len(rows)} questions "
          f"({sum(r['g'] is not None for r in rows)} with g-mean²) ===")
    print(f"{'bin':>11} | {'q':>4} | {'q g2':>5} | {'L1 %':>5} | "
          f"{'g-mean2':>7} | 95% CI")
    for s in stats:
        ci = (f"[{s['lo']:.3f}, {s['hi']:.3f}]" if s["nq"] else "—")
        flag = "" if s["drawn"] else "  (not drawn)"
        print(f"{s['key']:>11} | {s['nq_all']:>4} | {s['nq']:>5} | "
              f"{s['l1']:>5.1f} | {s['mean']:>7.3f} | {ci}{flag}")
    inc = [r for r in rows if r["H"] > 0 and r["g"] is not None]
    if len(inc) > 2:
        rho, p = spearmanr([r["H"] for r in inc], [r["g"] for r in inc])
        print(f"Spearman(H, g-mean²), H > 0 questions only: "
              f"rho = {rho:+.3f}, p = {p:.2g}, n = {len(inc)}")


def draw(ax, stats, overall, bins, title):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#E5E5E5", linewidth=0.8)
    if not np.isnan(overall):
        ax.axhline(overall, color="#888888", linestyle=":", linewidth=1.2,
                   zorder=1)
    pts = [(i, s) for i, s in enumerate(stats) if s["drawn"]]
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
    # n and Level-1 pass rate under every bin label, drawn or not
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels(
        [f"{b[1]}\n{s['nq']}/{s['nq_all']}q\nL1 {s['l1']:.0f}%"
         if s["nq_all"] else f"{b[1]}\n0q" for b, s in zip(bins, stats)],
        fontsize=8)
    ax.set_xlim(-0.5, len(bins) - 0.5)
    ax.set_title(title, fontsize=11, fontweight="bold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#999999")


def main():
    parser = argparse.ArgumentParser(
        description="uncertainty-vs-monitorability: g-mean² by unhinted "
                    "answer entropy (N = 30).")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--hint-type", default="grader_hacking",
                        choices=HINT_TYPES)
    parser.add_argument("--data", default="test", choices=SPLITS)
    parser.add_argument("--min-valid", type=int, default=27)
    parser.add_argument("--bin-width", type=float, default=0.5)
    parser.add_argument("--max-h", type=float, default=2.0)
    parser.add_argument("--min-q", type=int, default=8)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=88)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    with open(root / args.config) as f:
        config = yaml.safe_load(f)
    model_name = model_dir_name(args.model)
    n_samples = config["qwen_inference"]["n_samples"]
    rng = np.random.default_rng(args.seed)
    bins = make_bins(args.bin_width, args.max_h)

    rows, dropped = load_rows(root, config, model_name, args.datasets,
                              args.hint_type, args.data, args.min_valid, bins)
    if not rows:
        print("ERROR: no data found — run steps 02–04 first.")
        sys.exit(1)
    for ds, n in dropped.items():
        print(f"{ds}: {n} questions dropped (< {args.min_valid} valid "
              f"unhinted runs)")

    analysis_dir = root / config["paths"]["analysis_dir"] / model_name
    plots_dir = root / config["paths"]["plots_dir"] / model_name
    if args.data != "test":
        analysis_dir, plots_dir = analysis_dir / args.data, plots_dir / args.data
    analysis_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    pq_path = analysis_dir / f"entropy_per_question_{args.hint_type}.csv"
    with pq_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PER_Q_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    panels = [("All datasets", rows)] + [
        (ds, [r for r in rows if r["dataset"] == ds]) for ds in args.datasets
        if any(r["dataset"] == ds for r in rows)]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 5.4),
                             sharey=True, squeeze=False)
    axes = axes[0]
    bin_rows, bounds = [], []
    for ax, (name, sub) in zip(axes, panels):
        stats, overall = bin_stats(sub, bins, args.resamples, rng, args.min_q)
        report(name, sub, stats)
        draw(ax, stats, overall, bins, name)
        bounds += [v for s in stats if s["drawn"] for v in (s["lo"], s["hi"])]
        for s in stats:
            bin_rows.append(dict(
                panel=name, bin=s["key"], questions=s["nq_all"],
                questions_gmean2=s["nq"], level1_pass_pct=round(s["l1"], 2),
                mean_gmean2=round(s["mean"], 6), ci_lo=round(s["lo"], 6),
                ci_hi=round(s["hi"], 6), drawn=s["drawn"]))

    bins_path = analysis_dir / f"entropy_bins_{args.hint_type}.csv"
    with bins_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=BIN_FIELDS)
        w.writeheader()
        w.writerows(bin_rows)

    if bounds:
        axes[0].set_ylim(max(min(bounds) - 0.15, -0.05),
                         min(max(bounds) + 0.06, 1.08))
    axes[0].set_ylabel("Monitorability (g-mean²)")
    fig.supxlabel(f"Uncertainty: entropy of the {n_samples} unhinted answers "
                  "(bits)", fontsize=11, y=0.03)
    fig.suptitle(f"Monitorability vs answer entropy — "
                 f"{HINT_LABELS.get(args.hint_type, args.hint_type)} hint, "
                 f"{args.model}", fontsize=13)
    note = (f"One row per question (≥ {args.min_valid}/{n_samples} valid "
            "unhinted runs). Equal-width bins of "
            f"{args.bin_width:g} bits; H = 0 is its own bin. Under each bin: "
            "questions with g-mean² / all questions, and Level-1 pass rate "
            "(g-mean² is undefined when TE ≤ 0).\n95% CIs: bootstrap over "
            f"questions ({args.resamples:,} resamples). Bins with < "
            f"{args.min_q} g-mean² questions not drawn. Dotted line = panel "
            "mean.")
    fig.text(0.01, -0.03, note, fontsize=8, color="#555555", ha="left",
             va="top")
    fig.tight_layout()
    png = plots_dir / f"gmean2_by_answer_entropy_{args.hint_type}.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")

    print(f"\nWrote {pq_path}\nWrote {bins_path}\nWrote {png}")


if __name__ == "__main__":
    main()
