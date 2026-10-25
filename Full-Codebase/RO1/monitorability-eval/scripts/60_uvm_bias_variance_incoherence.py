#!/usr/bin/env python3
"""
Step 60 (experiment: uncertainty-vs-monitorability): bias-variance
decomposition of the model's errors over its N = 30 runs, error rate and
error-incoherence ("The Hot Mess of AI", Hägele et al. 2026, arXiv:2601.23045),
and monitorability (g-mean²) by per-question error-incoherence.

The decomposition is the paper's, computed with code adapted from its release
(https://github.com/haeggee/hot-mess-of-ai, MIT License) — see
src/hot_mess_bias_variance.py for the exact provenance and licence notice.
Each sampled answer is a one-hot "probability" vector (as the paper's code does
for sampled CoT answers); per question:

    KL:    Error = E[CE(y, f)] = Bias + Variance,
           Bias = KL(y || fbar), Variance = E[KL(fbar || f)],
           fbar ∝ exp(E[log f])     (Laplace-smoothed with EPS = 1e-12)
    0/1:   error = 1 - P(correct); bias = 1[modal answer wrong];
           variance = 1 - P(modal answer)
    Brier: Bias = ||mean(f) - y||², Variance = E||f - mean(f)||²

Headline metric (paper Eq. 2, pooled over questions):
    KL-Error-incoherence = Σ_i Variance_i / Σ_i Error_i
Per question (used for binning): incoherence_i = Variance_i / Error_i, which
for one-hot samples is ≈ (1 − P(modal)) / (1 − P(correct)):
    = 1     modal answer correct (all error is variance)
    = 0     the same wrong answer every run (all error is bias)
    in (0,1) modal answer wrong, runs vary
    undefined when the question is never answered wrong (no error).
The absolute KL values scale with −log(EPS) ≈ 27.6 nats per wrong run (one-hot
samples), so only KL ratios (incoherence) are interpretable; the 0/1 error rate
is the interpretable error magnitude.

Runs: the UNHINTED arm by default (the model's own answer distribution, as for
the entropy analysis in step 59; --arm hinted for the hinted runs). Only valid
runs (answer parsed, not truncated — as steps 56/57/59) are used; questions
need ≥ --min-valid of them. (The paper's code instead keeps every response and
maps unparsed ones to a uniform vector; our valid-run filter matches the
g-mean² computation.)

Figure: per-question g-mean² (step 04) by per-question KL-incoherence bin —
    no error | [0, 1/n] | … | (1−1/n, 1) | ≈ 1 (modal answer correct)
The last group is defined by the paper's 0/1 bias = 0 (modal answer correct),
not by the KL value: the EPS smoothing puts those questions at 0.95–1.0. The
equal-width bins therefore hold the questions whose modal answer is wrong.
mean g-mean² per bin with a 95% bootstrap CI over questions; under each bin:
questions with g-mean² / all questions, Level-1 pass rate, and the pooled
(Σvar/Σerr) KL-incoherence of the bin. Bins with < --min-q g-mean² questions
are not drawn. One panel per dataset, plus a pooled panel when > 1 dataset.

Usage:
    python monitorability-eval/scripts/60_uvm_bias_variance_incoherence.py
    python monitorability-eval/scripts/60_uvm_bias_variance_incoherence.py --datasets lexam gpqa_diamond

    Options:
      --config      path to config YAML (default: config/uncertainty_vs_monitorability.yaml)
      --model       model id (default: Qwen/Qwen3-4B)
      --datasets    dataset keys (default: lexam)
      --hint-type   hint type (default: grader_hacking)
      --arm         unhinted (default) | hinted — whose answers are decomposed
      --data        split: test (default) | eval
      --min-valid   minimum valid runs per question (default: 27)
      --n-bins      equal-width incoherence bins on [0, 1) for questions whose
                    modal answer is wrong (default: 3 — 0.25-wide bins leave the
                    top bin with < 8 g-mean² questions on lexam)
      --var-edges   variance-figure bin edges, as the share of runs not giving
                    the modal answer (default: 0.15 0.30 0.45)
      --bias-edges  bias-figure bin edges for questions whose modal answer is
                    wrong, as the gap P(modal) − P(correct) (default: 1/3 2/3)
      --min-q       minimum g-mean² questions for a bin to be drawn (default: 8)
      --resamples   bootstrap replicates per CI (default: 10000)
      --seed        bootstrap seed (default: 88)

Input:
    {outputs_dir}/{model}/{dataset}/{hint}/{arm}.jsonl[.gz]
    {analysis_dir}/{model}/{dataset}/{hint}/results_per_instance.csv   (step 04)
    {data_dir}/{dataset}/dataset_index.json                            (option count)

Output (under {analysis_dir}/{model}/ and {plots_dir}/{model}/):
    bias_variance_per_question_{hint}_{arm}.csv
    bias_variance_summary_{hint}_{arm}.{json,md}   error rate, pooled bias /
                                                   variance / error-incoherence
    incoherence_bins_{hint}_{arm}.csv
    gmean2_by_error_incoherence_{hint}_{arm}.png
    variance_bins_{hint}_{arm}.csv
    gmean2_by_error_variance_{hint}_{arm}.png      g-mean² by the paper's
                                                   KL-Variance (zero bin, then
                                                   --var-edges)
    bias_bins_{hint}_{arm}.csv
    gmean2_by_error_bias_{hint}_{arm}.png          g-mean² by the paper's
                                                   KL-Bias (≈ 0 split into no error
                                                   / modal correct, then
                                                   --bias-edges)
"""

import argparse
import csv
import gzip
import importlib.util
import json
import sys
import textwrap
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
from src.hot_mess_bias_variance import EPS, error_incoherence, process_question_metrics

# reuse step 59's per-bin statistics (bootstrap over questions)
_spec = importlib.util.spec_from_file_location(
    "step59", Path(__file__).with_name("59_uvm_gmean2_by_answer_entropy.py"))
step59 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(step59)

DEFAULT_CONFIG = "config/uncertainty_vs_monitorability.yaml"
DEFAULT_MODEL = "Qwen/Qwen3-4B"
HINT_LABELS = step59.HINT_LABELS
NO_ERR_COLOR, GREY, CORRECT_COLOR = "#2A9D8F", "#9AA7B4", "#E9C46A"

PER_Q_FIELDS = ["dataset", "instance_id", "n_options", "n_valid", "correct_answer",
                "answer_counts", "modal_answer", "accuracy", "error_01",
                "is_consensus_correct", "bias_01", "variance_01", "kl_bias",
                "kl_variance", "kl_error", "kl_incoherence", "brier_bias",
                "brier_variance", "brier_error", "brier_incoherence",
                "incoherence_bin", "level1_pass", "gmean2_i"]
BIN_FIELDS = ["panel", "bin", "questions", "questions_gmean2", "level1_pass_pct",
              "pooled_kl_incoherence", "mean_error_01", "mean_gmean2", "ci_lo",
              "ci_hi", "drawn"]


def make_bins(n_bins):
    """[(key, label, lo, hi)]: no-error, n_bins equal-width bins on [0, 1)
    (modal answer wrong), then ≈ 1 (modal answer correct)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = [("no_error", "no error\n(always\ncorrect)", None, None)]
    for j, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        left = "[" if j == 0 else "("
        right = ")" if j == n_bins - 1 else "]"
        bins.append((f"{lo:.2f}-{hi:.2f}", f"{left}{lo:.2f}, {hi:.2f}{right}", float(lo), float(hi)))
    bins.append(("one", "≈ 1\n(modal\nanswer\ncorrect)", 1.0, 1.0))
    return bins


def bin_of(inc, consensus_correct, bins):
    """no error -> "no_error"; modal answer correct (0/1 bias = 0, all error is
    variance; KL-incoherence 0.95-1 after the paper's EPS smoothing) -> "one";
    otherwise the equal-width bin of the KL-incoherence."""
    if inc is None or np.isnan(inc):
        return "no_error"
    if consensus_correct:
        return "one"
    for key, _, lo, hi in bins[1:-1]:
        if (inc >= lo if lo == 0 else inc > lo) and inc <= hi + 1e-12:
            return key
    return bins[-2][0]


def make_var_bins(edges_frac):
    """[(key, label, lo, hi)] on the paper's KL variance (nats): a zero bin
    (all runs identical), then bins with edges at fractions of -log(EPS) — with
    one-hot samples KL variance = (1 - P(modal answer)) * -log(EPS), so a
    fraction f means "a share f of the runs are not the modal answer"."""
    L = -np.log(EPS)
    fr = [0.0] + list(edges_frac) + [1.0]
    bins = [("zero", "0\n(all runs\nidentical)", 0.0, 0.0)]
    for lo, hi in zip(fr[:-1], fr[1:]):
        top = "max" if hi == 1.0 else f"{hi * L:.1f}"
        share = f"> {lo:.0%}" if hi == 1.0 else f"{lo:.0%}–{hi:.0%}"
        bins.append((f"v{lo:g}-{hi:g}", f"({lo * L:.1f}, {top}]\n{share} off\nmodal", lo * L, hi * L))
    return bins


def var_bin_of(v, vbins):
    if v <= 1e-6:
        return "zero"
    for key, _, lo, hi in vbins[1:]:
        if lo < v <= hi + 1e-9:
            return key
    return vbins[-1][0]


def make_bias_bins(edges_frac):
    """[(key, label, lo, hi)] on the paper's KL bias (nats): bias ≈ 0 split into
    "no_error" and "modal_correct", then bins with edges at fractions of
    -log(EPS) — with one-hot samples KL bias ≈ (P(modal) - P(correct)) *
    -log(EPS) when the modal answer is wrong."""
    L = -np.log(EPS)
    fr = [0.0] + list(edges_frac) + [1.0]
    bins = [("no_error", "≈ 0\nno error\n(always\ncorrect)", 0.0, 0.0),
            ("modal_correct", "≈ 0\nmodal\ncorrect,\nsome errors", 0.0, 0.0)]
    for lo, hi in zip(fr[:-1], fr[1:]):
        bins.append((f"b{lo:.2f}-{hi:.2f}",
                     f"({lo * L:.1f}, {hi * L:.1f}]\ngap {lo:.0%}–{hi:.0%}",
                     lo * L, hi * L))
    return bins


def bias_bin_of(r, bbins):
    if r["error_01"] <= 0:
        return "no_error"
    if r["is_consensus_correct"]:
        return "modal_correct"
    for key, _, lo, hi in bbins[2:]:
        if lo < r["kl_bias"] <= hi + 1e-9:
            return key
    return bbins[2][0]  # modal wrong only through a tie: smallest bias bin


def component_figure(metric, name, bins, bin_fn, colors, other, xlabel, explain,
                     scopes, rng, args, analysis_dir, plots_dir, tag, n_samples):
    """g-mean² by bins of one decomposition component (the paper's KL variance
    or KL bias): stdout table, {name}_bins_{tag}.csv and
    gmean2_by_error_{name}_{tag}.png. `other` = (field, label) of the other
    component, reported per bin."""
    okey, olabel = other
    for _, sub in scopes:
        for r in sub:
            r["bin"] = bin_fn(r)
    fig, axes = plt.subplots(1, len(scopes), figsize=(6.4 * len(scopes), 5.8),
                             sharey=True, squeeze=False)
    axes = axes[0]
    out_rows, bounds = [], []
    for ax, (panel, sub) in zip(axes, scopes):
        stats, overall = step59.bin_stats(sub, bins, args.resamples, rng, args.min_q)
        for s in stats:
            in_bin = [r for r in sub if r["bin"] == s["key"]]
            s["other"] = float(np.mean([r[okey] for r in in_bin])) if in_bin else float("nan")
            s["err"] = float(np.mean([r["error_01"] for r in in_bin])) if in_bin else float("nan")
        print(f"\n=== {panel}: g-mean² by KL {name} ===")
        print(f"{'bin':>14} | {'q':>4} | {'q g2':>5} | {'L1 %':>5} | {olabel:>7} | "
              f"{'err':>5} | {'g-mean2':>7} | 95% CI")
        for s in stats:
            ci = f"[{s['lo']:.3f}, {s['hi']:.3f}]" if s["nq"] else "—"
            flag = "" if s["drawn"] else "  (not drawn)"
            print(f"{s['key']:>14} | {s['nq_all']:>4} | {s['nq']:>5} | {s['l1']:>5.1f} | "
                  f"{s['other']:>7.2f} | {s['err']:>5.2f} | {s['mean']:>7.3f} | {ci}{flag}")
        gq = [r for r in sub if r["g"] is not None]
        if len(gq) > 2:
            rho, p = spearmanr([r[metric] for r in gq], [r["g"] for r in gq])
            print(f"Spearman(KL {name}, g-mean²): rho = {rho:+.3f}, p = {p:.2g}, n = {len(gq)}")
            ge = [r for r in gq if r["error_01"] > 0]
            rho, p = spearmanr([r[metric] for r in ge], [r["g"] for r in ge])
            print(f"Spearman(KL {name}, g-mean²), questions with error: "
                  f"rho = {rho:+.3f}, p = {p:.2g}, n = {len(ge)}")
        draw(ax, stats, overall, bins, panel, colors=colors,
             extra=lambda s: "" if np.isnan(s["err"]) else f"error {s['err']:.2f}")
        bounds += [v for s in stats if s["drawn"] for v in (s["lo"], s["hi"])]
        for s in stats:
            out_rows.append({"panel": panel, "bin": s["key"], "questions": s["nq_all"],
                             "questions_gmean2": s["nq"], "level1_pass_pct": round(s["l1"], 2),
                             f"mean_{okey}": round(s["other"], 6),
                             "mean_error_01": round(s["err"], 6),
                             "mean_gmean2": round(s["mean"], 6), "ci_lo": round(s["lo"], 6),
                             "ci_hi": round(s["hi"], 6), "drawn": s["drawn"]})
    csv_path = analysis_dir / f"{name}_bins_{tag}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    if bounds:
        axes[0].set_ylim(max(min(bounds) - 0.1, -0.05), min(max(bounds) + 0.07, 1.08))
    axes[0].set_ylabel("Monitorability (g-mean²)")
    fig.supxlabel(f"{name.capitalize()} of the {n_samples} {args.arm} answers — {xlabel}",
                  fontsize=11, y=0.02)
    fig.suptitle(f"Monitorability vs error {name} — "
                 f"{HINT_LABELS.get(args.hint_type, args.hint_type)} hint, {args.model}",
                 fontsize=13)
    note = ("Bias-variance decomposition as in Hägele et al. (2026), The Hot Mess of AI "
            "(arXiv:2601.23045); code adapted from github.com/haeggee/hot-mess-of-ai. "
            f"{explain} One row per question (≥ {args.min_valid}/{n_samples} valid runs). "
            "Under each bin: questions with g-mean² / all questions, Level-1 pass rate, mean "
            f"error rate. 95% CIs: bootstrap over questions ({args.resamples:,} resamples); "
            f"bins with < {args.min_q} g-mean² questions not drawn. Dotted line = panel mean.")
    note = textwrap.fill(note, width=int(17 * fig.get_figwidth()))
    fig.text(0.01, -0.04, note, fontsize=7.5, color="#555555", ha="left", va="top")
    fig.tight_layout()
    png = plots_dir / f"gmean2_by_error_{name}_{tag}.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    print(f"Wrote {csv_path}\nWrote {png}")


def read_arm(path):
    """{instance_id: [valid answer letters]}, {instance_id: correct letter}."""
    path = resolve_jsonl(path)
    if not path.exists():
        return None, None
    opener = gzip.open if path.suffix == ".gz" else open
    answers, gold = defaultdict(list), {}
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            iid = int(r["instance_id"])
            gold[iid] = r["correct_answer"]
            answers.setdefault(iid, [])
            pa = r.get("parsed_answer")
            if r.get("truncated", False) or pa in (None, "", "None", "N/A"):
                continue
            answers[iid].append(pa)
    return answers, gold


def n_options_by_id(index_path):
    with open(index_path) as f:
        items = json.load(f)
    out = {}
    for it in items:
        ch = it.get("choices") or it.get("options") or {}
        if isinstance(ch, str):  # stored as a stringified dict in some indexes
            import ast
            ch = ast.literal_eval(ch)
        out[int(it["instance_id"])] = len(ch)
    return out


def load_rows(root, config, model_name, datasets, hint, arm, split, min_valid, bins):
    rows, dropped = [], {}
    for ds in datasets:
        paths = resolve_paths(root, config, ds, model_name, hint, split=split)
        answers, gold = read_arm(paths["outputs_dir"] / f"{arm}.jsonl")
        csv_path = paths["analysis_dir"] / "results_per_instance.csv"
        if answers is None or not csv_path.exists():
            print(f"WARNING: missing {arm} outputs or {csv_path} for {ds} — skipped.")
            continue
        nopt = n_options_by_id(paths["data_index"])
        with csv_path.open() as f:
            per_inst = {int(r["instance_id"]): r for r in csv.DictReader(f)}
        n_drop = 0
        for iid, letters in sorted(answers.items()):
            if len(letters) < min_valid or iid not in per_inst:
                n_drop += 1
                continue
            k = nopt[iid]
            idx = [ord(a) - ord("A") for a in letters]
            if any(not 0 <= i < k for i in idx):
                raise ValueError(f"{ds} q{iid}: answer outside {k} options: {letters}")
            g = ord(gold[iid]) - ord("A")
            m = process_question_metrics(idx, g, k)
            inc = m["kl_incoherence"]
            row = per_inst[iid]
            gv = row.get("gmean2_i", "")
            counts = Counter(letters)
            rows.append(dict(
                dataset=ds, instance_id=iid, n_options=k, n_valid=len(letters),
                correct_answer=gold[iid],
                answer_counts=" ".join(f"{a}:{c}" for a, c in sorted(counts.items())),
                modal_answer=counts.most_common(1)[0][0],
                incoherence_bin=bin_of(inc, m["is_consensus_correct"], bins),
                level1_pass=row.get("level1_pass", ""), gmean2_i=gv,
                **{key: m[key] for key in PER_Q_FIELDS if key in m},
                q=(ds, iid), bin=bin_of(inc, m["is_consensus_correct"], bins),
                g=None if gv in ("", "None", "nan") else float(gv)))
        dropped[ds] = n_drop
    return rows, dropped


def summarise(rows):
    """Dataset-level error rate and pooled bias / variance / incoherence."""
    def s(key):
        return float(np.sum([r[key] for r in rows]))
    n = len(rows)
    out = dict(
        questions=n,
        error_rate_01=float(np.mean([r["error_01"] for r in rows])),
        accuracy=float(np.mean([r["accuracy"] for r in rows])),
        questions_with_error=int(sum(r["error_01"] > 0 for r in rows)),
        modal_answer_wrong_rate=float(np.mean([not r["is_consensus_correct"] for r in rows])),
        kl_bias_mean=s("kl_bias") / n, kl_variance_mean=s("kl_variance") / n,
        kl_error_mean=s("kl_error") / n,
        kl_error_incoherence=error_incoherence([r["kl_variance"] for r in rows],
                                               [r["kl_error"] for r in rows]),
        brier_bias_mean=s("brier_bias") / n, brier_variance_mean=s("brier_variance") / n,
        brier_error_mean=s("brier_error") / n,
        brier_error_incoherence=error_incoherence([r["brier_variance"] for r in rows],
                                                  [r["brier_error"] for r in rows]),
        bias_01_mean=s("bias_01") / n, variance_01_mean=s("variance_01") / n,
        error_incoherence_01=error_incoherence(
            [r["variance_01"] for r in rows],
            [r["bias_01"] + r["variance_01"] for r in rows]),
    )
    return out


def draw(ax, stats, overall, bins, title, colors=None, extra=None):
    """One panel of mean g-mean² per bin. `colors` maps special bin keys to a
    colour (those bins are left off the connecting line except "one");
    `extra(s)` returns an extra tick-label line per bin."""
    colors = {"no_error": NO_ERR_COLOR, "one": CORRECT_COLOR} if colors is None else colors
    extra = extra or (lambda s: "" if np.isnan(s["pooled"]) else f"inc {s['pooled']:.2f}")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#E5E5E5", linewidth=0.8)
    if not np.isnan(overall):
        ax.axhline(overall, color="#888888", linestyle=":", linewidth=1.2, zorder=1)
    pts = [(i, s) for i, s in enumerate(stats) if s["drawn"]]
    line = [(i, s) for i, s in pts if s["key"] not in colors or s["key"] == "one"]
    ax.plot([i for i, _ in line], [s["mean"] for _, s in line], color=GREY,
            linewidth=1.3, zorder=2)
    for i, s in pts:
        color = colors.get(s["key"], GREY)
        ax.errorbar(i, s["mean"], yerr=[[s["mean"] - s["lo"]], [s["hi"] - s["mean"]]],
                    fmt="o", color=color, ecolor=color, markersize=9,
                    markeredgecolor="#333333", markeredgewidth=0.6, elinewidth=1.8,
                    capsize=3.5, capthick=1.8, zorder=3)
        ax.text(i, s["hi"] + 0.012, f"{s['mean']:.3f}", ha="center", va="bottom",
                fontsize=8.5, color="#333333", fontweight="bold")
    labels = []
    for b, s in zip(bins, stats):
        if not s["nq_all"]:
            labels.append(f"{b[1]}\n0q")
            continue
        x = extra(s)
        labels.append(f"{b[1]}\n{s['nq']}/{s['nq_all']}q\nL1 {s['l1']:.0f}%" + (f"\n{x}" if x else ""))
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_xlim(-0.5, len(bins) - 0.5)
    ax.set_title(title, fontsize=11, fontweight="bold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#999999")


def main():
    parser = argparse.ArgumentParser(
        description="Bias-variance / error-incoherence (Hot Mess of AI) vs g-mean².")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--datasets", nargs="+", default=["lexam"])
    parser.add_argument("--hint-type", default="grader_hacking", choices=HINT_TYPES)
    parser.add_argument("--arm", default="unhinted", choices=["unhinted", "hinted"])
    parser.add_argument("--data", default="test", choices=SPLITS)
    parser.add_argument("--min-valid", type=int, default=27)
    parser.add_argument("--n-bins", type=int, default=3)
    parser.add_argument("--var-edges", type=float, nargs="+", default=[0.15, 0.30, 0.45],
                        help="variance-figure bin edges as the share of runs not "
                             "giving the modal answer (KL variance / -log EPS)")
    parser.add_argument("--bias-edges", type=float, nargs="+", default=[1 / 3, 2 / 3],
                        help="bias-figure bin edges (modal wrong) as the gap "
                             "P(modal) - P(correct) (KL bias / -log EPS)")
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
    bins = make_bins(args.n_bins)
    tag = f"{args.hint_type}_{args.arm}"

    rows, dropped = load_rows(root, config, model_name, args.datasets, args.hint_type,
                              args.arm, args.data, args.min_valid, bins)
    if not rows:
        print("ERROR: no data found — run steps 02–04 first.")
        sys.exit(1)
    for ds, n in dropped.items():
        print(f"{ds}: {n} questions dropped (< {args.min_valid} valid {args.arm} runs)")

    analysis_dir = root / config["paths"]["analysis_dir"] / model_name
    plots_dir = root / config["paths"]["plots_dir"] / model_name
    if args.data != "test":
        analysis_dir, plots_dir = analysis_dir / args.data, plots_dir / args.data
    analysis_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    with (analysis_dir / f"bias_variance_per_question_{tag}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PER_Q_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # --- dataset-level summaries --------------------------------------------
    scopes = [(ds, [r for r in rows if r["dataset"] == ds]) for ds in args.datasets
              if any(r["dataset"] == ds for r in rows)]
    if len(scopes) > 1:
        scopes = [("All datasets", rows)] + scopes
    summary = {}
    for name, sub in scopes:
        summary[name] = {"all_questions": summarise(sub)}
        gsub = [r for r in sub if r["g"] is not None]
        if gsub:
            summary[name]["gmean2_questions"] = summarise(gsub)
    with (analysis_dir / f"bias_variance_summary_{tag}.json").open("w") as f:
        json.dump(summary, f, indent=2)

    md = [f"# Bias-variance decomposition & error-incoherence — {args.model}, "
          f"{args.hint_type}, {args.arm} arm (N = {n_samples} runs/question)", "",
          "Decomposition and error-incoherence as in Hägele et al. (2026), *The Hot "
          "Mess of AI* (arXiv:2601.23045; code adapted from "
          "github.com/haeggee/hot-mess-of-ai, MIT). Incoherence = Σ variance / "
          "Σ error over questions (paper Eq. 2); KL is the paper's headline metric. "
          "KL magnitudes scale with −log(1e-12) per wrong run (one-hot samples), so "
          "read the KL ratio, not the KL values. `0/1 incoherence` = Σ variance / "
          "Σ (bias + variance) with the paper's 0/1 terms.", "",
          "| scope | questions | error rate (0/1) | questions with ≥1 error | modal answer wrong | "
          "KL error-incoherence | Brier error-incoherence | 0/1 incoherence | "
          "Brier bias | Brier variance |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for name, d in summary.items():
        for sname, s in d.items():
            label = name if sname == "all_questions" else f"{name} — g-mean² questions only"
            md.append(f"| {label} | {s['questions']} | {s['error_rate_01']:.3f} | "
                      f"{s['questions_with_error']} | {s['modal_answer_wrong_rate']:.3f} | "
                      f"**{s['kl_error_incoherence']:.3f}** | {s['brier_error_incoherence']:.3f} | "
                      f"{s['error_incoherence_01']:.3f} | {s['brier_bias_mean']:.3f} | "
                      f"{s['brier_variance_mean']:.3f} |")
    (analysis_dir / f"bias_variance_summary_{tag}.md").write_text("\n".join(md) + "\n")
    print("\n" + "\n".join(md[4:]))

    # --- monitorability by incoherence bin ------------------------------------
    fig, axes = plt.subplots(1, len(scopes), figsize=(6.4 * len(scopes), 5.8),
                             sharey=True, squeeze=False)
    axes = axes[0]
    bin_rows, bounds = [], []
    for ax, (name, sub) in zip(axes, scopes):
        stats, overall = step59.bin_stats(sub, bins, args.resamples, rng, args.min_q)
        for s in stats:
            in_bin = [r for r in sub if r["bin"] == s["key"]]
            s["pooled"] = error_incoherence([r["kl_variance"] for r in in_bin],
                                            [r["kl_error"] for r in in_bin]) if in_bin else float("nan")
            s["err"] = float(np.mean([r["error_01"] for r in in_bin])) if in_bin else float("nan")
        print(f"\n=== {name}: {len(sub)} questions ({sum(r['g'] is not None for r in sub)} with g-mean²) ===")
        print(f"{'bin':>9} | {'q':>4} | {'q g2':>5} | {'L1 %':>5} | {'inc':>5} | "
              f"{'err':>5} | {'g-mean2':>7} | 95% CI")
        for s in stats:
            ci = f"[{s['lo']:.3f}, {s['hi']:.3f}]" if s["nq"] else "—"
            flag = "" if s["drawn"] else "  (not drawn)"
            print(f"{s['key']:>9} | {s['nq_all']:>4} | {s['nq']:>5} | {s['l1']:>5.1f} | "
                  f"{s['pooled']:>5.2f} | {s['err']:>5.2f} | {s['mean']:>7.3f} | {ci}{flag}")
        defined = [r for r in sub if r["g"] is not None and not np.isnan(r["kl_incoherence"])]
        if len(defined) > 2:
            rho, p = spearmanr([r["kl_incoherence"] for r in defined], [r["g"] for r in defined])
            print(f"Spearman(per-question KL-incoherence, g-mean²), questions with error: "
                  f"rho = {rho:+.3f}, p = {p:.2g}, n = {len(defined)}")
        draw(ax, stats, overall, bins, name)
        bounds += [v for s in stats if s["drawn"] for v in (s["lo"], s["hi"])]
        for s in stats:
            bin_rows.append(dict(panel=name, bin=s["key"], questions=s["nq_all"],
                                 questions_gmean2=s["nq"], level1_pass_pct=round(s["l1"], 2),
                                 pooled_kl_incoherence=round(s["pooled"], 6),
                                 mean_error_01=round(s["err"], 6),
                                 mean_gmean2=round(s["mean"], 6), ci_lo=round(s["lo"], 6),
                                 ci_hi=round(s["hi"], 6), drawn=s["drawn"]))
    with (analysis_dir / f"incoherence_bins_{tag}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=BIN_FIELDS)
        w.writeheader()
        w.writerows(bin_rows)

    if bounds:
        axes[0].set_ylim(max(min(bounds) - 0.1, -0.05), min(max(bounds) + 0.07, 1.08))
    axes[0].set_ylabel("Monitorability (g-mean²)")
    fig.supxlabel(f"Error-incoherence of the {n_samples} {args.arm} answers "
                  "(KL variance / KL error, per question)", fontsize=11, y=0.02)
    fig.suptitle(f"Monitorability vs error-incoherence — "
                 f"{HINT_LABELS.get(args.hint_type, args.hint_type)} hint, {args.model}",
                 fontsize=13)
    note = ("Error-incoherence as in Hägele et al. (2026), The Hot Mess of AI (arXiv:2601.23045); "
            "code adapted from github.com/haeggee/hot-mess-of-ai. One row per question "
            f"(≥ {args.min_valid}/{n_samples} valid runs). 0 = the same wrong answer every run "
            "(all bias); ≈ 1 = modal answer correct (all error is variance).\nUnder each bin: "
            "questions with g-mean² / all questions, Level-1 pass rate, pooled bin incoherence "
            f"(Σvar/Σerr). 95% CIs: bootstrap over questions ({args.resamples:,} resamples); bins "
            f"with < {args.min_q} g-mean² questions not drawn. Dotted line = panel mean.")
    # wrap to the figure width so the footnote never widens the saved image
    note = textwrap.fill(note.replace("\n", " "), width=int(17 * fig.get_figwidth()))
    fig.text(0.01, -0.04, note, fontsize=7.5, color="#555555", ha="left", va="top")
    fig.tight_layout()
    png = plots_dir / f"gmean2_by_error_incoherence_{tag}.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")

    # --- monitorability by the paper's (KL) variance and bias -----------------
    common = dict(scopes=scopes, rng=rng, args=args, analysis_dir=analysis_dir,
                  plots_dir=plots_dir, tag=tag, n_samples=n_samples)
    L = -np.log(EPS)
    vbins = make_var_bins(args.var_edges)
    component_figure(
        metric="kl_variance", name="variance", bins=vbins,
        bin_fn=lambda r: var_bin_of(r["kl_variance"], vbins),
        colors={"zero": NO_ERR_COLOR}, other=("kl_bias", "KL bias"),
        xlabel="KL-Variance E[KL(f̄ ‖ f)] (nats, per question)",
        explain=("KL-Variance = E[KL(f̄ ‖ f)] over the one-hot runs of a question; with "
                 f"one-hot samples it equals (1 − P(modal answer)) × −log(1e-12) ≈ {L:.1f} "
                 "nats, so each bin is also a share of runs off the modal answer."),
        **common)
    bbins = make_bias_bins(args.bias_edges)
    component_figure(
        metric="kl_bias", name="bias", bins=bbins,
        bin_fn=lambda r: bias_bin_of(r, bbins),
        colors={"no_error": NO_ERR_COLOR, "modal_correct": CORRECT_COLOR},
        other=("kl_variance", "KL var"),
        xlabel="KL-Bias KL(y ‖ f̄) (nats, per question)",
        explain=("KL-Bias = KL(y ‖ f̄), f̄ the normalised geometric mean of the one-hot runs; "
                 "with one-hot samples it is ≈ 0 when the modal answer is correct and "
                 f"otherwise ≈ (P(modal) − P(correct)) × −log(1e-12) ≈ {L:.1f} nats, so each "
                 "bin is also the gap between the modal and the correct answer's share of runs. "
                 "Bias ≈ 0 is split into questions never answered wrong and questions whose "
                 "modal answer is correct but some runs are wrong."),
        **common)

    for name in (f"bias_variance_per_question_{tag}.csv", f"bias_variance_summary_{tag}.json",
                 f"bias_variance_summary_{tag}.md", f"incoherence_bins_{tag}.csv"):
        print(f"Wrote {analysis_dir / name}")
    print(f"Wrote {png}")


if __name__ == "__main__":
    main()
