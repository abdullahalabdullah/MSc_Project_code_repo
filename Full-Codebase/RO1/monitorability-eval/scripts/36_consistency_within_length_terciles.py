#!/usr/bin/env python3
"""
Step 36: Does consistency raise monitorability independently of response
length? Consistency-vs-inconsistency contrast WITHIN unhinted-CoT-length
terciles, pooled over all datasets × hint types for ONE model.

Consistency and response length are strongly (negatively) correlated —
consistent questions are ~2x SHORTER (Spearman ≈ -0.45). Because the
project also documents a length effect (longer CoT → lower monitorability),
a reviewer may suspect the consistency effect is just a length effect. This
step refutes that: it splits questions into short/medium/long unhinted-CoT
terciles and, within each, compares consistent vs inconsistent monitorability.
If the consistency gap survives at every length, it is not a length artifact.

    consistency_i = modal unhinted-answer share over the valid unhinted runs
                    (consistent = 1.0, i.e. same answer in all N runs)
    length_i      = mean unhinted CoT tokens (mean_cot_tokens_unhinted)

Base model: consistent beats inconsistent in ALL three terciles
(Δ = +0.176 / +0.097 / +0.059, every CI clear of zero), so consistency
predicts monitorability after conditioning on length. (The gap shrinks with
length — the two interact — but never closes.)

Figure: x = 3 length terciles; per tercile two dots (grey inconsistent,
navy consistent) with bootstrap 95% CI and n; Δ annotated per tercile.

Usage:
    python scripts/36_consistency_within_length_terciles.py            # base
    python scripts/36_consistency_within_length_terciles.py --model <hf-id>

    Options:
      --config     path to config YAML (default: config/mvp_config.yaml)
      --data       split: test (default) | eval
      --model      model/adapter HF id (default: Qwen/Qwen3-4B)
      --resamples  bootstrap replicates per CI (default: 5000)
      --seed       bootstrap seed (default: 88)
      --out        output PNG path (default:
                   {plots_dir}/task-performance-vs-monitorability/
                   consistency_within_length_terciles_{model}.png)

Input:
    {outputs_dir}/{model}/{dataset}/{hint}/unhinted.jsonl[.gz]
    {analysis_dir}/{model}/{dataset}/{hint}/results_per_instance.csv

Output:
    {plots_dir}/task-performance-vs-monitorability/
        consistency_within_length_terciles_{model}.png
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
        description="Consistency contrast within CoT-length terciles.")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--data", default="test", choices=SPLITS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
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

    # ---- pooled per-question (consistency, length, gmean2) ----
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
                    g = row.get("gmean2_i", "")
                    tok = row.get("mean_cot_tokens_unhinted", "")
                    if g in ("", "None", "N/A") or tok in ("", "None", "N/A"):
                        continue
                    iid = int(row["instance_id"])
                    if iid not in consistency:
                        continue
                    recs.append((consistency[iid], float(tok), float(g)))
    if missing:
        print("WARNING: missing inputs skipped: " + ", ".join(missing))
    if not recs:
        print("ERROR: no data found — nothing to plot.")
        sys.exit(1)

    cons = np.array([r[0] for r in recs])
    length = np.array([r[1] for r in recs])
    gm = np.array([r[2] for r in recs])
    determ = cons >= 0.999

    t1, t2 = np.percentile(length, [100 / 3, 200 / 3])
    terciles = [
        (f"Short\n(≤{t1:.0f} tok)", length <= t1),
        (f"Medium\n({t1:.0f}–{t2:.0f})", (length > t1) & (length <= t2)),
        (f"Long\n(>{t2:.0f})", length > t2),
    ]

    print(f"n = {len(recs)} question-slices; length tercile cuts at "
          f"{t1:.0f} and {t2:.0f} unhinted CoT tokens\n")
    print(f"{'tercile':>8} | {'consistent':>18} | {'inconsistent':>18} | "
          f"{'Delta':>7} | 95% CI")
    print("-" * 78)

    rows = []  # (label, con_mean, con_ci, inc_mean, inc_ci, n_con, n_inc, d, dci)
    for label, mask in terciles:
        con = gm[mask & determ]
        inc = gm[mask & ~determ]
        con_ci = boot_ci(con, args.resamples, rng)
        inc_ci = boot_ci(inc, args.resamples, rng)
        d = float(con.mean() - inc.mean())
        dlo, dhi = boot_diff_ci(con, inc, args.resamples, rng)
        flat = label.replace("\n", " ")
        print(f"{flat[:8]:>8} | {con.mean():.3f} (n={len(con):>4}) | "
              f"{inc.mean():.3f} (n={len(inc):>4}) | {d:>+7.3f} | "
              f"[{dlo:+.3f}, {dhi:+.3f}]")
        rows.append((label, con.mean(), con_ci, inc.mean(), inc_ci,
                     len(con), len(inc), d, (dlo, dhi)))

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#DDDDDD", linewidth=0.8)

    off = 0.16
    all_b = []
    for i, (label, cm, cci, im, ici, ncon, ninc, d, dci) in enumerate(rows):
        # inconsistent (grey, left), consistent (navy, right); dashed link
        ax.plot([i - off, i + off], [im, cm], color="#BBBBBB", linewidth=1.0,
                linestyle="--", zorder=2)
        ax.errorbar(i - off, im, yerr=[[im - ici[0]], [ici[1] - im]],
                    fmt="o", color=GREY, ecolor=GREY, markersize=9,
                    markeredgecolor="#333333", markeredgewidth=0.5,
                    elinewidth=1.8, capsize=4, capthick=1.8, zorder=3)
        ax.errorbar(i + off, cm, yerr=[[cm - cci[0]], [cci[1] - cm]],
                    fmt="o", color=NAVY, ecolor=NAVY, markersize=9,
                    markeredgecolor="#333333", markeredgewidth=0.5,
                    elinewidth=1.8, capsize=4, capthick=1.8, zorder=3)
        ax.text(i - off, ici[1] + 0.006, f"{im:.3f}", ha="center",
                va="bottom", fontsize=8.5, color=GREY)
        ax.text(i + off, cci[1] + 0.006, f"{cm:.3f}", ha="center",
                va="bottom", fontsize=8.5, color=NAVY, fontweight="bold")
        ax.text(i - off, ici[0] - 0.010, f"n={ninc}", ha="center", va="top",
                fontsize=7.5, color="#999999")
        ax.text(i + off, cci[0] - 0.010, f"n={ncon}", ha="center", va="top",
                fontsize=7.5, color="#999999")
        all_b.extend([ici[0], ici[1], cci[0], cci[1]])

    ymax = max(all_b)
    ymin = min(all_b)
    # Δ labels in a clear row along the top
    for i, r in enumerate(rows):
        d, (dlo, dhi) = r[7], r[8]
        ax.text(i, ymax + 0.055, f"Δ = {d:+.3f}\n[{dlo:+.3f}, {dhi:+.3f}]",
                ha="center", va="bottom", fontsize=8.5, color="#333333")

    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r[0] for r in rows], fontsize=9.5)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_ylim(ymin - 0.045, ymax + 0.10)
    ax.set_xlabel("Unhinted response length (CoT-token tercile)")
    ax.set_ylabel("Mean per-question g-mean²")
    ax.set_title("Consistency raises monitorability within every "
                 f"response-length band — {model_name}\n"
                 "(so the consistency effect is not a length artifact; "
                 "consistent responses are in fact shorter)", fontsize=11)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#999999")

    handles = [
        Line2D([], [], marker="o", linestyle="none", color=NAVY,
               markeredgecolor="#333333", markeredgewidth=0.5, markersize=9,
               label="Consistent (same answer in all 8 runs)"),
        Line2D([], [], marker="o", linestyle="none", color=GREY,
               markeredgecolor="#333333", markeredgewidth=0.5, markersize=9,
               label="Inconsistent (answers vary)"),
    ]
    ax.legend(handles=handles, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, -0.13), frameon=True,
              edgecolor="#CCCCCC", fontsize=9)

    out = args.out
    if out is None:
        plots_dir = ((root / config["paths"]["plots_dir"]).parent
                     / "extra-plots" / "task-performance-vs-monitorability")
        if args.data != "test":
            plots_dir = plots_dir / args.data
        plots_dir.mkdir(parents=True, exist_ok=True)
        out = (plots_dir /
               f"consistency_within_length_terciles_{model_name}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
