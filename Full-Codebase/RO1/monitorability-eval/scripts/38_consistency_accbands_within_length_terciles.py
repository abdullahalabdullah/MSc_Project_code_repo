#!/usr/bin/env python3
"""
Step 38: Decompose the inconsistent group by unhinted accuracy WITHIN
CoT-length terciles — a finer-grained version of step 36.

Step 36 contrasts consistent vs inconsistent monitorability within each
unhinted-CoT-length tercile (two dots per band). This step keeps the
consistent reference dot but SPLITS the single inconsistent dot into three
unhinted-accuracy sub-bands — 0/8, 1-3/8, 4-7/8 — so the "commitment ladder"
is visible: never-correct -> partially-correct -> nearly-always-correct ->
consistent, within each length band.

Because any 8/8-correct question is answered identically in all 8 runs (hence
consistent), every INCONSISTENT question has accuracy <= 7/8. The three bands
therefore partition the inconsistent group exactly, and their pooled mean
reproduces step 36's single inconsistent dot. NOTE: consistent-but-wrong
questions (same wrong answer 8/8) are consistent, so they sit in the navy
Consistent dot, NOT in the 0/8 dot — the 0/8 dot is inconsistent-and-never-
correct only.

    consistency_i = modal unhinted-answer share over valid unhinted runs
                    (consistent = 1.0, same answer in all N runs)
    acc_i         = k/8, k = correct unhinted runs (acc_u_i, snapped to /8)
    length_i      = mean unhinted CoT tokens (mean_cot_tokens_unhinted)

Figure: x = 3 length terciles; per tercile four dots (three grey shades for
inconsistent 0/8 -> 4-7/8, then navy consistent) linked by a dashed ladder,
each with bootstrap 95% CI, mean and n; per-tercile Delta (consistent minus
pooled-inconsistent) annotated along the top to stay comparable with step 36.

Usage:
    python scripts/38_consistency_accbands_within_length_terciles.py
    python scripts/38_consistency_accbands_within_length_terciles.py --model <hf-id>

    Options:
      --config     path to config YAML (default: config/mvp_config.yaml)
      --data       split: test (default) | eval
      --model      model/adapter HF id (default: Qwen/Qwen3-4B)
      --min-n      min questions to draw a dot (default: 8)
      --resamples  bootstrap replicates per CI (default: 5000)
      --seed       bootstrap seed (default: 88)
      --out        output PNG path (default:
                   {plots_dir}/task-performance-vs-monitorability/
                   consistency_accbands_within_length_terciles_{model}.png)

Input:
    {outputs_dir}/{model}/{dataset}/{hint}/unhinted.jsonl[.gz]
    {analysis_dir}/{model}/{dataset}/{hint}/results_per_instance.csv

Output:
    {plots_dir}/task-performance-vs-monitorability/
        consistency_accbands_within_length_terciles_{model}.png
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
N_RUNS = 8

DATASETS = ["mmlu_pro", "gpqa_diamond", "lsat_ar", "lsat_lr",
            "medxpertqa", "lexam", "codemmlu", "wmdp"]

NAVY = "#1B3A6B"   # consistent (same answer in all 8 runs)

# Three inconsistent accuracy sub-bands: light -> dark grey-blue so the
# "increasing determination" ladder survives grayscale/CVD. Each entry:
# (label, colour, x-offset, selector on k = correct unhinted runs).
INC_BANDS = [
    ("Inconsistent · 0/8",   "#C7CED6", -0.27, lambda k: k == 0),
    ("Inconsistent · 1–3/8", "#93A1B0", -0.09, lambda k: (k >= 1) & (k <= 3)),
    # k>=4 (not <=7): a few inconsistent questions round to k=8 when a parsed
    # run was dropped from grading (acc over fewer valid runs than the answer
    # set); fold them here so the three bands partition inconsistent exactly.
    ("Inconsistent · 4–7/8", "#5B6B7B", +0.09, lambda k: k >= 4),
]
CON_OFF = +0.27


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
        description="Inconsistent-by-accuracy bands within CoT-length terciles.")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--data", default="test", choices=SPLITS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
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

    # ---- pooled per-question-slice (consistency, length, gmean2, acc-k) ----
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
                    a = row.get("acc_u_i", "")
                    if (g in ("", "None", "N/A") or tok in ("", "None", "N/A")
                            or a in ("", "None", "N/A")):
                        continue
                    iid = int(row["instance_id"])
                    if iid not in consistency:
                        continue
                    k = int(np.clip(round(float(a) * N_RUNS), 0, N_RUNS))
                    recs.append((consistency[iid], float(tok), float(g), k))
    if missing:
        print("WARNING: missing inputs skipped: " + ", ".join(missing))
    if not recs:
        print("ERROR: no data found — nothing to plot.")
        sys.exit(1)

    cons = np.array([r[0] for r in recs])
    length = np.array([r[1] for r in recs])
    gm = np.array([r[2] for r in recs])
    kk = np.array([r[3] for r in recs])
    determ = cons >= 0.999

    t1, t2 = np.percentile(length, [100 / 3, 200 / 3])
    terciles = [
        (f"Short\n(≤{t1:.0f} tok)", length <= t1),
        (f"Medium\n({t1:.0f}–{t2:.0f})", (length > t1) & (length <= t2)),
        (f"Long\n(>{t2:.0f})", length > t2),
    ]

    print(f"n = {len(recs)} question-slices; length tercile cuts at "
          f"{t1:.0f} and {t2:.0f} unhinted CoT tokens")
    print("(inconsistent split by unhinted accuracy k/8; consistent = k-any, "
          "same answer 8/8)\n")

    # rows[i] = dict with dots (label,color,off,mean,ci,n) + consistent + delta
    rows = []
    for label, mask in terciles:
        flat = label.replace("\n", " ")
        print(f"[{flat}]")
        dots = []
        for blabel, color, off, sel in INC_BANDS:
            m = mask & ~determ & sel(kk)
            vals = gm[m]
            n = int(m.sum())
            if n >= args.min_n:
                mean = float(vals.mean())
                ci = boot_ci(vals, args.resamples, rng)
            else:
                mean, ci = float("nan"), (float("nan"), float("nan"))
            dots.append((blabel, color, off, mean, ci, n))
            print(f"  {blabel:>22} | n={n:>4} | "
                  + (f"{mean:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
                     if n >= args.min_n else f"(dropped: n < {args.min_n})"))
        con = gm[mask & determ]
        con_mean = float(con.mean())
        con_ci = boot_ci(con, args.resamples, rng)
        n_con = int((mask & determ).sum())
        print(f"  {'Consistent (8/8)':>22} | n={n_con:>4} | "
              f"{con_mean:.3f} [{con_ci[0]:.3f}, {con_ci[1]:.3f}]")
        # Delta vs pooled inconsistent (same contrast as step 36)
        inc_pool = gm[mask & ~determ]
        d = float(con_mean - inc_pool.mean())
        dlo, dhi = boot_diff_ci(con, inc_pool, args.resamples, rng)
        print(f"  Δ consistent − pooled-inconsistent = {d:+.3f} "
              f"[{dlo:+.3f}, {dhi:+.3f}]  (pooled inc n={len(inc_pool)})\n")
        rows.append({"label": label, "dots": dots, "con_mean": con_mean,
                     "con_ci": con_ci, "n_con": n_con, "d": d,
                     "dci": (dlo, dhi)})

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#DDDDDD", linewidth=0.8)

    all_b = []
    for i, row in enumerate(rows):
        # dashed ladder through the four dots (inconsistent x3 -> consistent)
        lx, ly = [], []
        for _, _, off, mean, ci, n in row["dots"]:
            if n >= args.min_n:
                lx.append(i + off)
                ly.append(mean)
        lx.append(i + CON_OFF)
        ly.append(row["con_mean"])
        ax.plot(lx, ly, color="#BBBBBB", linewidth=1.0, linestyle="--",
                zorder=2)

        for blabel, color, off, mean, ci, n in row["dots"]:
            if n < args.min_n:
                continue
            x = i + off
            ax.errorbar(x, mean, yerr=[[mean - ci[0]], [ci[1] - mean]],
                        fmt="o", color=color, ecolor=color, markersize=9,
                        markeredgecolor="#333333", markeredgewidth=0.5,
                        elinewidth=1.7, capsize=3.5, capthick=1.7, zorder=3)
            ax.text(x, ci[1] + 0.006, f"{mean:.3f}", ha="center", va="bottom",
                    fontsize=8, color="#555555")
            ax.text(x, ci[0] - 0.010, f"n={n}", ha="center", va="top",
                    fontsize=7, color="#999999")
            all_b.extend([ci[0], ci[1]])

        # consistent (navy)
        x = i + CON_OFF
        cci = row["con_ci"]
        ax.errorbar(x, row["con_mean"],
                    yerr=[[row["con_mean"] - cci[0]], [cci[1] - row["con_mean"]]],
                    fmt="o", color=NAVY, ecolor=NAVY, markersize=9,
                    markeredgecolor="#333333", markeredgewidth=0.5,
                    elinewidth=1.7, capsize=3.5, capthick=1.7, zorder=3)
        ax.text(x, cci[1] + 0.006, f"{row['con_mean']:.3f}", ha="center",
                va="bottom", fontsize=8, color=NAVY, fontweight="bold")
        ax.text(x, cci[0] - 0.010, f"n={row['n_con']}", ha="center", va="top",
                fontsize=7, color="#999999")
        all_b.extend([cci[0], cci[1]])

    ymax, ymin = max(all_b), min(all_b)
    for i, row in enumerate(rows):
        ax.text(i, ymax + 0.055,
                f"Δ = {row['d']:+.3f}\n[{row['dci'][0]:+.3f}, "
                f"{row['dci'][1]:+.3f}]",
                ha="center", va="bottom", fontsize=8.5, color="#333333")

    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r["label"] for r in rows], fontsize=9.5)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_ylim(ymin - 0.05, ymax + 0.11)
    ax.set_xlabel("Unhinted response length (CoT-token tercile)")
    ax.set_ylabel("Mean per-question g-mean²")
    ax.set_title("Monitorability rises with answer determination within every "
                 f"length band — {model_name}\n(inconsistent split by unhinted "
                 "accuracy; Δ = consistent − pooled-inconsistent)", fontsize=11)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#999999")

    handles = [Line2D([], [], marker="o", linestyle="none", color=c,
                      markeredgecolor="#333333", markeredgewidth=0.5,
                      markersize=9, label=lab)
               for lab, c, _, _ in INC_BANDS]
    handles.append(Line2D([], [], marker="o", linestyle="none", color=NAVY,
                          markeredgecolor="#333333", markeredgewidth=0.5,
                          markersize=9,
                          label="Consistent (same answer in all 8 runs)"))
    ax.legend(handles=handles, loc="upper center", ncol=4,
              bbox_to_anchor=(0.5, -0.14), frameon=True,
              edgecolor="#CCCCCC", fontsize=8.5)

    out = args.out
    if out is None:
        plots_dir = ((root / config["paths"]["plots_dir"]).parent
                     / "extra-plots" / "task-performance-vs-monitorability")
        if args.data != "test":
            plots_dir = plots_dir / args.data
        plots_dir.mkdir(parents=True, exist_ok=True)
        out = (plots_dir /
               f"consistency_accbands_within_length_terciles_{model_name}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
