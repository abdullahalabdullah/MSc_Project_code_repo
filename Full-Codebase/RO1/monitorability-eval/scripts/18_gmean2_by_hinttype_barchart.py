#!/usr/bin/env python3
"""
Step 18: Grouped bar chart of g-mean^2 per HINT TYPE, averaged over all
datasets — base vs anti-sycophancy DPO vs harmlessness DPO.

Companion to 16/17: here the x-axis is the three hint types (professor,
grader_hacking, unethical); each hint type has three columns (base,
anti-sycophancy DPO, harmlessness DPO), and each column is the unweighted mean
over the 8 datasets of that dataset's per-hint g-mean^2 (mean over its defined
instances) — datasets weighted equally, never instance-pooled, mirroring the
pooling rule of 15_pooled_summary_tables.py with the roles of hint type and
dataset swapped.

CI — paired cluster bootstrap over instances (95%). Each replicate draws, for
every dataset, n_d instance ids with replacement from that dataset's shared
pool, and reuses the SAME dataset-wise draws for every hint type AND every
model (all models/hints run on the identical prepared pools, so this pairing
propagates the shared-question correlation, exactly as in steps 14/15).
Per replicate: per-dataset mean over the drawn ids defined for that
(model, dataset, hint), then the mean over datasets. 10,000 resamples,
seed 88, 2.5/97.5 percentiles. Datasets and hint types are treated as fixed.

Each model has its own color (base light blue, harmlessness DPO light pink,
anti-sycophancy DPO light green — matching 17_pooled_gmean2_barchart.py) plus
the same model hatches as steps 16/17 (base solid, harmlessness DPO striped,
anti-sycophancy DPO dotted), so the figure survives grayscale.

Usage:
    python scripts/18_gmean2_by_hinttype_barchart.py

    Options:
      --config    path to config YAML (default: config/mvp_config.yaml)
      --data      split: test (default) | eval
      --resamples bootstrap replicates (default: gmean2.bootstrap_resamples or 10000)
      --seed      bootstrap seed (default: 88)
      --base      base model id (default: Qwen/Qwen3-4B)
      --harmless  harmlessness DPO id (default: er-abd/harmless-dpo-qwen3-4B-step-735)
      --antisyc   anti-sycophancy DPO id (default: er-abd/anti-syc-dpo-qwen3-4B-step-520)
      --out       output PNG path (default: {plots_dir}/gmean2_by_hinttype_3models.png)

Input (per model / dataset / hint type, produced by 04_compute_gmean2.py):
    {analysis_dir}/{model}/{dataset}/{hint}/results_per_instance.csv

Output:
    {plots_dir}/gmean2_by_hinttype_3models.png
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import yaml

from src.dataset import SPLITS, model_dir_name, resolve_paths
from src.hints import HINT_TYPES

DEFAULT_RESAMPLES = 10000
DEFAULT_SEED = 88

DEFAULT_BASE = "Qwen/Qwen3-4B"
DEFAULT_HARMLESS = "er-abd/harmless-dpo-qwen3-4B-step-735"
DEFAULT_ANTISYC = "er-abd/anti-syc-dpo-qwen3-4B-step-520"

DATASETS = ["mmlu_pro", "gpqa_diamond", "lsat_ar", "lsat_lr",
            "medxpertqa", "lexam", "codemmlu", "wmdp"]

HINT_LABELS = {
    "professor": "Professor hint",
    "grader_hacking": "Grader-hacking hint",
    "unethical": "Unethical hint",
}

# Model colors: the same trio as 16_gmean2_barchart.py (there bound to hint
# types), here bound to models — matching 17_pooled_gmean2_barchart.py.
COLOR_BASE = "#92C5E8"      # light blue
COLOR_HARMLESS = "#F6A0B8"  # light pink
COLOR_ANTISYC = "#9ED973"   # light green

EDGE = "#555555"
BAR_W = 0.22
MODEL_OFFSETS = [-0.25, 0.0, 0.25]


def load_gmean2_map(analysis_dir: Path):
    """instance_id -> gmean2_i for defined instances, plus the full pool."""
    per_instance = analysis_dir / "results_per_instance.csv"
    if not per_instance.exists():
        return None
    defined, pool = {}, []
    with per_instance.open() as f:
        for row in csv.DictReader(f):
            iid = int(row["instance_id"])
            pool.append(iid)
            raw = row.get("gmean2_i", "")
            if raw not in ("", "None", "N/A"):
                defined[iid] = float(raw)
    return {"defined": defined, "pool": pool}


def main():
    parser = argparse.ArgumentParser(
        description="Bar chart of dataset-averaged g-mean^2 per hint type "
                    "with paired-bootstrap CIs.")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--data", default="test", choices=SPLITS)
    parser.add_argument("--resamples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--harmless", default=DEFAULT_HARMLESS)
    parser.add_argument("--antisyc", default=DEFAULT_ANTISYC)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    with config_path.open() as f:
        config = yaml.safe_load(f)
    n_resamples = args.resamples or config.get("gmean2", {}).get(
        "bootstrap_resamples", DEFAULT_RESAMPLES)

    # Bar order within each hint-type set, as presented; hatches stay bound to
    # the model (consistent with 16/17) regardless of order.
    models = [
        (model_dir_name(args.base), "Base (Qwen3-4B)", None, COLOR_BASE),
        (model_dir_name(args.antisyc), "Anti-sycophancy DPO", "..",
         COLOR_ANTISYC),
        (model_dir_name(args.harmless), "Harmlessness DPO", "///",
         COLOR_HARMLESS),
    ]

    # loaded[dataset][(model, hint)] = {"defined": ..., "pool": ...}
    loaded = {ds: {} for ds in DATASETS}
    for ds in DATASETS:
        for model_name, _, _, _ in models:
            for hint in HINT_TYPES:
                paths = resolve_paths(root, config, ds, model_name, hint,
                                      split=args.data)
                entry = load_gmean2_map(paths["analysis_dir"])
                if entry is None:
                    print(f"ERROR: missing results_per_instance.csv for "
                          f"{model_name}/{ds}/{hint}")
                    sys.exit(1)
                loaded[ds][(model_name, hint)] = entry

    # Per dataset: one shared union pool (all models/hints run on the same
    # prepared instances) and one NaN-padded value vector per (model, hint).
    combos = [(m, h) for m, _, _, _ in models for h in HINT_TYPES]
    pools, vecs = {}, {ds: {} for ds in DATASETS}
    for ds in DATASETS:
        union = sorted(set().union(*(set(e["pool"])
                                     for e in loaded[ds].values())))
        pools[ds] = union
        index = {iid: j for j, iid in enumerate(union)}
        for combo in combos:
            v = np.full(len(union), np.nan)
            for iid, val in loaded[ds][combo]["defined"].items():
                v[index[iid]] = val
            vecs[ds][combo] = v

    # Point estimates: per-dataset mean over defined instances, then the
    # unweighted mean over datasets.
    point = {c: float(np.mean([np.nanmean(vecs[ds][c]) for ds in DATASETS]))
             for c in combos}

    # Paired bootstrap: per dataset one index matrix (n_resamples x n_d),
    # reused across every model and hint type.
    rng = np.random.default_rng(args.seed)
    reps = {c: np.zeros((n_resamples, len(DATASETS))) for c in combos}
    with np.errstate(invalid="ignore"):
        for d_idx, ds in enumerate(DATASETS):
            n_d = len(pools[ds])
            idx = rng.integers(0, n_d, size=(n_resamples, n_d))
            for c in combos:
                reps[c][:, d_idx] = np.nanmean(vecs[ds][c][idx], axis=1)
    ci = {}
    for c in combos:
        per_rep = np.nanmean(reps[c], axis=1)
        ci[c] = [float(v) for v in np.percentile(per_rep, [2.5, 97.5])]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#DDDDDD", linewidth=0.8)
    ax.set_ylim(0, 1)
    ax.set_yticks([i / 10 for i in range(11)])

    for h_idx, hint in enumerate(HINT_TYPES):
        for (model_name, _, hatch, color), off in zip(models, MODEL_OFFSETS):
            c = (model_name, hint)
            est, (lo, hi) = point[c], ci[c]
            x = h_idx + off
            ax.bar(x, est, width=BAR_W, color=color, hatch=hatch,
                   edgecolor=EDGE, linewidth=0.6, zorder=3)
            ax.errorbar(x, est, yerr=[[est - lo], [hi - est]], fmt="none",
                        ecolor="#333333", elinewidth=1.0, capsize=3,
                        capthick=1.0, zorder=4)
            ax.text(x, hi + 0.012, f"{est:.2f}", ha="center", va="bottom",
                    fontsize=8, color="black", zorder=5, clip_on=False)

    # Reference line: unweighted mean of every bar drawn in this figure.
    avg = float(np.mean(list(point.values())))
    ax.axhline(avg, color="red", linestyle="--", linewidth=1.2, zorder=4)

    ax.set_xticks(range(len(HINT_TYPES)))
    ax.set_xticklabels([HINT_LABELS[h] for h in HINT_TYPES])
    ax.set_xlim(-0.55, len(HINT_TYPES) - 0.45)
    ax.set_xlabel("Hint type")
    ax.set_ylabel("g-mean² (mean over 8 datasets)")
    ax.set_title("g-mean² per hint type, averaged over datasets "
                 "(paired-bootstrap 95% CI)", pad=22)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#999999")

    handles = ([Patch(facecolor=color, edgecolor=EDGE, linewidth=0.8,
                      hatch=hatch, label=lbl) for _, lbl, hatch, color in models]
               + [Line2D([], [], color="red", linestyle="--", linewidth=1.2,
                         label=f"Average ({avg * 100:.1f}%)")])
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.005, 1.0),
              frameon=True, edgecolor="#CCCCCC", fontsize=9)

    out = args.out
    if out is None:
        # RO1 bespoke analysis plots live under extra-plots/, not the main plots/ tree
        plots_dir = ((root / config["paths"]["plots_dir"]).parent
                     / "extra-plots" / "cross-model-comparison")
        if args.data != "test":
            plots_dir = plots_dir / args.data
        plots_dir.mkdir(parents=True, exist_ok=True)
        out = plots_dir / "gmean2_by_hinttype_3models.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
