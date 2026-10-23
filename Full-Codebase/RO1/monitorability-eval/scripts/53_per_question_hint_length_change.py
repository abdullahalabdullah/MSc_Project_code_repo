#!/usr/bin/env python3
"""
Step 53: Per-question change in response length when the hint is introduced —
one figure per dataset, for a single model (default: base Qwen3-4B).

The paired, per-question view of the hint effect on length that the median
bar charts (step 46) cannot show. For each question:

    ratio = median hinted length / median unhinted length

over that question's runs (8 hinted runs per hint type, 8 unhinted runs; the
two arms are paired by instance_id). ratio > 1 = the hint made that question's
responses longer. Ratios are plotted on a log2 axis so ×2 and ×0.5 sit
symmetrically around the no-change line at 1.

Each dataset figure has four columns: the three hint types separately
(professor / grader-hacking / unethical, 200 questions each) and "All hints
pooled" (per question, the median over all 24 hinted runs vs the same 8
unhinted runs). Each column = violin (distribution shape) + box (median and
IQR, whiskers 5th–95th percentile) + one jittered dot per question. Printed
above each column: the median per-question % change and the share of
questions that got longer.

Lengths come from step 08's length_pools and INCLUDE runs truncated at the
16384-token cap (as in steps 44–46).

Usage:
    python scripts/53_per_question_hint_length_change.py

    Options:
      --config    path to config YAML (default: config/mvp_config.yaml)
      --data      split: test (default) | eval
      --model     model id (default: Qwen/Qwen3-4B)
      --out-dir   output directory (default: {plots_dir}/../extra-plots/
                  hint-effect-on-length/{model})

Input (per dataset x hint, produced by 08_cot_token_analysis.py):
    {analysis_dir}/{model}/{dataset}/{hint}/cot_token_stats.json

Output (one per dataset):
    {out_dir}/length_change_per_question_{dataset}_{model}.png
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.dataset import SPLITS, model_dir_name, resolve_paths

DEFAULT_MODEL = "Qwen/Qwen3-4B"

DATASETS = ["mmlu_pro", "gpqa_diamond", "lsat_ar", "lsat_lr",
            "medxpertqa", "lexam", "codemmlu", "wmdp"]

# Same hint colors as steps 16/46; pooled column in plum (the hinted-arm hue
# of steps 37/41/46).
COLUMNS = [
    ("professor", "Professor", "#92C5E8"),
    ("grader_hacking", "Grader-hacking", "#F6A0B8"),
    ("unethical", "Unethical", "#9ED973"),
    (None, "All hints pooled", "#C7ADE6"),
]
EDGE = "#555555"


def runs_by_question(arm):
    """{instance_id: [token_count, ...]} for one arm's length pool."""
    out = defaultdict(list)
    for iid, n in zip(arm["instance_id"], arm["token_count"]):
        out[iid].append(n)
    return out


def load_dataset(root, config, model_name, ds, split):
    """Returns (hinted_by_hint, unhinted) as per-question run lists, or None
    if any hint cell is missing."""
    hinted, unhinted = {}, None
    for hint, _, _ in COLUMNS[:-1]:
        paths = resolve_paths(root, config, ds, model_name, hint, split=split)
        jp = paths["analysis_dir"] / "cot_token_stats.json"
        if not jp.exists():
            print(f"WARNING: missing {jp}")
            return None
        lp = json.load(jp.open()).get("length_pools")
        if lp is None:
            print(f"WARNING: no length_pools in {jp} — re-run step 08")
            return None
        hinted[hint] = runs_by_question(lp["hinted"])
        if unhinted is None:  # the one shared unhinted arm
            unhinted = runs_by_question(lp["unhinted"])
    return hinted, unhinted


def per_question_ratios(hinted, unhinted):
    """{column label: np.array of per-question hinted/unhinted median
    ratios}, questions paired by instance_id."""
    ratios = {}
    for hint, label, _ in COLUMNS:
        r = []
        for iid, u_runs in unhinted.items():
            if hint is None:
                h_runs = [n for h in hinted for n in hinted[h].get(iid, [])]
            else:
                h_runs = hinted[hint].get(iid, [])
            u_med = np.median(u_runs)
            if h_runs and u_med > 0:
                r.append(np.median(h_runs) / u_med)
        ratios[label] = np.array(r)
    return ratios


def plot_dataset(ds, ratios, model_label, out):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    rng = np.random.default_rng(0)
    log_data = [np.log2(ratios[label]) for _, label, _ in COLUMNS]
    pos = np.arange(len(COLUMNS))

    vp = ax.violinplot(log_data, positions=pos, widths=0.8,
                       showextrema=False)
    for body, (_, _, color) in zip(vp["bodies"], COLUMNS):
        body.set_facecolor(color)
        body.set_edgecolor(EDGE)
        body.set_linewidth(0.6)
        body.set_alpha(0.45)

    for x, vals, (_, _, color) in zip(pos, log_data, COLUMNS):
        ax.scatter(x + rng.uniform(-0.14, 0.14, len(vals)), vals, s=6,
                   color=color, edgecolor=EDGE, linewidth=0.25, alpha=0.8,
                   zorder=3)

    ax.boxplot(log_data, positions=pos, widths=0.22, whis=(5, 95),
               showfliers=False, patch_artist=True, zorder=4,
               boxprops=dict(facecolor="white", edgecolor="black",
                             linewidth=0.9, alpha=0.85),
               medianprops=dict(color="black", linewidth=1.6),
               whiskerprops=dict(color="black", linewidth=0.8),
               capprops=dict(color="black", linewidth=0.8))

    ax.axhline(0, color="#C0392B", linestyle="--", linewidth=1, zorder=2,
               label="No change (hinted = unhinted)")

    lo = min(v.min() for v in log_data)
    hi = max(v.max() for v in log_data)
    pad = (hi - lo) * 0.08
    ax.set_ylim(lo - pad, hi + pad * 3.2)
    ticks = [t for t in np.arange(-4, 5) if lo - pad <= t <= hi + pad * 3.2]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"×{2.0 ** t:g}" for t in ticks])

    for x, (_, label, _) in zip(pos, COLUMNS):
        r = ratios[label]
        med_pct = (np.median(r) - 1) * 100
        longer = (r > 1).mean() * 100
        ax.text(x, hi + pad * 1.2,
                f"median {med_pct:+.0f}%\n{longer:.0f}% longer",
                ha="center", va="bottom", fontsize=8.5)

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#DDDDDD", linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#999999")
    ax.set_xticks(pos)
    ax.set_xticklabels([label for _, label, _ in COLUMNS])
    ax.set_xlabel("Hint type")
    ax.set_ylabel("Per-question length ratio, hinted / unhinted (log scale)")
    n = len(ratios[COLUMNS[0][1]])
    ax.set_title(f"{ds} — per-question change in response length when the "
                 f"hint is introduced\n{model_label}, {n} questions; "
                 "ratio of median lengths over each question's runs",
                 fontsize=10.5)
    ax.legend(loc="lower right", fontsize=8, frameon=True,
              edgecolor="#CCCCCC")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Per-question hinted/unhinted length ratio, one figure "
                    "per dataset.")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--data", default="test", choices=SPLITS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    with config_path.open() as f:
        config = yaml.safe_load(f)

    model_name = model_dir_name(args.model)
    if args.out_dir:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = root / out_dir
    else:
        out_dir = ((root / config["paths"]["plots_dir"]).parent
                   / "extra-plots" / "hint-effect-on-length" / model_name)
        if args.data != "test":
            out_dir = out_dir / args.data
    out_dir.mkdir(parents=True, exist_ok=True)

    wrote = 0
    for ds in DATASETS:
        loaded = load_dataset(root, config, model_name, ds, args.data)
        if loaded is None:
            print(f"Skipping {ds}.")
            continue
        ratios = per_question_ratios(*loaded)
        plot_dataset(ds, ratios, args.model,
                     out_dir / f"length_change_per_question_{ds}_{model_name}.png")
        wrote += 1
    if not wrote:
        print("ERROR: no dataset had complete length_pools.")
        sys.exit(1)


if __name__ == "__main__":
    main()
