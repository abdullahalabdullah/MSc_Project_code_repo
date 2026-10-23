#!/usr/bin/env python3
"""
Step 46: Response-length bar charts — base vs harmlessness DPO vs
anti-sycophancy DPO — the length analogues of the three cross-model g-mean²
figures (16/17/18), for summarising response length in the paper as figures
instead of tables.

Three PNGs, mirroring the g-mean² figures' layout/colors/hatches:

    length_by_hint_3models.png      (like 16: x = 8 datasets, an unhinted set
                                     + 3 hint-type sets, x 3 model bars each)
    length_pooled_3models.png       (like 17: x = 8 datasets, 3 model bars,
                                     hint types pooled)
    length_by_arm_3models.png       (x = 8 datasets, unhinted set + hinted
                                     set (3 hint types pooled) x 3 model bars)

(No hint-type-level figure: a datasets-pooled median mixes very different
length regimes — lsat_ar ~5.5k vs wmdp ~0.9k median tokens — and mostly
repeats the model effect three times, so it was dropped; likewise no red
average-of-all-bars reference line — the comparison is bar-vs-bar and the
all-bar mean corresponds to no citable quantity.)

Each bar = the MEDIAN completion length in tokens. NO error bars —
descriptive summaries of observed runs, matching the length tables' no-CI
convention. The value is printed above each bar.

Pools come from step 08's length_pools and INCLUDE truncated runs at the
16384-token cap (truncation is caused by length; dropping the longest runs
would bias length down). The by-hint figure uses per-arm medians: the ONE
shared unhinted arm (1600) and each hint type's hinted arm (1600). The
per-dataset pooled figure pools the 3 hint types' hinted runs (4800) + the
ONE shared unhinted arm (1600, not tripled — a 3:1 hinted-weighted mix).

Colors/hatches match 16/17: unhinted light gray and hint types light
blue/pink/green in the by-hint figure, with model identity carried by hatch
(base solid, harmlessness DPO striped, anti-sycophancy DPO dotted); the same
trio of colors is bound to
MODELS in the pooled figure, hatches unchanged, so both figures survive
grayscale.

Usage:
    python scripts/46_length_barcharts.py

    Options:
      --config    path to config YAML (default: config/mvp_config.yaml)
      --data      split: test (default) | eval
      --base      base model id (default: Qwen/Qwen3-4B)
      --harmless  harmlessness DPO id (default: er-abd/harmless-dpo-qwen3-4B-step-735)
      --antisyc   anti-sycophancy DPO id (default: er-abd/anti-syc-dpo-qwen3-4B-step-520)
      --out-dir   output directory (default: {plots_dir}/../extra-plots/
                  cross-model-comparison, same as the g-mean² trio)

Input (per model x dataset x hint, produced by 08_cot_token_analysis.py —
needs the enriched length_pools):
    {analysis_dir}/{model}/{dataset}/{hint}/cot_token_stats.json

Output:
    {out_dir}/length_by_hint_3models.png
    {out_dir}/length_pooled_3models.png
    {out_dir}/length_by_arm_3models.png
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import yaml

from src.dataset import SPLITS, model_dir_name, resolve_paths
from src.hints import HINT_TYPES

DEFAULT_BASE = "Qwen/Qwen3-4B"
DEFAULT_HARMLESS = "er-abd/harmless-dpo-qwen3-4B-step-735"
DEFAULT_ANTISYC = "er-abd/anti-syc-dpo-qwen3-4B-step-520"

# Same presentation order as the main-results tables.
DATASETS = ["mmlu_pro", "gpqa_diamond", "lsat_ar", "lsat_lr",
            "medxpertqa", "lexam", "codemmlu", "wmdp"]

HINTS = [
    ("professor", "Professor hint", "#92C5E8"),            # light blue
    ("grader_hacking", "Grader-hacking hint", "#F6A0B8"),  # light pink
    ("unethical", "Unethical hint", "#9ED973"),            # light green
]
COLOR_BASE = "#92C5E8"      # light blue
COLOR_HARMLESS = "#F6A0B8"  # light pink
COLOR_ANTISYC = "#9ED973"   # light green
# Arm colors for the by-arm figure: light tints of the project's arm hues
# (hinted = plum, unhinted = blue — steps 37/41).
COLOR_HINTED = "#C7ADE6"    # light plum
COLOR_UNHINTED = "#92C5E8"  # light blue
# Unhinted set in the by-hint figure: gray, so it can't be confused with the
# light-blue professor hint.
COLOR_UNHINTED_GRAY = "#D3D3D3"  # light gray
EDGE = "#555555"


def load_all(root, config, model_name, split):
    """data[ds][hint] = the cell's length_pools (needs enriched step-08 json)."""
    data = {}
    missing = []
    for ds in DATASETS:
        data[ds] = {}
        for hint in HINT_TYPES:
            paths = resolve_paths(root, config, ds, model_name, hint,
                                  split=split)
            jp = paths["analysis_dir"] / "cot_token_stats.json"
            if not jp.exists():
                missing.append(f"{ds}/{hint}")
                continue
            j = json.load(jp.open())
            if "length_pools" not in j:
                missing.append(f"{ds}/{hint} (no length_pools — re-run step 08)")
                continue
            data[ds][hint] = j["length_pools"]
    return data, missing


def style_axes(ax, top):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#DDDDDD", linewidth=0.8)
    ax.set_ylim(0, top)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#999999")


def add_legend(ax, handles):
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.005, 1.0),
              frameon=True, edgecolor="#CCCCCC", fontsize=9)


def fig_by_hint(models, out):
    """Like 16: x = datasets, 4 sets x 3 model bars — the shared unhinted arm
    (gray, 1600 runs) then the 3 hint types' hinted arms (1600 runs each);
    set carried by color, model by hatch. Bars are per-arm medians (not the
    combined hinted + unhinted pool) so each hint set compares directly
    against the unhinted set."""
    BAR_W, SET_OFFSETS = 0.065, [-0.375, -0.125, 0.125, 0.375]
    sets = [(None, "Unhinted", COLOR_UNHINTED_GRAY)] + HINTS
    fig, ax = plt.subplots(figsize=(19, 5.5))
    bars = []  # (x, value, color, hatch)
    for d_idx, ds in enumerate(DATASETS):
        for (hint, _, color), s_off in zip(sets, SET_OFFSETS):
            for m_idx, (label, mdata, hatch, _mcolor) in enumerate(models):
                hints = [h for h in HINT_TYPES if mdata[ds].get(h)]
                if not hints:
                    continue
                if hint is None:  # the one shared unhinted arm
                    tokens = mdata[ds][hints[0]]["unhinted"]["token_count"]
                elif mdata[ds].get(hint):
                    tokens = mdata[ds][hint]["hinted"]["token_count"]
                else:
                    continue
                med = float(np.median(tokens))
                x = d_idx + s_off + (m_idx - 1) * (BAR_W + 0.01)
                bars.append((x, med, color, hatch))
    top = max(v for _, v, _, _ in bars) * 1.28
    style_axes(ax, top)
    for x, med, color, hatch in bars:
        ax.bar(x, med, width=BAR_W, color=color, hatch=hatch,
               edgecolor=EDGE, linewidth=0.6, zorder=3)
        ax.text(x, med + top * 0.012, f"{med:.0f}", rotation=90,
                ha="center", va="bottom", fontsize=6.5, color="black",
                zorder=5, clip_on=False)
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels(DATASETS)
    ax.set_xlim(-0.55, len(DATASETS) - 0.45)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Median response length (tokens)")
    ax.set_title("Median response length per dataset: unhinted arm vs each "
                 "hint type's hinted arm — "
                 + " vs ".join(m[0] for m in models), pad=22)
    handles = ([Patch(facecolor=c, edgecolor=EDGE, linewidth=0.6, label=lbl)
                for _, lbl, c in sets]
               + [Patch(facecolor="white", edgecolor=EDGE, linewidth=0.8,
                        hatch=hatch, label=label)
                  for label, _, hatch, _ in models])
    add_legend(ax, handles)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def fig_pooled(models, out):
    """Like 17: x = datasets, 3 model bars, hint types pooled. Bar = median
    over 3 hints' hinted runs + the ONE shared unhinted arm (4800+1600)."""
    BAR_W, MODEL_OFFSETS = 0.22, [-0.25, 0.0, 0.25]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = []
    for d_idx, ds in enumerate(DATASETS):
        for (label, mdata, hatch, color), off in zip(models, MODEL_OFFSETS):
            hints = [h for h in HINT_TYPES if mdata[ds].get(h)]
            if not hints:
                continue
            tokens = []
            for h in hints:
                tokens += mdata[ds][h]["hinted"]["token_count"]
            tokens += mdata[ds][hints[0]]["unhinted"]["token_count"]
            med = float(np.median(tokens))
            bars.append((d_idx + off, med, color, hatch))
    top = max(v for _, v, _, _ in bars) * 1.22
    style_axes(ax, top)
    for x, med, color, hatch in bars:
        ax.bar(x, med, width=BAR_W, color=color, hatch=hatch,
               edgecolor=EDGE, linewidth=0.6, zorder=3)
        ax.text(x, med + top * 0.012, f"{med:.0f}", ha="center", va="bottom",
                fontsize=7.5, color="black", zorder=5, clip_on=False)
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels(DATASETS, rotation=20, ha="right")
    ax.set_xlim(-0.55, len(DATASETS) - 0.45)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Median response length (tokens)")
    ax.set_title("Median response length per dataset, hint types pooled "
                 "(hinted + unhinted runs)", pad=22)
    handles = [Patch(facecolor=color, edgecolor=EDGE, linewidth=0.8,
                     hatch=hatch, label=label)
               for label, _, hatch, color in models]
    add_legend(ax, handles)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def fig_by_arm(models, out):
    """x = datasets, TWO sets (Unhinted / Hinted, left to right) x 3 model bars — shows
    whether the training-induced length increase appears in both arms or only
    one. Hinted set pools the 3 hint types' hinted runs (4800); unhinted set
    is the ONE shared arm (1600). Arm carried by color (plum/blue, matching
    the 37/41 arm hues), model by hatch."""
    BAR_W, ARM_OFFSETS = 0.11, [-0.2, 0.2]
    arms = [("Unhinted", COLOR_UNHINTED, False),
            ("Hinted (3 hint types pooled)", COLOR_HINTED, True)]
    fig, ax = plt.subplots(figsize=(16, 5.5))
    bars = []
    for d_idx, ds in enumerate(DATASETS):
        for (_, color, is_hinted), a_off in zip(arms, ARM_OFFSETS):
            for m_idx, (label, mdata, hatch, _mcolor) in enumerate(models):
                hints = [h for h in HINT_TYPES if mdata[ds].get(h)]
                if not hints:
                    continue
                if is_hinted:
                    tokens = []
                    for h in hints:
                        tokens += mdata[ds][h]["hinted"]["token_count"]
                else:
                    tokens = mdata[ds][hints[0]]["unhinted"]["token_count"]
                med = float(np.median(tokens))
                x = d_idx + a_off + (m_idx - 1) * (BAR_W + 0.012)
                bars.append((x, med, color, hatch))
    top = max(v for _, v, _, _ in bars) * 1.24
    style_axes(ax, top)
    for x, med, color, hatch in bars:
        ax.bar(x, med, width=BAR_W, color=color, hatch=hatch,
               edgecolor=EDGE, linewidth=0.6, zorder=3)
        ax.text(x, med + top * 0.012, f"{med:.0f}", rotation=90,
                ha="center", va="bottom", fontsize=7, color="black",
                zorder=5, clip_on=False)
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels(DATASETS)
    ax.set_xlim(-0.55, len(DATASETS) - 0.45)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Median response length (tokens)")
    ax.set_title("Median response length per dataset, hinted vs unhinted arm — "
                 + " vs ".join(m[0] for m in models), pad=22)
    handles = ([Patch(facecolor=c, edgecolor=EDGE, linewidth=0.6, label=lbl)
                for lbl, c, _ in arms]
               + [Patch(facecolor="white", edgecolor=EDGE, linewidth=0.8,
                        hatch=hatch, label=label)
                  for label, _, hatch, _ in models])
    add_legend(ax, handles)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Median response-length bar charts (3 models), the length "
                    "analogues of the g-mean² figure trio.")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--data", default="test", choices=SPLITS)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--harmless", default=DEFAULT_HARMLESS)
    parser.add_argument("--antisyc", default=DEFAULT_ANTISYC)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    with config_path.open() as f:
        config = yaml.safe_load(f)

    specs = [("Base (Qwen3-4B)", args.base, None, COLOR_BASE),
             ("Harmlessness DPO", args.harmless, "///", COLOR_HARMLESS),
             ("Anti-sycophancy DPO", args.antisyc, "..", COLOR_ANTISYC)]
    models = []
    for label, mid, hatch, color in specs:
        data, missing = load_all(root, config, model_dir_name(mid), args.data)
        if missing:
            print(f"WARNING [{label} — {mid}]: missing/unenriched cells: "
                  + ", ".join(missing))
        if not any(data[ds] for ds in DATASETS):
            print(f"ERROR: no usable cot_token_stats.json for {mid}.")
            sys.exit(1)
        models.append((label, data, hatch, color))

    if args.out_dir:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = root / out_dir
    else:
        # Same home as the g-mean² trio.
        out_dir = ((root / config["paths"]["plots_dir"]).parent
                   / "extra-plots" / "cross-model-comparison")
        if args.data != "test":
            out_dir = out_dir / args.data
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_by_hint(models, out_dir / "length_by_hint_3models.png")
    fig_pooled(models, out_dir / "length_pooled_3models.png")
    fig_by_arm(models, out_dir / "length_by_arm_3models.png")


if __name__ == "__main__":
    main()
