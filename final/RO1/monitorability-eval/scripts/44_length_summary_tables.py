#!/usr/bin/env python3
"""
Step 44: Response-length summary tables for ONE model, at three aggregation
levels — the length analogue of the main_results_*_level_tables.

Reads the per-(dataset × hint) cot_token_stats.json emitted by step 08
(which carries the raw valid-run token pools per condition), and writes:

    {results_root}/length_dataset_hint_level_tables.md   (24 cells)
    {results_root}/length_dataset_level_tables.md        (8 datasets, hints pooled)
    {results_root}/length_hint_type_level_tables.md      (3 hint types, datasets pooled)

Metrics per condition (chosen for right-skewed, heavy-tailed, truncated token
counts — see the header block written into each file):

  - Median [Q1–Q3]  — PRIMARY central tendency + spread, skew/outlier-robust
  - p90             — the upper tail (where verbosity inflation & truncation
                      live; also what you size max_new_tokens against). Replaces
                      the single-outlier Max.
  - Mean (SD)       — SECONDARY, for comparability with prior work and because
                      totals/cost scale with the mean; the mean−median gap
                      itself flags skew.
  - n runs (inst)   — sample size behind the cell (instances only at cell level)

Per section (below the 4-condition table):
  - Hint effect (Hinted − Unhinted): Δ median (tokens) · ratio (H/U median) —
    the within-model hint-length effect, expressed robustly (NOT Δ mean).
  - Truncation: one honest rate over EVERY response in both arms (this replaces
    the per-bucket histogram); pooled levels sum truncated/total over cells.

Conditions (all four kept, matching step 08 / cot_overall_summary):
  Hinted (X=1, all) · Unhinted (X=0, all) ·
  Hinted, followed (X=1, Y=1) · Hinted, not followed (X=1, Y=0)

All distributional stats use VALID runs only (not truncated, answer parsed) —
the same pools the results artifacts use — so truncation is reported
separately as a rate rather than mixed into the percentiles.

Pooling notes:
  - Dataset level pools the 3 hint types. The UNHINTED arm is identical across
    hint types (hint-independent), so it is taken from ONE hint (not tripled);
    hinted / followed / not-followed pools are concatenated over the 3 hints.
    Truncation sums truncated/total over the 3 cells (the shared unhinted arm is
    counted per-hint there — a minor over-weighting of unhinted, noted in-file).
  - Hint-type level pools the 8 datasets (each dataset's unhinted arm is
    distinct, so no double counting).

Usage:
    python scripts/44_length_summary_tables.py                 # base model
    python scripts/44_length_summary_tables.py --model <hf-id>

    Options:
      --config   path to config YAML (default: config/mvp_config.yaml)
      --data     split: test (default) | eval
      --model    model/adapter HF id (default: Qwen/Qwen3-4B)
      --out-dir  directory for the 3 .md files (default: results_root, next to
                 the main_results_*_level_tables)

Input (per dataset × hint, produced by 08_cot_token_analysis.py):
    {analysis_dir}/{model}/{dataset}/{hint}/cot_token_stats.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from src.dataset import SPLITS, model_dir_name, resolve_paths
from src.hints import HINT_TYPES

DEFAULT_MODEL = "Qwen/Qwen3-4B"

DATASETS = ["mmlu_pro", "gpqa_diamond", "lsat_ar", "lsat_lr",
            "medxpertqa", "lexam", "codemmlu", "wmdp"]

# (display label, pool key, condition_stats prefix)
CONDITIONS = [
    ("Hinted (X=1, all)", "hinted", "hinted"),
    ("Unhinted (X=0, all)", "unhinted", "unhinted"),
    ("Hinted, followed (X=1, Y=1)", "hint_followed", "hint_followed"),
    ("Hinted, not followed (X=1, Y=0)", "hint_not_followed", "hint_not_followed"),
]

HINT_LABEL = {"professor": "Professor", "grader_hacking": "Grader-hacking",
              "unethical": "Unethical"}


def stats(pool):
    """median, q1, q3, p90, mean, std, n from a raw token array."""
    a = np.asarray(pool, float)
    if a.size == 0:
        return None
    q1, med, q3, p90 = np.percentile(a, [25, 50, 75, 90])
    return {"median": med, "q1": q1, "q3": q3, "p90": p90,
            "mean": float(a.mean()), "std": float(a.std(ddof=1)) if a.size > 1
            else 0.0, "n": int(a.size)}


def fmt_row(label, s, n_inst=None):
    if s is None:
        n = "0" if n_inst is None else f"0 ({n_inst})"
        return f"| {label} | — | — | — | {n} |"
    med = f"{s['median']:.0f} [{s['q1']:.0f}–{s['q3']:.0f}]"
    p90 = f"{s['p90']:.0f}"
    mean = f"{s['mean']:.0f} ({s['std']:.0f})"
    n = f"{s['n']}" if n_inst is None else f"{s['n']} ({n_inst})"
    return f"| {label} | {med} | {p90} | {mean} | {n} |"


def section(cond_pools, cond_inst, trunc, unhinted_note=False):
    """Render one section body: 4-condition table + gap + truncation."""
    out = ["| Condition | Median [Q1–Q3] | p90 | Mean (SD) | n runs"
           + (" (inst)" if cond_inst else "") + " |",
           "|---|---|---|---|---|"]
    s = {}
    for label, key, _ in CONDITIONS:
        st = stats(cond_pools.get(key, []))
        s[key] = st
        out.append(fmt_row(label, st, cond_inst.get(key) if cond_inst else None))
    # gap: hinted vs unhinted median
    extra = []
    if s.get("hinted") and s.get("unhinted"):
        dmed = s["hinted"]["median"] - s["unhinted"]["median"]
        ratio = s["hinted"]["median"] / s["unhinted"]["median"]
        extra.append(f"**Hint effect (Hinted − Unhinted):** "
                     f"Δ median = {dmed:+.0f} tok · ratio = {ratio:.2f}×")
    if trunc is not None:
        tn, tt = trunc
        rate = (tn / tt * 100) if tt else 0.0
        note = " (both arms; unhinted counted per-hint)" if unhinted_note \
            else " (both arms)"
        extra.append(f"**Truncation{note}:** {rate:.1f}% ({tn}/{tt})")
    return "\n".join(out) + "\n\n" + "\n\n".join(extra) + "\n"


HEADER_COLS = (
    "Columns: **Median [Q1–Q3]** primary central tendency + spread (skew-robust) · "
    "**p90** upper tail · **Mean (SD)** secondary (comparability/cost; the "
    "mean−median gap flags skew) · **n runs** valid runs behind the cell. "
    "Distributional stats use VALID runs only (not truncated, answer parsed); "
    "**Truncation** is reported separately as one rate over every response in "
    "both arms. **Hint effect** is the Hinted-vs-Unhinted median contrast "
    "(Δ median tokens and H/U median ratio) — the robust analogue of a mean gap."
)


def load_all(root, config, model_name, split):
    data = {}
    missing = []
    for ds in DATASETS:
        data[ds] = {}
        for hint in HINT_TYPES:
            paths = resolve_paths(root, config, ds, model_name, hint, split=split)
            jp = paths["analysis_dir"] / "cot_token_stats.json"
            if not jp.exists():
                missing.append(f"{ds}/{hint}")
                continue
            data[ds][hint] = json.load(jp.open())
    return data, missing


def cond_inst_from_json(j):
    cs = j["condition_stats"]
    return {key: cs.get(f"{key}_n_instances") for _, key, _ in CONDITIONS}


def build_dataset_hint(data, model_id):
    out = [f"# Response length — per (dataset, hint type), no pooling", "",
           f"Model: `{model_id}`", "", HEADER_COLS, ""]
    for ds in DATASETS:
        out.append(f"## {ds}")
        out.append("")
        for hint in HINT_TYPES:
            j = data[ds].get(hint)
            if not j:
                continue
            pools = j["cot_token_pools"]
            trunc = (j["truncation"]["truncated"], j["truncation"]["total"])
            out.append(f"### {ds} — {HINT_LABEL[hint]} hint\n")
            out.append(section(pools, cond_inst_from_json(j), trunc))
    return "\n".join(out)


def build_dataset(data, model_id):
    out = [f"# Response length — per dataset (hint types pooled)", "",
           f"Model: `{model_id}`", "", HEADER_COLS,
           "\n*Unhinted is hint-independent, so it is taken from one hint arm "
           "(not tripled); hinted/followed/not-followed pool the 3 hint types.*",
           ""]
    for ds in DATASETS:
        hints = [h for h in HINT_TYPES if data[ds].get(h)]
        if not hints:
            continue
        pools = {}
        for _, key, _ in CONDITIONS:
            if key == "unhinted":
                pools[key] = data[ds][hints[0]]["cot_token_pools"].get(key, [])
            else:
                acc = []
                for h in hints:
                    acc += data[ds][h]["cot_token_pools"].get(key, [])
                pools[key] = acc
        tn = sum(data[ds][h]["truncation"]["truncated"] for h in hints)
        tt = sum(data[ds][h]["truncation"]["total"] for h in hints)
        out.append(f"## {ds}\n")
        out.append(section(pools, None, (tn, tt), unhinted_note=True))
    return "\n".join(out)


def build_hint_type(data, model_id):
    out = [f"# Response length — per hint type (datasets pooled)", "",
           f"Model: `{model_id}`", "", HEADER_COLS,
           "\n*Datasets pooled; each dataset's unhinted arm is distinct, so no "
           "double counting. Note the pool mixes very different length regimes "
           "(e.g. lsat_ar ~5.5k vs medxpertqa ~1.6k median tokens) — read as a "
           "summary, per-dataset tables are the meaningful unit.*", ""]
    for hint in HINT_TYPES:
        dss = [ds for ds in DATASETS if data[ds].get(hint)]
        if not dss:
            continue
        pools = {}
        for _, key, _ in CONDITIONS:
            acc = []
            for ds in dss:
                acc += data[ds][hint]["cot_token_pools"].get(key, [])
            pools[key] = acc
        tn = sum(data[ds][hint]["truncation"]["truncated"] for ds in dss)
        tt = sum(data[ds][hint]["truncation"]["total"] for ds in dss)
        out.append(f"## {HINT_LABEL[hint]} hint\n")
        out.append(section(pools, None, (tn, tt)))
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(
        description="Response-length summary tables at 3 aggregation levels.")
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
    data, missing = load_all(root, config, model_name, args.data)
    if missing:
        print("WARNING: missing cot_token_stats.json for: " + ", ".join(missing))
    if not any(data[ds] for ds in DATASETS):
        print("ERROR: no cot_token_stats.json found — run step 08 first.")
        sys.exit(1)

    # results_root = .../monitorability-results (parent of the model dir)
    sample = next((resolve_paths(root, config, ds, model_name, h, split=args.data)
                   for ds in DATASETS for h in HINT_TYPES if data[ds].get(h)),
                  None)
    results_root = sample["analysis_dir"].parents[2]
    out_dir = Path(args.out_dir) if args.out_dir else results_root
    if args.out_dir and not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "length_dataset_hint_level_tables.md": build_dataset_hint(data, args.model),
        "length_dataset_level_tables.md": build_dataset(data, args.model),
        "length_hint_type_level_tables.md": build_hint_type(data, args.model),
    }
    for name, text in files.items():
        p = out_dir / name
        p.write_text(text.rstrip() + "\n", encoding="utf-8")
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
