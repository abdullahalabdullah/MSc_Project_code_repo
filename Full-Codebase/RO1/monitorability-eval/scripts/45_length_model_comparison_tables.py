#!/usr/bin/env python3
"""
Step 45: Response-length model-comparison tables — Base vs Harmlessness DPO vs
Anti-sycophancy DPO — at the same three aggregation levels as step 44 (the
length analogue of the main_results comparison tables, but purely descriptive:
NO confidence intervals).

Reads the per-(dataset × hint) cot_token_stats.json `length_pools` (per-run
instance_id + token_count + truncated, truncated INCLUDED at the 16384-token
cap) for the THREE models and writes:

    {results_root}/length_dataset_level_model_comparison.md       (8 datasets, hints pooled)
    {results_root}/length_dataset_hint_level_model_comparison.md  (24 cells)
    {results_root}/length_hint_type_level_model_comparison.md     (3 hint types, datasets pooled)

One table per section, models as columns:

    | Metric | Base | Harmlessness DPO | Anti-sycophancy DPO |
      Harmlessness DPO / Base | Anti-sycophancy DPO / Base |

Rows (median-first, matching step 44's design for right-skewed truncated
token counts):

  - Hinted median tokens (X=1, all)    — model cells: Median [Q1–Q3];
                                         contrast cells: PAIRED GMR trained/base
  - Unhinted median tokens (X=0, all)  — same

The contrast is the paired per-question geometric-mean ratio (the cross-model
analogue of step 44's H/U GMR): each question's runs are collapsed to their
per-question MEDIAN length per model, then
GMR = exp(mean_q[ln median(trained_q) − ln median(base_q)]) over questions
present in both models — the typical per-question multiplicative length
change. Per level:

  - dataset×hint level (the finest cells): bucket = the question's 8 runs for
    that hint. The plain per-question GMR.
  - POOLED levels — ONE RULE: the contrast is the GM of the constituent
    dataset×hint cell GMRs, equal weight per cell (derivable by hand from the
    dataset×hint file):
      * dataset level, hinted row = GM of that dataset's 3 per-hint cell GMRs
        (equal weight per hint type). With this complete balanced grid (every
        question × 3 hints × 8 runs in all three models) this equals the
        bottom-up construction — per-(question, hint) log-ratios averaged over
        hints within each question, then over questions. Deliberately NOT a
        GMR over 3-hint-pooled 24-run per-question medians: a mixture median
        tracks whichever runs straddle the middle of the mixed distribution
        rather than averaging the per-hint effects.
      * dataset level, unhinted row = the plain per-question GMR on the single
        shared unhinted arm (8 runs/question) — which already IS its one cell
        GMR, so the rule holds trivially.
      * hint-type level, both rows = GM of that hint's 8 per-dataset cell
        GMRs (equal weight per DATASET — the datasets are the meaningful
        strata, so a dataset never gains weight from its question count; with
        near-equal counts, 198-200 questions/dataset, this differs from the
        question-weighted GMR only in the 2nd decimal).

The pairing is clean at EVERY level because all three models ran the
identical instances (unlike the hinted-vs-unhinted contrast, where pooled
levels mix hint types against one unhinted arm). NOTE the GMR is NOT derivable
from the displayed pooled medians (a ratio of those is the marginal contrast;
the paired GMR weights every question equally instead).

Deliberately omitted (per the comparison design): the hint-effect
(Hinted − Unhinted) row (derivable from the two rows kept; the per-model hint
effect lives in the step-44 files), the followed/not-followed splits
(conditioned on the outcome Y, which itself shifts under training, so
cross-model deltas confound composition vs length), p90 rows and truncation
rows (truncation is ≤1.0% in 7/8 datasets, ~4.6–5.9% in wmdp across the three
models — far below the 50% that would move a median; per-model tails live in the step-44
files), Mean (SD), and any CIs — these are descriptive summaries of the
observed runs.

Pooling matches step 44 exactly: dataset level pools the 3 hint types and takes
the hint-independent unhinted arm from ONE hint (not tripled); hint-type level
pools the 8 datasets (each dataset's unhinted arm distinct). Length stats
INCLUDE truncated runs at the cap (dropping the longest runs would bias length
down; the cap understates the true tail, so estimates are conservative).

Usage:
    python scripts/45_length_model_comparison_tables.py
    # optional overrides:
    #   --base / --harmless / --antisyc   HF ids of the three models
    #   --config --data --out-dir         as in step 44

Input (per model × dataset × hint, produced by 08_cot_token_analysis.py —
re-run step 08 for a model if its cot_token_stats.json predates length_pools):
    {analysis_dir}/{model}/{dataset}/{hint}/cot_token_stats.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from src.dataset import SPLITS, model_dir_name, resolve_paths
from src.hints import HINT_TYPES

DEFAULT_BASE = "Qwen/Qwen3-4B"
DEFAULT_HARMLESS = "er-abd/harmless-dpo-qwen3-4B-step-735"
DEFAULT_ANTISYC = "er-abd/anti-syc-dpo-qwen3-4B-step-520"

DATASETS = ["mmlu_pro", "gpqa_diamond", "lsat_ar", "lsat_lr",
            "medxpertqa", "lexam", "codemmlu", "wmdp"]

HINT_LABEL = {"professor": "Professor", "grader_hacking": "Grader-hacking",
              "unethical": "Unethical"}

HEADER = (
    "Columns: three models' **Median [Q1–Q3]** (skew-robust central tendency + "
    "spread, in tokens), then the two trained-vs-base **paired GMR** contrasts "
    "— the per-question geometric-mean ratio: each question's runs are "
    "collapsed to their per-question MEDIAN length per model, then GMR = "
    "exp(mean over questions of [ln median(trained_q) − ln median(base_q)]) "
    "(>1× = the trained model is typically longer on the same question; "
    "point estimates only — descriptive summaries of observed runs, no "
    "confidence intervals). The pairing is exact at every level: all three "
    "models ran the identical instances, and questions are keyed by "
    "(dataset, instance_id) so pooled levels pair correctly. Note the GMR is "
    "NOT the ratio of the displayed pooled medians (that is the marginal "
    "contrast; the paired GMR weights every question equally). Length stats "
    "INCLUDE truncated runs at the 16384-token cap (truncation is caused by "
    "length, so dropping them would bias length down; the cap understates the "
    "true tail, so estimates are conservative). Hint-effect (Hinted − "
    "Unhinted) and followed/not-followed rows are omitted; per-model hint "
    "effects, p90 and truncation live in the step-44 `length_*_level_tables` "
    "files."
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
            j = json.load(jp.open())
            if "length_pools" not in j:
                missing.append(f"{ds}/{hint} (no length_pools — re-run step 08)")
                continue
            data[ds][hint] = j
    return data, missing


def pool_tokens(cells, key, unhinted_single=False):
    """Concatenate one condition's token counts over (ds, json) cells.
    unhinted_single keeps one cell (the hint-independent unhinted arm) so
    pooling hint types doesn't triple-count it. Also returns (n_runs, n_inst)
    with distinct instances counted within each dataset then summed."""
    if unhinted_single:
        cells = cells[:1]
    tokens = []
    per_ds = defaultdict(set)
    for ds, j in cells:
        lp = j["length_pools"].get(key)
        if not lp:
            continue
        tokens += lp["token_count"]
        per_ds[ds].update(lp["instance_id"])
    n_inst = sum(len(s) for s in per_ds.values())
    return np.asarray(tokens, float), len(tokens), n_inst


def per_question_medians(cells, key, unhinted_single=False):
    """Collapse one condition's runs to a per-question MEDIAN length, keyed by
    (dataset, instance_id) so pooled levels never collide on the reused
    instance_ids 0..199. At pooled levels a question's runs are pooled across
    its cells (e.g. the 3 hint types) before the median. Same unhinted_single
    convention as pool_tokens."""
    if unhinted_single:
        cells = cells[:1]
    buckets = defaultdict(list)
    for ds, j in cells:
        lp = j["length_pools"].get(key)
        if not lp:
            continue
        for iid, tok in zip(lp["instance_id"], lp["token_count"]):
            buckets[(ds, iid)].append(tok)
    return {k: float(np.median(v)) for k, v in buckets.items()}


def paired_gmr(trained_pq, base_pq):
    """Paired per-question geometric-mean ratio trained/base:
    exp(mean_q[ln median(trained_q) − ln median(base_q)]) over questions
    present in both models. Token counts are strictly positive, so the logs
    are always defined."""
    lr = [np.log(trained_pq[k]) - np.log(base_pq[k])
          for k in trained_pq.keys() & base_pq.keys()
          if trained_pq[k] > 0 and base_pq[k] > 0]
    return float(np.exp(np.mean(lr))) if lr else None


def med_stats(tokens):
    if tokens.size == 0:
        return None
    q1, med, q3 = np.percentile(tokens, [25, 50, 75])
    return {"median": med, "q1": q1, "q3": q3}


def model_summary(cells, unhinted_single):
    """Per-model summary for one section: hinted/unhinted pooled median stats,
    per-question medians (for the paired GMR contrast), and run counts."""
    h_tok, h_runs, h_inst = pool_tokens(cells, "hinted")
    u_tok, u_runs, u_inst = pool_tokens(cells, "unhinted",
                                        unhinted_single=unhinted_single)
    return {"hinted": med_stats(h_tok), "unhinted": med_stats(u_tok),
            "pq": {"hinted": per_question_medians(cells, "hinted"),
                   "unhinted": per_question_medians(
                       cells, "unhinted", unhinted_single=unhinted_single)},
            "n": (h_runs, h_inst, u_runs, u_inst)}


def fmt_med(s):
    return f"{s['median']:.0f} [{s['q1']:.0f}–{s['q3']:.0f}]" if s else "—"


def fmt_gmr(trained, base, key):
    """Paired GMR contrast cell for one condition row."""
    g = paired_gmr(trained["pq"][key], base["pq"][key])
    return f"{g:.2f}×" if g is not None else "—"


def fmt_ratio(g):
    return f"{g:.2f}×" if g is not None else "—"


def gm_of_cell_gmrs(pairs, key):
    """GM over dataset×hint cells of the per-cell trained/base GMR: pairs is a
    list of ((ds, trained_json), (ds, base_json)) cell pairs; each cell's GMR
    is the plain per-question paired GMR, then the cells are combined with a
    geometric mean (equal weight per cell)."""
    logs = []
    for t_cell, b_cell in pairs:
        g = paired_gmr(per_question_medians([t_cell], key),
                       per_question_medians([b_cell], key))
        if g is not None:
            logs.append(np.log(g))
    return float(np.exp(np.mean(logs))) if logs else None


def section(models, gmr_override=None):
    """Render one section from [(label, summary), ...] (Base first).
    gmr_override maps a condition key ("hinted"/"unhinted") to
    {trained label: ratio}, replacing that row's contrast cells — the pooled
    levels pass their GM-of-constituent-cell-GMRs there."""
    labels = [lb for lb, _ in models]
    out = ["| Metric | " + " | ".join(labels[:3])
           + f" | {labels[1]} / {labels[0]} | {labels[2]} / {labels[0]} |",
           "|---|---|---|---|---|---|"]
    base, harm, anti = (s for _, s in models)
    for row, key in [("Hinted median tokens (X=1, all)", "hinted"),
                     ("Unhinted median tokens (X=0, all)", "unhinted")]:
        if gmr_override and key in gmr_override:
            contrasts = (fmt_ratio(gmr_override[key].get(labels[1])),
                         fmt_ratio(gmr_override[key].get(labels[2])))
        else:
            contrasts = (fmt_gmr(harm, base, key), fmt_gmr(anti, base, key))
        out.append(f"| {row} | " + " | ".join(fmt_med(s[key]) for s in
                                              (base, harm, anti))
                   + f" | {contrasts[0]} | {contrasts[1]} |")
    ns = " · ".join(f"{lb} {s['n'][0]}/{s['n'][2]}" for lb, s in models)
    out.append("")
    out.append(f"*n runs (hinted/unhinted): {ns}*")
    return "\n".join(out) + "\n"


def title_block(title, models, note=None):
    lines = [f"# {title}", ""]
    lines += [f"- **{lb}**: `{mid}`" for lb, mid in models]
    lines += ["", HEADER, ""]
    if note:
        lines += [note, ""]
    return lines


def build_dataset_hint(data, models):
    out = title_block("Response length model comparison — per (dataset, "
                      "hint type), no pooling", models)
    for ds in DATASETS:
        out.append(f"## {ds}")
        out.append("")
        for hint in HINT_TYPES:
            rows = []
            for label, _mid in models:
                j = data[label][ds].get(hint)
                if not j:
                    break
                rows.append((label, model_summary([(ds, j)],
                                                  unhinted_single=False)))
            if len(rows) < len(models):
                continue
            out.append(f"### {ds} — {HINT_LABEL[hint]} hint\n")
            out.append(section(rows))
    return "\n".join(out)


def build_dataset(data, models):
    out = title_block("Response length model comparison — per dataset "
                      "(hint types pooled)", models,
                      note="*Unhinted is hint-independent, so it is taken from "
                      "one hint arm (not tripled); the hinted model cells pool "
                      "the 3 hint types. The hinted CONTRAST cells are the "
                      "geometric mean of the dataset's three per-hint cell "
                      "GMRs (equal weight per hint type — verifiable from the "
                      "dataset×hint file; with this complete balanced grid it "
                      "equals averaging per-(question, hint) log-ratios over "
                      "hints within each question, then over questions), NOT a "
                      "GMR over 3-hint-pooled per-question medians — a mixture "
                      "median would track whichever runs straddle the middle "
                      "of the mixed distribution instead of averaging the "
                      "per-hint effects.*")
    for ds in DATASETS:
        rows = []
        for label, _mid in models:
            hints = [h for h in HINT_TYPES if data[label][ds].get(h)]
            if not hints:
                break
            cells = [(ds, data[label][ds][h]) for h in hints]
            rows.append((label, model_summary(cells, unhinted_single=True)))
        if len(rows) < len(models):
            continue
        # hinted contrast = GM of the per-hint cell GMRs (equal hint weight);
        # unhinted has one shared arm per dataset, so its plain per-question
        # GMR already IS its (single) cell GMR — no override needed
        base_label = models[0][0]
        common = [h for h in HINT_TYPES
                  if all(data[lb][ds].get(h) for lb, _ in models)]
        hinted_gmr = {
            label: gm_of_cell_gmrs(
                [((ds, data[label][ds][h]), (ds, data[base_label][ds][h]))
                 for h in common], "hinted")
            for label, _mid in models[1:]}
        out.append(f"## {ds}\n")
        out.append(section(rows, gmr_override={"hinted": hinted_gmr}))
    return "\n".join(out)


def build_hint_type(data, models):
    out = title_block("Response length model comparison — per hint type "
                      "(datasets pooled)", models,
                      note="*Datasets pooled; each dataset's unhinted arm is "
                      "distinct, so no double counting. Note the pool mixes "
                      "very different length regimes — read as a summary, "
                      "per-dataset tables are the meaningful unit. "
                      "Accordingly, BOTH contrast cells are the geometric "
                      "mean of the hint type's eight per-dataset cell GMRs "
                      "(equal weight per dataset — verifiable from the "
                      "dataset×hint file), not a GMR over the pooled "
                      "question set.*")
    for hint in HINT_TYPES:
        rows = []
        for label, _mid in models:
            dss = [ds for ds in DATASETS if data[label][ds].get(hint)]
            if not dss:
                break
            cells = [(ds, data[label][ds][hint]) for ds in dss]
            rows.append((label, model_summary(cells, unhinted_single=False)))
        if len(rows) < len(models):
            continue
        # both contrasts = GM of the 8 per-dataset cell GMRs (equal dataset
        # weight — the datasets are the meaningful strata, per the note above)
        base_label = models[0][0]
        common_ds = [ds for ds in DATASETS
                     if all(data[lb][ds].get(hint) for lb, _ in models)]
        override = {}
        for key in ("hinted", "unhinted"):
            override[key] = {
                label: gm_of_cell_gmrs(
                    [((ds, data[label][ds][hint]),
                      (ds, data[base_label][ds][hint])) for ds in common_ds],
                    key)
                for label, _mid in models[1:]}
        out.append(f"## {HINT_LABEL[hint]} hint\n")
        out.append(section(rows, gmr_override=override))
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(
        description="Base vs Harmlessness DPO vs Anti-sycophancy DPO "
                    "response-length comparison tables (no CIs).")
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

    models = [("Base", args.base),
              ("Harmlessness DPO", args.harmless),
              ("Anti-sycophancy DPO", args.antisyc)]

    data = {}
    for label, mid in models:
        d, missing = load_all(root, config, model_dir_name(mid), args.data)
        if missing:
            print(f"WARNING [{label} — {mid}]: missing/unenriched "
                  "cot_token_stats.json for: " + ", ".join(missing))
        if not any(d[ds] for ds in DATASETS):
            print(f"ERROR: no usable cot_token_stats.json for {mid} — "
                  "run step 08 for it first.")
            sys.exit(1)
        data[label] = d

    sample = next(resolve_paths(root, config, ds, model_dir_name(args.base),
                                h, split=args.data)
                  for ds in DATASETS for h in HINT_TYPES
                  if data["Base"][ds].get(h))
    results_root = sample["analysis_dir"].parents[2]
    out_dir = Path(args.out_dir) if args.out_dir else results_root
    if args.out_dir and not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "length_dataset_level_model_comparison.md":
            build_dataset(data, models),
        "length_dataset_hint_level_model_comparison.md":
            build_dataset_hint(data, models),
        "length_hint_type_level_model_comparison.md":
            build_hint_type(data, models),
    }
    for name, text in files.items():
        p = out_dir / name
        p.write_text(text.rstrip() + "\n", encoding="utf-8")
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
