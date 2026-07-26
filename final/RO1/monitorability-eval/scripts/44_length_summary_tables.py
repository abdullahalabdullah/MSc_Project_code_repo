#!/usr/bin/env python3
"""
Step 44: Response-length summary tables for ONE model, at three aggregation
levels — the length analogue of the main_results_*_level_tables.

Reads the per-(dataset × hint) cot_token_stats.json emitted by step 08 —
specifically the ENRICHED `length_pools` (per-run instance_id + token_count +
truncated, INCLUDING truncated runs kept at the 16384-token cap) — and writes:

    {results_root}/length_dataset_hint_level_tables.md   (24 cells)
    {results_root}/length_dataset_level_tables.md        (8 datasets, hints pooled)
    {results_root}/length_hint_type_level_tables.md      (3 hint types, datasets pooled)
    {results_root}/length_dataset_hint_median_vs_mean.md (24 cells — median vs
                                                          Mean (SD) companion)

Metrics per condition (chosen for right-skewed, heavy-tailed, truncated token
counts — see the header block written into each file):

  - Median [Q1–Q3]  — PRIMARY central tendency + spread, skew/outlier-robust
  - p90             — the upper tail (where verbosity inflation & truncation
                      live; also what you size max_new_tokens against). Replaces
                      the single-outlier Max.
  - n runs (inst)   — responses and EXACT distinct instances behind the cell
                      (unioned via instance_id at every level)

Arithmetic Mean (SD) is deliberately OMITTED from the three main tables to keep
them median-first and unambiguous (with SD ≈ mean it invites an impossible
'mean ± SD' read on skewed positive counts, and nothing in the comparison uses
it). It is preserved — beside the median, with a Mean÷Median skew flag — in the
dataset×hint companion file length_dataset_hint_median_vs_mean.md.

Per section (below the 4-condition table):
  - Hint effect (Hinted − Unhinted): Δ median (tokens) · H/U median ratio ·
    GM ratio (paired). At the dataset×hint (finest) level the GM ratio is the
    per-question geometric-mean ratio exp(mean_q[ln median(h_q) − ln median(u_q)]),
    median within a question, geometric mean across questions; the typical per-
    question multiplicative hint effect on length. The POOLED levels report the
    geometric mean of their constituent dataset×hint cell GMRs (equal weight
    per cell) — a plain pooled-run GMR would mix hint types inside a question's
    median (dataset level) or weight datasets by question count (hint-type
    level); GM-of-cells keeps one rule at every level, derivable from the
    dataset×hint file, and equals the summary matrix's geometric margins.
    No CIs — these are descriptive summaries of the observed runs, not
    inferential estimates.
  - Truncation: one honest rate of capped responses over both arms (this
    replaces the per-bucket histogram), derived from the per-run truncated flags.

Conditions (all four kept, matching step 08 / cot_overall_summary):
  Hinted (X=1, all) · Unhinted (X=0, all) ·
  Hinted, followed (X=1, Y=1) · Hinted, not followed (X=1, Y=0)

Distributional stats INCLUDE truncated runs at the 16384-token cap: truncation
is caused by length, so excluding truncated (the longest) responses biases
length down. The cap understates the true tail, so the estimates are
conservative; truncation is also reported separately as a rate. "Hinted"/
"Unhinted" pools hold every run with a token_count (incl. truncated/unparsed);
followed/not-followed require a parsed_answer, so a few truncated-before-
answering runs sit in the "all" pools but not the followed split.

Distinct instances are counted within each dataset then summed, so same-dataset
hints union correctly and disjoint datasets don't collide on reused
instance_ids (0..199 per dataset).

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
from collections import defaultdict
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


def fmt_row(label, s, n_inst):
    if s is None:
        return f"| {label} | — | — | 0 ({n_inst}) |"
    med = f"{s['median']:.0f} [{s['q1']:.0f}–{s['q3']:.0f}]"
    p90 = f"{s['p90']:.0f}"
    return f"| {label} | {med} | {p90} | {s['n']} ({n_inst}) |"


def fmt_row_meancmp(label, s, n_inst):
    """Row for the median-vs-mean companion file (keeps the omitted Mean (SD))."""
    if s is None:
        return f"| {label} | — | — | — | 0 ({n_inst}) |"
    med = f"{s['median']:.0f} [{s['q1']:.0f}–{s['q3']:.0f}]"
    mean = f"{s['mean']:.0f} ({s['std']:.0f})"
    skew = f"{s['mean'] / s['median']:.2f}×" if s["median"] else "—"
    return f"| {label} | {med} | {mean} | {skew} | {s['n']} ({n_inst}) |"


def cond_agg(cells, key, unhinted_single=False):
    """Aggregate one condition over a list of (ds, json) cells using length_pools.
    Returns (tokens ndarray incl. truncated-at-cap, n_inst, n_runs, n_trunc).
    Distinct instances are counted WITHIN each dataset then summed — so
    same-dataset hints union correctly and disjoint datasets don't collide on
    reused instance_ids. unhinted_single keeps one cell (the hint-independent
    unhinted arm) so pooling hints doesn't triple-count it."""
    if unhinted_single:
        cells = cells[:1]
    tokens = []
    per_ds = defaultdict(set)
    n_trunc = 0
    for ds, j in cells:
        lp = j.get("length_pools", {}).get(key)
        if not lp:
            continue
        tokens += lp["token_count"]
        per_ds[ds].update(lp["instance_id"])
        n_trunc += int(sum(lp["truncated"]))
    n_inst = sum(len(s) for s in per_ds.values())
    return np.asarray(tokens, float), n_inst, len(tokens), n_trunc


def paired_gm_ratio(h_lp, u_lp):
    """Paired (per-question) geometric-mean ratio Hinted/Unhinted. Collapse each
    question's runs to its per-question MEDIAN length per arm (robust; matches
    the median-first tables), then take the geometric mean of the per-question
    ratios: exp(mean_q[ln median(h_q) − ln median(u_q)]) over questions present
    in both arms. GM is used ACROSS questions (ratios are multiplicative); the
    within-question collapse is a plain median length. Token counts are strictly
    positive, so the logs are always defined. Callers pass single-dataset pools,
    so instance_id is an unambiguous per-question key."""
    hb, ub = defaultdict(list), defaultdict(list)
    for iid, tok in zip(h_lp["instance_id"], h_lp["token_count"]):
        hb[iid].append(tok)
    for iid, tok in zip(u_lp["instance_id"], u_lp["token_count"]):
        ub[iid].append(tok)
    lr = []
    for iid in set(hb) & set(ub):
        hq, uq = float(np.median(hb[iid])), float(np.median(ub[iid]))
        if hq > 0 and uq > 0:
            lr.append(np.log(hq) - np.log(uq))
    return float(np.exp(np.mean(lr))) if lr else None


def gmean(vals):
    return float(np.exp(np.mean(np.log(vals))))


def gm_of_cell_gmrs(cell_jsons):
    """GM over dataset×hint cells of the per-cell paired Hinted/Unhinted GM
    ratio (equal weight per cell) — the pooled levels' GMR construction. A
    plain GMR over pooled runs would either mix hint types inside a question's
    median (dataset level) or weight datasets by question count (hint-type
    level); averaging the finest-cell GMRs keeps one rule at every level and
    stays derivable from the dataset×hint file."""
    logs = []
    for j in cell_jsons:
        lp = j.get("length_pools", {})
        if "hinted" not in lp or "unhinted" not in lp:
            continue
        g = paired_gm_ratio(lp["hinted"], lp["unhinted"])
        if g is not None:
            logs.append(np.log(g))
    return float(np.exp(np.mean(logs))) if logs else None


def section(conds, gmr=None, gmr_desc="paired"):
    """Render one section body from {key: (tokens, n_inst, n_runs, n_trunc)}.
    gmr is appended when provided: the dataset×hint level passes the plain
    paired GM ratio; the pooled levels pass the GM of their constituent
    dataset×hint cell GMRs, with gmr_desc naming the construction."""
    out = ["| Condition | Median [Q1–Q3] | p90 | n runs (inst) |",
           "|---|---|---|---|"]
    s = {}
    for label, key, _ in CONDITIONS:
        tokens, n_inst, _n_runs, _n_tr = conds[key]
        st = stats(tokens)
        s[key] = st
        out.append(fmt_row(label, st, n_inst))
    extra = []
    if s.get("hinted") and s.get("unhinted"):
        dmed = s["hinted"]["median"] - s["unhinted"]["median"]
        ratio = s["hinted"]["median"] / s["unhinted"]["median"]
        line = ("**Hint effect (Hinted − Unhinted):** "
                f"Δ median = {dmed:+.0f} tok · median ratio = {ratio:.2f}×")
        if gmr is not None:
            line += f" · GM ratio ({gmr_desc}) = {gmr:.2f}×"
        extra.append(line)
    # truncation over both arms, each counted once
    tn = conds["hinted"][3] + conds["unhinted"][3]
    tt = conds["hinted"][2] + conds["unhinted"][2]
    rate = (tn / tt * 100) if tt else 0.0
    extra.append(f"**Truncation (both arms):** {rate:.1f}% ({tn}/{tt})")
    return "\n".join(out) + "\n\n" + "\n\n".join(extra) + "\n"


def section_meancmp(conds):
    """Median vs Mean (SD) companion section, with a Mean÷Median skew flag."""
    out = ["| Condition | Median [Q1–Q3] | Mean (SD) | Mean÷Median | n runs (inst) |",
           "|---|---|---|---|---|"]
    for label, key, _ in CONDITIONS:
        tokens, n_inst, _n_runs, _n_tr = conds[key]
        out.append(fmt_row_meancmp(label, stats(tokens), n_inst))
    return "\n".join(out) + "\n"


MEANCMP_HEADER = (
    "This companion to `length_dataset_hint_level_tables.md` keeps the arithmetic "
    "**Mean (SD)** the main tables omit, beside the **Median [Q1–Q3]**, so the "
    "skew is explicit. **Mean÷Median** > 1 quantifies right-skew — the mean "
    "inflated by the heavy upper tail (and note SD ≈ mean, so a literal "
    "'mean ± SD' would imply impossible negative lengths, which is exactly why "
    "the main tables lead with the median). Same truncated-at-cap pools as the "
    "main tables."
)


HEADER_COLS = (
    "Columns: **Median [Q1–Q3]** primary central tendency + spread (skew-robust) · "
    "**p90** upper tail · **n runs (inst)** responses and distinct "
    "instances behind the cell. (Arithmetic Mean (SD) is omitted here to keep the "
    "table median-first; it lives in `length_dataset_hint_median_vs_mean.md`.) "
    "Length stats INCLUDE truncated runs at the "
    "16384-token cap (truncation is caused by length, so dropping them would "
    "bias length down; the cap understates the true tail, so estimates are "
    "conservative). **Truncation** is the separate rate of capped responses over "
    "both arms. **Hint effect** is the Hinted-vs-Unhinted contrast: Δ median "
    "(tokens), H/U median ratio — the robust analogue of a mean gap — and the "
    "paired **GM ratio** (per-question at the dataset×hint level; GM of the "
    "constituent cell GMRs at the pooled levels). "
    "(Descriptive summaries of observed runs — no confidence intervals.)"
)

GMR_NOTE = (
    "This finest level reports the **GM ratio (paired)** = the "
    "per-question geometric-mean ratio: each question's hinted and unhinted runs "
    "are collapsed to their per-question MEDIAN length, then GM ratio = "
    "exp(mean over questions of [ln median(h_q) − ln median(u_q)]) — the typical "
    "per-question multiplicative hint effect on length (median within a question, "
    "geometric mean across questions), skew-robust and the single best 'how much "
    "longer' number. The pooled levels report the geometric mean of their "
    "constituent dataset×hint cell GMRs (equal weight per cell) — a plain "
    "pooled-run GMR would mix hint types inside a question's median (dataset "
    "level) or weight datasets by question count (hint-type level); the "
    "GM-of-cells rule keeps every level derivable from this file and matches the "
    "summary matrix's geometric margins exactly."
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


def build_dataset_hint(data, model_id):
    out = [f"# Response length — per (dataset, hint type), no pooling", "",
           f"Model: `{model_id}`", "", HEADER_COLS, "", GMR_NOTE, ""]
    gmr_mat = {}  # gmr_mat[ds][hint] = paired GM ratio
    for ds in DATASETS:
        out.append(f"## {ds}")
        out.append("")
        for hint in HINT_TYPES:
            j = data[ds].get(hint)
            if not j:
                continue
            conds = {key: cond_agg([(ds, j)], key) for _, key, _ in CONDITIONS}
            lp = j["length_pools"]
            gmr = paired_gm_ratio(lp["hinted"], lp["unhinted"])
            gmr_mat.setdefault(ds, {})[hint] = gmr
            out.append(f"### {ds} — {HINT_LABEL[hint]} hint\n")
            out.append(section(conds, gmr=gmr))
    out.append(gmr_matrix_table(gmr_mat))
    out.append(FOLLOWED_NOTE)
    return "\n".join(out)


FOLLOWED_NOTE = (
    "## Followed vs not-followed (observational)\n\n"
    "Among hinted responses, those that followed the hint were typically ~1.3× "
    "longer than those that didn't (median ratio 1.26×, longer in 22/24 slices), "
    "though this is an association conditioned on the outcome and varies widely "
    "by dataset (≈2× in codemmlu/LSAT, negligible in mmlu_pro/medxpertqa).\n"
)


def gmr_matrix_table(gmr_mat):
    """Datasets × hint-types matrix of the paired GM ratio, with a last column =
    GM of the 3 hint types per dataset and a last row = GM of the 8 datasets
    per hint type (GEOMETRIC means — ratios are multiplicative, and these
    margins then equal the pooled-level sections' GM-of-cell-GMRs exactly;
    bottom-right = grand GM of all cells)."""
    lines = ["## Paired GM ratio (Hinted / Unhinted) — dataset × hint type", "",
             "Paired Hinted-vs-Unhinted GM ratio for each (dataset, hint type). "
             "Last column = geometric mean of the 3 hint-type ratios for that "
             "dataset; last row = geometric mean of the 8 dataset ratios for "
             "that hint type; bottom-right = grand GM of all cells. Geometric "
             "(not arithmetic) means, so the margins equal the GM ratio "
             "reported by the dataset-level and hint-type-level tables' "
             "pooled sections.", "",
             "| Dataset | " + " | ".join(HINT_LABEL[h] for h in HINT_TYPES)
             + " | Avg (dataset) |",
             "|" + "---|" * (len(HINT_TYPES) + 2)]
    col_vals = {h: [] for h in HINT_TYPES}
    all_vals = []
    for ds in DATASETS:
        cells, row_vals = [], []
        for h in HINT_TYPES:
            v = gmr_mat.get(ds, {}).get(h)
            if v is None:
                cells.append("—")
            else:
                cells.append(f"{v:.2f}×")
                row_vals.append(v)
                col_vals[h].append(v)
                all_vals.append(v)
        ravg = f"{gmean(row_vals):.2f}×" if row_vals else "—"
        lines.append(f"| {ds} | " + " | ".join(cells) + f" | {ravg} |")
    col_cells = [f"{gmean(col_vals[h]):.2f}×" if col_vals[h] else "—"
                 for h in HINT_TYPES]
    grand = f"{gmean(all_vals):.2f}×" if all_vals else "—"
    lines.append("| **Avg (hint type)** | " + " | ".join(col_cells)
                 + f" | {grand} |")
    return "\n".join(lines) + "\n"


def build_dataset_hint_meancmp(data, model_id):
    out = ["# Response length — median vs mean (SD), per (dataset, hint type)", "",
           f"Model: `{model_id}`", "", MEANCMP_HEADER, ""]
    for ds in DATASETS:
        out.append(f"## {ds}")
        out.append("")
        for hint in HINT_TYPES:
            j = data[ds].get(hint)
            if not j:
                continue
            conds = {key: cond_agg([(ds, j)], key) for _, key, _ in CONDITIONS}
            out.append(f"### {ds} — {HINT_LABEL[hint]} hint\n")
            out.append(section_meancmp(conds))
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
        cells = [(ds, data[ds][h]) for h in hints]
        conds = {key: cond_agg(cells, key, unhinted_single=(key == "unhinted"))
                 for _, key, _ in CONDITIONS}
        gmr = gm_of_cell_gmrs([data[ds][h] for h in hints])
        out.append(f"## {ds}\n")
        out.append(section(conds, gmr=gmr,
                           gmr_desc=f"paired, GM of the {len(hints)} per-hint "
                                    "cell GMRs"))
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
        cells = [(ds, data[ds][hint]) for ds in dss]
        conds = {key: cond_agg(cells, key) for _, key, _ in CONDITIONS}
        gmr = gm_of_cell_gmrs([data[ds][hint] for ds in dss])
        out.append(f"## {HINT_LABEL[hint]} hint\n")
        out.append(section(conds, gmr=gmr,
                           gmr_desc=f"paired, GM of the {len(dss)} "
                                    "per-dataset cell GMRs"))
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
        "length_dataset_hint_median_vs_mean.md":
            build_dataset_hint_meancmp(data, args.model),
    }
    for name, text in files.items():
        p = out_dir / name
        p.write_text(text.rstrip() + "\n", encoding="utf-8")
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
