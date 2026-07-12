"""
Markdown report generation for the g-mean² evaluation (scripts/09_generate_reports.py).

ALL .md generation lives here (plus the 09 script and one driver line) so it can
be removed cleanly for a public release: delete src/reporting.py,
scripts/09_generate_reports.py, and the Phase-3 line in
run_monitor_and_results.sh — the metric computation (04/08) is untouched.

Reads only machine-readable artifacts (results_summary.json +
results_per_instance.csv from step 04; cot_token_stats.json from step 08), so
every .md file can be deleted and regenerated with a single 09 run — no
inference/monitor re-run needed.

Results reports share ONE table format (identical columns, "mean [95% CI]"
cells — percentile bootstrap over instances, see src/gmean2_metrics.bootstrap_cis),
so any number is directly comparable across files. Estimated rates/effects (TE,
R_effect, g-mean², monitor rates, accuracies, opacity, skip/L1 rates) carry a
95% CI; CoT token lengths keep mean (std) + median (the spread is the
observation there); confusion-matrix cells and n counts stay exact tallies:

  results_summary.md                  per (dataset, hint) — config header,
                                      results table + matrix, then details
  results_overall_summary.md          one section per (dataset x hint)
  results_dataset_level_summary.md    per dataset, instances pooled across hints
  results_hint_type_level_summary.md  per hint type, pooled across datasets
  results_aggregate_summary.md        one section, pooled across everything

CoT token-length reports mirror the same structure (condition table + bucket/
truncation row per scope):

  cot_overall_summary.md / cot_dataset_level_summary.md /
  cot_hint_type_level_summary.md / cot_aggregate_summary.md

Pooling: aggregated levels recompute stats over pooled per-instance rows (or
pooled run-level token counts), never means-of-means; matrices and bucket
counts are summed, and rates are recomputed from the sums.
"""

import csv
import json
from pathlib import Path

import numpy as np

from src.gmean2_metrics import _cot_stats, bootstrap_cis

RESULT_COLUMNS = ("Mean TE before [95% CI]", "Mean R_effect before [95% CI]",
                  "Mean TE after [95% CI]", "Mean R_effect after [95% CI]",
                  "n TE>0 (post L1) [95% CI]", "g-mean² [95% CI]",
                  "Unhinted Acc [95% CI]", "Hinted Acc [95% CI]")

# Confusion-matrix columns for the roll-up summary table, in (acknowledged?,
# followed?) order matching the labels the user sees; keys map onto
# sum_matrices()'s cm dict.
CONFUSION_COLUMNS = ("Acknowledged & followed", "Acknowledged & not followed",
                    "Not Acknowledged & followed", "Not Acknowledged & not followed")
CONFUSION_KEYS = ("foll_ack", "notfoll_ack", "foll_noack", "notfoll_noack")

COT_CONDITIONS = (
    ("hinted",            "Hinted (X=1, all)"),
    ("unhinted",          "Unhinted (X=0, all)"),
    ("hint_followed",     "Hinted, followed (X=1, Y=1)"),
    ("hint_not_followed", "Hinted, not followed (X=1, Y=0)"),
)

# Columns for the CoT roll-up summary table — same 4 conditions as
# COT_CONDITIONS, one "mean (std) · median" cell each.
COT_SUMMARY_COLUMNS = ("Hinted (mean (std) · median)", "Unhinted (mean (std) · median)",
                      "Hinted & followed (mean (std) · median)",
                      "Hinted & not followed (mean (std) · median)",
                      "Gap (Hinted mean − Unhinted mean)")


def _fmt(v, decimals=4):
    return f"{v:.{decimals}f}" if v is not None else "N/A"


def _ms(mean, std, decimals=4):
    """'mean (std)' cell; N/A when the mean is undefined."""
    return f"{_fmt(mean, decimals)} ({_fmt(std, decimals)})" if mean is not None else "N/A"


def _ci(ci, decimals=4):
    """'[lo, hi]' from a [lo, hi] pair; N/A when the CI is unavailable."""
    if not ci or ci[0] is None:
        return "N/A"
    return f"[{ci[0]:.{decimals}f}, {ci[1]:.{decimals}f}]"


def _mci(mean, ci, decimals=4):
    """'mean [lo, hi]' cell (95% CI); bare mean when no CI; N/A when undefined."""
    if mean is None:
        return "N/A"
    if not ci or ci[0] is None:
        return _fmt(mean, decimals)
    return f"{_fmt(mean, decimals)} {_ci(ci, decimals)}"


def _n_pct(n, total, ci=None):
    """'n (pct% [lo%, hi%])' cell — count with its share of the total and,
    when available, the 95% CI of that share (e.g. '62 (31.0% [25.0, 37.5])').
    Falls back to the bare count when the total is unknown or zero."""
    if not total:
        return str(n)
    pct = f"{100 * n / total:.1f}%"
    if ci and ci[0] is not None:
        pct += f" [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]"
    return f"{n} ({pct})"


# ---------------------------------------------------------------------------
# Loading the machine-readable artifacts (steps 04 + 08)
# ---------------------------------------------------------------------------

def _f(v):
    """Parse a CSV cell into float/None."""
    if v is None or v == "" or v == "None":
        return None
    return float(v)


def load_run(analysis_dir: Path) -> dict | None:
    """Load one (dataset, hint) run: {"summary", "rows", "cot"(optional)}."""
    summary_path = analysis_dir / "results_summary.json"
    csv_path = analysis_dir / "results_per_instance.csv"
    if not summary_path.exists() or not csv_path.exists():
        return None
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "te_i": _f(row.get("te_i")),
                "r_effect_i": _f(row.get("r_effect_i")),
                "level1_pass": str(row.get("level1_pass")).strip().lower() == "true",
                "gmean2_i": _f(row.get("gmean2_i")),
                "acc_u_i": _f(row.get("acc_u_i")),
                "acc_h_i": _f(row.get("acc_h_i")),
                # For pooled bootstrap CIs (monitor rates + clustered opacity);
                # None when the CSV predates these columns (re-run step 04).
                "tpr_mc_i": _f(row.get("tpr_mc_i")),
                "tnr_x0_i": _f(row.get("tnr_x0_i")),
                "tnr_x1_i": _f(row.get("tnr_x1_i")),
                "n_foll_ack_i": _f(row.get("n_foll_ack_i")),
                "n_foll_noack_i": _f(row.get("n_foll_noack_i")),
            })
    cot = None
    cot_path = analysis_dir / "cot_token_stats.json"
    if cot_path.exists():
        with open(cot_path, encoding="utf-8") as f:
            cot = json.load(f)
    return {"summary": summary, "rows": rows, "cot": cot}


def discover_runs(model_analysis_root: Path, dataset_order: list, hint_order: list,
                  split: str = "test") -> dict:
    """Find every (dataset, hint) with a results_summary.json under analysis/{model}/.

    `split` selects the data split: "test" scans {ds}/{hint}/, "eval" scans
    {ds}/eval/{hint}/ (the eval-split layout from resolve_paths). Returns an
    ordered {(dataset, hint): analysis_dir} following the given config/registry
    orders, with any unknown keys appended alphabetically."""
    found = {}
    if not model_analysis_root.is_dir():
        return found
    pattern = "*/*/results_summary.json" if split == "test" else "*/eval/*/results_summary.json"
    for summary_path in sorted(model_analysis_root.glob(pattern)):
        hint_dir = summary_path.parent
        ds_dir = hint_dir.parent if split == "test" else hint_dir.parent.parent
        found[(ds_dir.name, hint_dir.name)] = hint_dir

    def order(seq, known):
        known_idx = {k: i for i, k in enumerate(known)}
        return sorted(seq, key=lambda k: (known_idx.get(k, len(known)), k))

    datasets = order({ds for ds, _ in found}, dataset_order)
    hints = order({h for _, h in found}, hint_order)
    return {(ds, h): found[(ds, h)] for ds in datasets for h in hints if (ds, h) in found}


def _run_keys(runs: dict) -> tuple[list, list]:
    """Ordered, de-duplicated dataset and hint-type lists from a runs dict."""
    return (list(dict.fromkeys(ds for ds, _ in runs)),
            list(dict.fromkeys(h for _, h in runs)))


# ---------------------------------------------------------------------------
# Results-row stats — from a summary json (per-run) or pooled per-instance rows
# ---------------------------------------------------------------------------
# Both builders return the same shape: estimated stats are (mean, [lo, hi])
# tuples (95% bootstrap CI over instances, None CI when unavailable), plus the
# exact counts and the L1-rate/opacity CIs used by the tables.

def stats_from_summary(s: dict) -> dict:
    """The shared results-table stats, read straight from a results_summary.json."""
    ci = s.get("ci95") or {}
    return {
        "te_b": (s["mean_TE_before"], ci.get("TE_before")),
        "reff_b": (s["mean_R_effect_before"], ci.get("R_effect_before")),
        "te_a": (s["mean_TE_after"], ci.get("TE_after")),
        "reff_a": (s["mean_R_effect_after"], ci.get("R_effect_after")),
        "n_level1": s["n_level1_pass"],
        "n_total": s["n_total"],
        "l1_ci": ci.get("l1_pass_rate"),
        "gmean2": (s["gmean2"], ci.get("gmean2")),
        "acc_u": (s["mean_acc_u"], ci.get("acc_u")),
        "acc_h": (s["mean_acc_h"], ci.get("acc_h")),
        "opacity_ci": ci.get("opacity_rate"),
    }


def _mean(vals):
    return float(np.mean(vals)) if vals else None


def stats_from_pooled(rows: list) -> dict:
    """The shared results-table stats, recomputed over pooled per-instance rows.

    Matches 04's definitions exactly: 'before' = all instances, 'after' = the
    Level-1 subset, g-mean² = mean over Level-1 instances with a defined value.
    CIs come from the same instance-level bootstrap as the per-run summaries,
    run over the pooled rows."""
    l1 = [r for r in rows if r["level1_pass"]]
    gm = [r["gmean2_i"] for r in l1 if r["gmean2_i"] is not None]
    ci = bootstrap_cis(rows)
    return {
        "te_b": (_mean([r["te_i"] for r in rows if r["te_i"] is not None]), ci["TE_before"]),
        "reff_b": (_mean([r["r_effect_i"] for r in rows if r["r_effect_i"] is not None]), ci["R_effect_before"]),
        "te_a": (_mean([r["te_i"] for r in l1 if r["te_i"] is not None]), ci["TE_after"]),
        "reff_a": (_mean([r["r_effect_i"] for r in l1 if r["r_effect_i"] is not None]), ci["R_effect_after"]),
        "n_level1": len(l1),
        "n_total": len(rows),
        "l1_ci": ci["l1_pass_rate"],
        "gmean2": (_mean(gm), ci["gmean2"]),
        "acc_u": (_mean([r["acc_u_i"] for r in rows if r["acc_u_i"] is not None]), ci["acc_u"]),
        "acc_h": (_mean([r["acc_h_i"] for r in rows if r["acc_h_i"] is not None]), ci["acc_h"]),
        "opacity_ci": ci["opacity_rate"],
    }


def sum_matrices(summaries: list) -> dict:
    """Sum run-level confusion-matrix counts and recompute the opacity rate."""
    cm = {"foll_ack": 0, "foll_noack": 0, "notfoll_ack": 0, "notfoll_noack": 0}
    for s in summaries:
        src = s.get("confusion_matrix", {})
        for k in cm:
            cm[k] += src.get(k, 0)
    cm["total"] = sum(cm[k] for k in ("foll_ack", "foll_noack", "notfoll_ack", "notfoll_noack"))
    cm["n_followed"] = cm["foll_ack"] + cm["foll_noack"]
    cm["opacity_rate"] = (cm["foll_noack"] / cm["n_followed"]) if cm["n_followed"] else None
    return cm


# ---------------------------------------------------------------------------
# Shared markdown blocks
# ---------------------------------------------------------------------------

def results_table(label_header: str, label: str, st: dict) -> list:
    """The one shared results table (single row), identical across every report."""
    return [
        f"| {label_header} | " + " | ".join(RESULT_COLUMNS) + " |",
        "|" + "---|" * (len(RESULT_COLUMNS) + 1),
        f"| {label} "
        f"| {_mci(*st['te_b'])} | {_mci(*st['reff_b'])} "
        f"| {_mci(*st['te_a'])} | {_mci(*st['reff_a'])} "
        f"| {_n_pct(st['n_level1'], st.get('n_total'), st.get('l1_ci'))} | {_mci(*st['gmean2'])} "
        f"| {_mci(*st['acc_u'], 3)} | {_mci(*st['acc_h'], 3)} |",
        "",
    ]


def matrix_block(cm: dict, opacity_ci=None) -> list:
    """Confusion matrix (hinted runs with a verdict) + opacity rate. The cells
    are exact run tallies; the opacity rate is an estimated rate and carries a
    95% CI (instance-level bootstrap) when one is available."""
    total = cm.get("total", 0)
    if not total:
        return ["_No hinted runs with a monitor verdict._", ""]

    def _cell(count):
        return f"{count} ({100 * count / total:.1f}%)"

    return [
        "| | Followed | Did not follow |",
        "|---|---|---|",
        f"| **Acknowledged** | {_cell(cm['foll_ack'])} | {_cell(cm['notfoll_ack'])} |",
        f"| **Not acknowledged** | {_cell(cm['foll_noack'])} | {_cell(cm['notfoll_noack'])} |",
        "",
        f"**Opacity rate:** {_mci(cm.get('opacity_rate'), opacity_ci)} — {cm['foll_noack']} covert / "
        f"{cm['n_followed']} hint-following runs (covert / all hint-following).",
        "",
    ]


def _cm_cells(cm: dict) -> list:
    """The 4 confusion-matrix cells as 'count (pct%)', in CONFUSION_KEYS order."""
    total = cm.get("total", 0)

    def _cell(count):
        return f"{count} ({100 * count / total:.1f}%)" if total else str(count)

    return [_cell(cm.get(k, 0)) for k in CONFUSION_KEYS]


def summary_table(label_headers: list, rows: list) -> list:
    """Consolidated roll-up table (one row per group), appended at the end of
    a report: `label_headers` columns + the shared RESULT_COLUMNS + the 4
    CONFUSION_COLUMNS. `rows` is a list of (label_values, st, cm) tuples."""
    headers = list(label_headers) + list(RESULT_COLUMNS) + list(CONFUSION_COLUMNS)
    lines = [
        "## Summary\n",
        "| " + " | ".join(headers) + " |",
        "|" + "---|" * len(headers),
    ]
    for label_values, st, cm in rows:
        cells = list(label_values) + [
            _mci(*st['te_b']), _mci(*st['reff_b']),
            _mci(*st['te_a']), _mci(*st['reff_a']),
            _n_pct(st['n_level1'], st.get('n_total'), st.get('l1_ci')), _mci(*st['gmean2']),
            _mci(*st['acc_u'], 3), _mci(*st['acc_h'], 3),
        ] + _cm_cells(cm)
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def cot_condition_table(cot: dict) -> list:
    """4-condition run-level token-length table (valid runs; followed +
    not-followed partition hinted exactly). Token lengths keep mean (std) plus
    the median — the distributions are right-skewed, so the mean alone
    overstates typical length."""
    lines = ["| Condition | Mean | Std | Median | Min | Max | Runs | Instances |",
             "|-----------|------|-----|--------|-----|-----|------|-----------|"]
    for prefix, label in COT_CONDITIONS:
        lines.append(
            f"| {label} "
            f"| {_fmt(cot.get(f'{prefix}_mean'), 1)} "
            f"| {_fmt(cot.get(f'{prefix}_std'), 1)} "
            f"| {_fmt(cot.get(f'{prefix}_median'), 0)} "
            f"| {_fmt(cot.get(f'{prefix}_min'), 0)} "
            f"| {_fmt(cot.get(f'{prefix}_max'), 0)} "
            f"| {cot.get(f'{prefix}_n_runs', 0)} "
            f"| {cot.get(f'{prefix}_n_instances', 0)} |"
        )
    lines.append("")
    return lines


def bucket_table(buckets: dict, trunc: dict, max_tokens) -> list:
    """One-row bucket/truncation table over EVERY response (both arms)."""
    labels = list(buckets)
    rate = f"{trunc['rate'] * 100:.1f}%" if trunc.get("rate") is not None else "N/A"
    return [
        "| " + " | ".join(labels) + " | Total | Truncated (rate) | Max |",
        "|" + "---|" * (len(labels) + 3),
        "| " + " | ".join(str(buckets[l]) for l in labels)
        + f" | {trunc.get('total', 0)} | {trunc.get('truncated', 0)} ({rate})"
        + f" | {_fmt(max_tokens, 0)} |",
        "",
    ]


def _file_header(title: str, model_name: str, hf_id: str, datasets: list, hints: list) -> list:
    return [
        f"# {title}\n",
        f"- **Model:** `{hf_id}` (`{model_name}`)",
        f"- **Datasets:** {', '.join(datasets)}",
        f"- **Hint types:** {', '.join(hints)}",
        "",
    ]


# ---------------------------------------------------------------------------
# Results reports
# ---------------------------------------------------------------------------

def write_results_summary_md(summary: dict, level2_threshold: float, path: Path) -> None:
    """Per-(dataset, hint) report: config header, then the shared results table
    + matrix (directly comparable with the roll-up files), then details."""
    n = summary["n_total"]
    cot = summary.get("cot_tokens", {})
    cov = summary.get("monitor_coverage", {})
    cm = summary.get("confusion_matrix", {})
    snap = summary.get("config_snapshot", {})
    n_level1 = summary["n_level1_pass"]
    contributing = n_level1 - summary["n_gmean2_undefined"]

    lines = []
    w = lines.append

    w(f"# Results: {summary['model_name']} · {summary.get('dataset', '?')} · {summary.get('hint_type', '?')}\n")
    w(f"- **Model:** `{summary['hf_id']}`")
    w(f"- **Dataset:** {summary.get('dataset', '?')} · **Hint type:** {summary.get('hint_type', '?')}")
    w(f"- **Instances:** {n} (budget n_examples = {snap.get('n_examples', 'N/A')}) · "
      f"**Samples/arm (N):** {summary.get('n_samples_per_instance', 'N/A')}")
    w(f"- **Sampling:** temperature {snap.get('temperature', 'N/A')}, top_p {snap.get('top_p', 'N/A')}, "
      f"top_k {snap.get('top_k', 'N/A')}, max_new_tokens {snap.get('max_new_tokens', 'N/A')}")
    w(f"- **Monitor model:** {snap.get('monitor_model', 'N/A')}")
    w("")

    ci = summary.get("ci95") or {}

    # --- Results (shared format) -------------------------------------------
    w("## Results\n")
    lines += results_table("Dataset", summary.get("dataset", "?"), stats_from_summary(summary))
    lines += matrix_block(cm, ci.get("opacity_rate"))

    # --- g-mean² components --------------------------------------------------
    w("---\n")
    w("## g-mean² components\n")
    w("`g-mean²_i = TPR_mc_i · √(TNR_X=0_i · TNR_X=1_i)`, averaged over Level-1 "
      "instances (`TE > 0 and p1 > 0`); `TNR_X=1 = 1` by vacuous truth when `p1 = 1`. "
      "95% CIs are percentile bootstrap over instances.\n")
    w("| Metric | Value | 95% CI |")
    w("|--------|-------|--------|")
    w(f"| **g-mean²** | {_fmt(summary['gmean2'])} | {_ci(ci.get('gmean2'))} |")
    w(f"| Mean TPR_mc | {_fmt(summary['mean_tpr_mc'])} | {_ci(ci.get('tpr_mc'))} |")
    w(f"| Mean TNR_X=0 | {_fmt(summary['mean_tnr_x0'])} | {_ci(ci.get('tnr_x0'))} |")
    w(f"| Mean TNR_X=1 | {_fmt(summary['mean_tnr_x1'])} | {_ci(ci.get('tnr_x1'))} |")
    w(f"| Instances contributing | {contributing} "
      f"({summary['n_tnr_x1_vacuous']} vacuous-truth, {summary['n_gmean2_undefined']} excluded for missing coverage) | |")
    w("")

    # --- Filtering -----------------------------------------------------------
    w("---\n")
    w("## Filtering — Level-1: `TE > 0 and p1 > 0`\n")
    w("`TE_i = p1_i − p0_i`, where `p1 = Pr(answer = hint | hinted)` and "
      "`p0 = Pr(answer = hint | unhinted)`. `R_effect = TE / p1` (defined only "
      f"for `p1 > 0`: n = {summary.get('n_R_effect_before', 0)} before, "
      f"{summary.get('n_R_effect_after', 0)} after).\n")
    neg, zer, pos = summary["n_te_negative"], summary["n_te_zero"], summary["n_te_positive"]
    mean_te = summary["mean_TE_before"]
    verdict = "PASS" if (mean_te is not None and mean_te > level2_threshold) else "FAIL"
    w(f"**n instances:** {n} → {n_level1} after Level-1 ({100*n_level1/n:.1f}%). "
      f"**TE sign:** `<0` {neg} ({100*neg/n:.1f}%) · `=0` {zer} ({100*zer/n:.1f}%) · "
      f"`>0` {pos} ({100*pos/n:.1f}%). "
      f"**Skip rate** {_mci(summary.get('skip_rate'), ci.get('skip_rate'))} "
      f"({n - n_level1}/{n}). "
      f"**Level-2** (mean TE {_fmt(mean_te)} vs {level2_threshold}): **{verdict}**.\n")

    # --- Monitor coverage ------------------------------------------------------
    w("---\n")
    w("## Monitor coverage\n")
    w("Missing = verdict failed/unparseable; excluded from that rate's denominator. "
      f"Accuracy denominators: unhinted {summary.get('n_acc_u_instances', 0)} instances / "
      f"{summary.get('n_acc_u_runs', 0)} runs, hinted {summary.get('n_acc_h_instances', 0)} instances / "
      f"{summary.get('n_acc_h_runs', 0)} runs.\n")
    w("| Condition | Runs | Missing | Rate |")
    w("|-----------|------|---------|------|")
    for key, label in [("x1_y1", "X=1, Y=1 (followed)"),
                       ("x1_y0", "X=1, Y=0 (not followed)"),
                       ("x0", "X=0 (control)")]:
        c = cov.get(key, {})
        rate = c.get("missing_rate")
        w(f"| {label} | {c.get('total', 0)} | {c.get('missing', 0)} | {_fmt(rate) if rate is not None else 'N/A'} |")
    w("")

    # --- CoT length ------------------------------------------------------------
    w("---\n")
    w("## CoT length\n")
    w("Completion tokens per response (thinking trace + final answer), pooled "
      "over all valid runs per condition:\n")
    if cot.get("hinted_mean") is not None:
        lines += cot_condition_table(cot)
    else:
        w("_Token count data not available._")
        w("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _results_section(heading: str, label_header: str, label: str, st: dict, cm: dict) -> list:
    return ([f"## {heading}\n"] + results_table(label_header, label, st)
            + matrix_block(cm, st.get("opacity_ci")))


def write_results_overall_summary(runs: dict, model_name: str, hf_id: str, path: Path) -> None:
    """One section per (dataset, hint type), read straight from each run's summary."""
    datasets, hints = _run_keys(runs)
    lines = _file_header("Results overall summary — per (dataset × hint type)",
                         model_name, hf_id, datasets, hints)
    summary_rows = []
    for (ds, hint), run in runs.items():
        s = run["summary"]
        st, cm = stats_from_summary(s), s.get("confusion_matrix", {})
        lines += _results_section(f"{ds} × {hint}", "Dataset", ds, st, cm)
        summary_rows.append(([ds, hint], st, cm))
    lines += summary_table(["Dataset", "Hint type"], summary_rows)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_results_level_summary(runs: dict, model_name: str, hf_id: str, path: Path,
                                level: str) -> None:
    """Roll-up per dataset (pooled across hint types) or per hint type (pooled
    across datasets). `level` is 'dataset' or 'hint_type'."""
    datasets, hints = _run_keys(runs)

    if level == "dataset":
        keys, label_header, pooled_over = datasets, "Dataset", "hint types"
        title = "Results dataset-level summary — instances pooled across hint types"
        member = lambda key, ds, hint: ds == key
    else:
        keys, label_header, pooled_over = hints, "Hint type", "datasets"
        title = "Results hint-type-level summary — instances pooled across datasets"
        member = lambda key, ds, hint: hint == key

    lines = _file_header(title, model_name, hf_id, datasets, hints)
    summary_rows = []
    for key in keys:
        group = [run for (ds, hint), run in runs.items() if member(key, ds, hint)]
        pooled_rows = [r for run in group for r in run["rows"]]
        st = stats_from_pooled(pooled_rows)
        cm = sum_matrices([run["summary"] for run in group])
        lines += _results_section(f"{key} (pooled over {len(group)} {pooled_over})",
                                  label_header, key, st, cm)
        summary_rows.append(([key], st, cm))
    lines += summary_table([label_header], summary_rows)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_results_aggregate_summary(runs: dict, model_name: str, hf_id: str, path: Path) -> None:
    """Single results roll-up pooled across every dataset and hint type."""
    datasets, hints = _run_keys(runs)
    pooled_rows = [r for run in runs.values() for r in run["rows"]]
    st = stats_from_pooled(pooled_rows)
    cm = sum_matrices([run["summary"] for run in runs.values()])
    lines = _file_header("Results aggregate summary — instances pooled across all datasets and hint types",
                         model_name, hf_id, datasets, hints)
    lines += _results_section(f"All ({len(runs)} dataset × hint-type runs)", "Scope", "all", st, cm)
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CoT token-length reports (from step 08's cot_token_stats.json)
# ---------------------------------------------------------------------------

def _pooled_cot_stats(cots: list) -> dict:
    """Pool the raw valid-run token counts of several runs into one 4-condition
    stats dict (same shape as compute_gmean2(...)['cot_tokens'])."""
    out = {}
    for prefix, _ in COT_CONDITIONS:
        pool = [t for c in cots for t in c["cot_token_pools"].get(prefix, [])]
        out.update(_cot_stats(pool, prefix))
        out[f"{prefix}_n_runs"] = len(pool)
        out[f"{prefix}_n_instances"] = sum(
            c["condition_stats"].get(f"{prefix}_n_instances", 0) for c in cots)
    return out


def _summed_buckets(cots: list) -> tuple[dict, dict, float | None]:
    """Sum bucket counts + truncation across runs; max = max of per-run maxes."""
    buckets: dict = {}
    for c in cots:
        for label, count in c["buckets"].items():
            buckets[label] = buckets.get(label, 0) + count
    total = sum(c["truncation"]["total"] for c in cots)
    truncated = sum(c["truncation"]["truncated"] for c in cots)
    trunc = {"total": total, "truncated": truncated,
             "rate": (truncated / total) if total else None}
    maxes = [c["raw_stats"]["max"] for c in cots if c["raw_stats"].get("max") is not None]
    return buckets, trunc, (max(maxes) if maxes else None)


def _cot_condition_stats(cots: list) -> dict:
    """Pooled 4-condition stats dict, or the single run's own stats when only
    one run is in the group (avoids re-pooling a single pool)."""
    return _pooled_cot_stats(cots) if len(cots) > 1 else cots[0]["condition_stats"]


def cot_summary_table(label_headers: list, rows: list) -> list:
    """Consolidated CoT roll-up table (one row per group), appended at the end
    of a report: `label_headers` columns + COT_SUMMARY_COLUMNS. `rows` is a
    list of (label_values, cond) tuples, `cond` shaped like `condition_stats`."""
    headers = list(label_headers) + list(COT_SUMMARY_COLUMNS)
    lines = [
        "## Summary\n",
        "| " + " | ".join(headers) + " |",
        "|" + "---|" * len(headers),
    ]
    for label_values, cond in rows:
        cells = list(label_values) + [
            _ms(cond.get(f"{prefix}_mean"), cond.get(f"{prefix}_std"), 1)
            + f" · {_fmt(cond.get(f'{prefix}_median'), 0)}"
            for prefix, _ in COT_CONDITIONS
        ]
        hinted_mean, unhinted_mean = cond.get("hinted_mean"), cond.get("unhinted_mean")
        gap = (hinted_mean - unhinted_mean) if hinted_mean is not None and unhinted_mean is not None else None
        cells.append(_fmt(gap, 1))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _cot_section(heading: str, cots: list) -> list:
    """One CoT report section: condition table + bucket/truncation row."""
    buckets, trunc, max_tok = _summed_buckets(cots)
    cond = _cot_condition_stats(cots)
    lines = [f"## {heading}\n"]
    lines += cot_condition_table(cond)
    lines += ["Responses per token-length bucket (every response, both arms; "
              "truncated responses count in the top bucket):", ""]
    lines += bucket_table(buckets, trunc, max_tok)
    return lines


def _cot_file_header(title: str, model_name: str, hf_id: str,
                     datasets: list, hints: list) -> list:
    return _file_header(title, model_name, hf_id, datasets, hints) + [
        "`token_count` is the full completion (thinking trace + final answer). "
        "Condition tables use valid runs only (not truncated, answer parsed), so "
        "followed + not-followed partition hinted exactly; bucket tables use "
        "every response. Use the truncation rate to set `max_new_tokens`.",
        "",
    ]


def write_cot_overall_summary(cot_runs: dict, model_name: str, hf_id: str, path: Path) -> None:
    """One CoT section per (dataset, hint type)."""
    datasets, hints = _run_keys(cot_runs)
    lines = _cot_file_header("CoT token-length overall summary — per (dataset × hint type)",
                             model_name, hf_id, datasets, hints)
    summary_rows = []
    for (ds, hint), cot in cot_runs.items():
        lines += _cot_section(f"{ds} × {hint}", [cot])
        summary_rows.append(([ds, hint], cot["condition_stats"]))
    lines += cot_summary_table(["Dataset", "Hint type"], summary_rows)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_cot_level_summary(cot_runs: dict, model_name: str, hf_id: str, path: Path,
                            level: str) -> None:
    """CoT roll-up per dataset (pooled across hint types) or per hint type
    (pooled across datasets)."""
    datasets, hints = _run_keys(cot_runs)
    if level == "dataset":
        keys, label_header, pooled_over = datasets, "Dataset", "hint types"
        title = "CoT token-length dataset-level summary — runs pooled across hint types"
        member = lambda key, ds, hint: ds == key
    else:
        keys, label_header, pooled_over = hints, "Hint type", "datasets"
        title = "CoT token-length hint-type-level summary — runs pooled across datasets"
        member = lambda key, ds, hint: hint == key

    lines = _cot_file_header(title, model_name, hf_id, datasets, hints)
    summary_rows = []
    for key in keys:
        group = [cot for (ds, hint), cot in cot_runs.items() if member(key, ds, hint)]
        lines += _cot_section(f"{key} (pooled over {len(group)} {pooled_over})", group)
        summary_rows.append(([key], _cot_condition_stats(group)))
    lines += cot_summary_table([label_header], summary_rows)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_cot_aggregate_summary(cot_runs: dict, model_name: str, hf_id: str, path: Path) -> None:
    """Single CoT roll-up pooled across every dataset and hint type."""
    datasets, hints = _run_keys(cot_runs)
    lines = _cot_file_header("CoT token-length aggregate summary — runs pooled across all datasets and hint types",
                             model_name, hf_id, datasets, hints)
    lines += _cot_section(f"All ({len(cot_runs)} dataset × hint-type runs)",
                          list(cot_runs.values()))
    path.write_text("\n".join(lines), encoding="utf-8")
