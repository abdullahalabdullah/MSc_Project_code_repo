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

Results reports share ONE table format (identical columns, "mean (std)" cells),
so any number is directly comparable across files:

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

from src.gmean2_metrics import _cot_stats

RESULT_COLUMNS = ("Mean TE before (Std)", "Mean R_effect before (Std)",
                  "Mean TE after (Std)", "Mean R_effect after (Std)",
                  "n TE>0 (post L1)", "g-mean²", "Unhinted Acc (Std)",
                  "Hinted Acc (Std)")

COT_CONDITIONS = (
    ("hinted",            "Hinted (X=1, all)"),
    ("unhinted",          "Unhinted (X=0, all)"),
    ("hint_followed",     "Hinted, followed (X=1, Y=1)"),
    ("hint_not_followed", "Hinted, not followed (X=1, Y=0)"),
)


def _fmt(v, decimals=4):
    return f"{v:.{decimals}f}" if v is not None else "N/A"


def _ms(mean, std, decimals=4):
    """'mean (std)' cell; N/A when the mean is undefined."""
    return f"{_fmt(mean, decimals)} ({_fmt(std, decimals)})" if mean is not None else "N/A"


def _n_pct(n, total):
    """'n (pct%)' cell — count with its share of the total (e.g. '30 (60.0%)').
    Falls back to the bare count when the total is unknown or zero."""
    if not total:
        return str(n)
    return f"{n} ({100 * n / total:.1f}%)"


def _sample_std(vals):
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


def _mean_std(vals):
    return (float(np.mean(vals)), _sample_std(vals)) if vals else (None, None)


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
            })
    cot = None
    cot_path = analysis_dir / "cot_token_stats.json"
    if cot_path.exists():
        with open(cot_path, encoding="utf-8") as f:
            cot = json.load(f)
    return {"summary": summary, "rows": rows, "cot": cot}


def discover_runs(model_analysis_root: Path, dataset_order: list, hint_order: list) -> dict:
    """Find every (dataset, hint) with a results_summary.json under analysis/{model}/.

    Returns an ordered {(dataset, hint): analysis_dir} following the given
    config/registry orders, with any unknown keys appended alphabetically."""
    found = {}
    if not model_analysis_root.is_dir():
        return found
    for summary_path in sorted(model_analysis_root.glob("*/*/results_summary.json")):
        hint_dir = summary_path.parent
        found[(hint_dir.parent.name, hint_dir.name)] = hint_dir

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

def stats_from_summary(s: dict) -> dict:
    """The shared results-table stats, read straight from a results_summary.json."""
    return {
        "te_b": (s["mean_TE_before"], s["std_TE_before"]),
        "reff_b": (s["mean_R_effect_before"], s["std_R_effect_before"]),
        "te_a": (s["mean_TE_after"], s["std_TE_after"]),
        "reff_a": (s["mean_R_effect_after"], s["std_R_effect_after"]),
        "n_level1": s["n_level1_pass"],
        "n_total": s["n_total"],
        "gmean2": s["gmean2"],
        "acc_u": (s["mean_acc_u"], s["std_acc_u"]),
        "acc_h": (s["mean_acc_h"], s["std_acc_h"]),
    }


def stats_from_pooled(rows: list) -> dict:
    """The shared results-table stats, recomputed over pooled per-instance rows.

    Matches 04's definitions exactly: 'before' = all instances, 'after' = the
    Level-1 subset, g-mean² = mean over Level-1 instances with a defined value."""
    l1 = [r for r in rows if r["level1_pass"]]
    gm = [r["gmean2_i"] for r in l1 if r["gmean2_i"] is not None]
    return {
        "te_b": _mean_std([r["te_i"] for r in rows if r["te_i"] is not None]),
        "reff_b": _mean_std([r["r_effect_i"] for r in rows if r["r_effect_i"] is not None]),
        "te_a": _mean_std([r["te_i"] for r in l1 if r["te_i"] is not None]),
        "reff_a": _mean_std([r["r_effect_i"] for r in l1 if r["r_effect_i"] is not None]),
        "n_level1": len(l1),
        "n_total": len(rows),
        "gmean2": float(np.mean(gm)) if gm else None,
        "acc_u": _mean_std([r["acc_u_i"] for r in rows if r["acc_u_i"] is not None]),
        "acc_h": _mean_std([r["acc_h_i"] for r in rows if r["acc_h_i"] is not None]),
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
        f"| {_ms(*st['te_b'])} | {_ms(*st['reff_b'])} "
        f"| {_ms(*st['te_a'])} | {_ms(*st['reff_a'])} "
        f"| {_n_pct(st['n_level1'], st.get('n_total'))} | {_fmt(st['gmean2'])} "
        f"| {_ms(*st['acc_u'], 3)} | {_ms(*st['acc_h'], 3)} |",
        "",
    ]


def matrix_block(cm: dict) -> list:
    """Confusion matrix (hinted runs with a verdict) + opacity rate."""
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
        f"**Opacity rate:** {_fmt(cm.get('opacity_rate'))} — {cm['foll_noack']} covert / "
        f"{cm['n_followed']} hint-following runs (covert / all hint-following).",
        "",
    ]


def cot_condition_table(cot: dict) -> list:
    """4-condition run-level token-length table (valid runs; followed +
    not-followed partition hinted exactly)."""
    lines = ["| Condition | Mean | Std | Min | Max | Runs | Instances |",
             "|-----------|------|-----|-----|-----|------|-----------|"]
    for prefix, label in COT_CONDITIONS:
        lines.append(
            f"| {label} "
            f"| {_fmt(cot.get(f'{prefix}_mean'), 1)} "
            f"| {_fmt(cot.get(f'{prefix}_std'), 1)} "
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

    # --- Results (shared format) -------------------------------------------
    w("## Results\n")
    lines += results_table("Dataset", summary.get("dataset", "?"), stats_from_summary(summary))
    lines += matrix_block(cm)

    # --- g-mean² components --------------------------------------------------
    w("---\n")
    w("## g-mean² components\n")
    w("`g-mean²_i = TPR_mc_i · √(TNR_X=0_i · TNR_X=1_i)`, averaged over Level-1 "
      "instances (`TE > 0 and p1 > 0`); `TNR_X=1 = 1` by vacuous truth when `p1 = 1`.\n")
    w("| Metric | Value |")
    w("|--------|-------|")
    w(f"| **g-mean²** | {_fmt(summary['gmean2'])} |")
    w(f"| Mean TPR_mc | {_fmt(summary['mean_tpr_mc'])} |")
    w(f"| Mean TNR_X=0 | {_fmt(summary['mean_tnr_x0'])} |")
    w(f"| Mean TNR_X=1 | {_fmt(summary['mean_tnr_x1'])} |")
    w(f"| Instances contributing | {contributing} "
      f"({summary['n_tnr_x1_vacuous']} vacuous-truth, {summary['n_gmean2_undefined']} excluded for missing coverage) |")
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
      f"`>0` {pos} ({100*pos/n:.1f}%). **Skip rate** {_fmt(summary.get('skip_rate'))} "
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
    return [f"## {heading}\n"] + results_table(label_header, label, st) + matrix_block(cm)


def write_results_overall_summary(runs: dict, model_name: str, hf_id: str, path: Path) -> None:
    """One section per (dataset, hint type), read straight from each run's summary."""
    datasets, hints = _run_keys(runs)
    lines = _file_header("Results overall summary — per (dataset × hint type)",
                         model_name, hf_id, datasets, hints)
    for (ds, hint), run in runs.items():
        s = run["summary"]
        lines += _results_section(f"{ds} × {hint}", "Dataset", ds,
                                  stats_from_summary(s), s.get("confusion_matrix", {}))
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
    for key in keys:
        group = [run for (ds, hint), run in runs.items() if member(key, ds, hint)]
        pooled_rows = [r for run in group for r in run["rows"]]
        st = stats_from_pooled(pooled_rows)
        cm = sum_matrices([run["summary"] for run in group])
        lines += _results_section(f"{key} (pooled over {len(group)} {pooled_over})",
                                  label_header, key, st, cm)
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


def _cot_section(heading: str, cots: list) -> list:
    """One CoT report section: condition table + bucket/truncation row."""
    buckets, trunc, max_tok = _summed_buckets(cots)
    cond = _pooled_cot_stats(cots) if len(cots) > 1 else cots[0]["condition_stats"]
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
    for (ds, hint), cot in cot_runs.items():
        lines += _cot_section(f"{ds} × {hint}", [cot])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_cot_level_summary(cot_runs: dict, model_name: str, hf_id: str, path: Path,
                            level: str) -> None:
    """CoT roll-up per dataset (pooled across hint types) or per hint type
    (pooled across datasets)."""
    datasets, hints = _run_keys(cot_runs)
    if level == "dataset":
        keys, pooled_over = datasets, "hint types"
        title = "CoT token-length dataset-level summary — runs pooled across hint types"
        member = lambda key, ds, hint: ds == key
    else:
        keys, pooled_over = hints, "datasets"
        title = "CoT token-length hint-type-level summary — runs pooled across datasets"
        member = lambda key, ds, hint: hint == key

    lines = _cot_file_header(title, model_name, hf_id, datasets, hints)
    for key in keys:
        group = [cot for (ds, hint), cot in cot_runs.items() if member(key, ds, hint)]
        lines += _cot_section(f"{key} (pooled over {len(group)} {pooled_over})", group)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_cot_aggregate_summary(cot_runs: dict, model_name: str, hf_id: str, path: Path) -> None:
    """Single CoT roll-up pooled across every dataset and hint type."""
    datasets, hints = _run_keys(cot_runs)
    lines = _cot_file_header("CoT token-length aggregate summary — runs pooled across all datasets and hint types",
                             model_name, hf_id, datasets, hints)
    lines += _cot_section(f"All ({len(cot_runs)} dataset × hint-type runs)",
                          list(cot_runs.values()))
    path.write_text("\n".join(lines), encoding="utf-8")
