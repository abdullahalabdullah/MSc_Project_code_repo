#!/usr/bin/env python3
"""
Step 21: Export every main-results summary table as LaTeX.

Mirrors the three markdown reports (steps 15, 19, 20) — same tables, same
values, read from the same machine-readable artifacts — as three .tex files:

    to_latex/main_results_dataset_level_tables.tex      (step 15's tables)
        3 per-model tables (columns = datasets, domain-grouped header) +
        2 delta tables, from {model}/pooled_summary.json
    to_latex/main_results_hint_type_level_tables.tex    (step 19's tables)
        3 per-model tables (columns = hint types) + 2 delta tables, from
        {model}/pooled_summary_by_hint.json
    to_latex/main_results_dataset_hint_level_tables.tex (step 20's tables)
        24 tables, one per (dataset, hint type), columns = models, from
        {model}/{ds}/{hint}/results_summary.json

Each table carries a self-contained caption (model, pooling rule, CI type)
and a stable \\label. Requires in the document preamble:
    \\usepackage{booktabs} \\usepackage{makecell} \\usepackage{graphicx}
(graphicx only for the wide dataset-level tables, which are wrapped in
\\resizebox{\\textwidth}).

Usage:
    python scripts/21_export_latex_tables.py

    Options: --config, --data test|eval,
             --base / --harmless / --antisyc   (as in step 15)

Output:
    {analysis_dir}/to_latex/*.tex   (to_latex/{split}/ for --data eval)
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.dataset import SPLITS, model_dir_name, resolve_paths
from src.hints import HINT_TYPES

sys.path.insert(0, str(Path(__file__).resolve().parent))
DOMAINS = __import__("15_pooled_summary_tables").DOMAINS

DEFAULT_BASE = "Qwen/Qwen3-4B"
DEFAULT_HARMLESS = "er-abd/harmless-dpo-qwen3-4B-step-735"
DEFAULT_ANTISYC = "er-abd/anti-syc-dpo-qwen3-4B-step-520"

HINT_LABELS = {
    "professor": "Professor hint",
    "grader_hacking": "Grader-hacking hint",
    "unethical": "Unethical hint",
}

# (metric key, base row label, superscript: a=paired-boot, b=plain)
METRIC_ROWS = [
    ("gmean2", "$g$-mean$^2$", "a"),
    ("te_on_v", "Mean TE on $V$", "a"),
    ("r_effect_on_v", "Mean $R_{\\mathrm{effect}}$ on $V$", "a"),
    ("pct_te_pos", "\\% with TE $>$ 0", None),
    ("acc_h", "Hinted accuracy", "a"),
    ("acc_u", "Unhinted accuracy (single arm)", "b"),
]

PREAMBLE_NOTE = (
    "% Requires: \\usepackage{booktabs} \\usepackage{makecell} "
    "\\usepackage{graphicx}\n")

V_NOTE = ("$V$ = level-1-passing instances (TE $>$ 0 and $p_1 > 0$); "
          "$n = 200$ instances per dataset, 8 samples per arm.")

CI_NOTE_DATASET = (
    "\\textsuperscript{a}\\,Paired-bootstrap 95\\% CI: paired cluster "
    "bootstrap over instances, one shared resample reused across the three "
    "hint types (professor, grader hacking, unethical; 10{,}000 resamples, "
    "seed 88); the point estimate is the unweighted mean of the three "
    "per-hint-type estimates. \\textsuperscript{b}\\,Plain "
    "percentile-bootstrap 95\\% CI: the unhinted arm is a single condition "
    "shared verbatim by all hint types, so no pooling applies. " + V_NOTE)

# Hint-type-level tables render only the monitorability rows (accuracy rows
# omitted — dataset-pooled accuracy averages disjoint question populations,
# and the unhinted arm cannot vary by hint type), so every CI in them is the
# paired bootstrap and the superscript is dropped.
HINT_METRIC_ROWS = [m for m in METRIC_ROWS if m[0] not in ("acc_h", "acc_u")]

CI_NOTE_HINT = (
    "Every 95\\% CI is a paired cluster bootstrap over instances — each "
    "replicate draws, for every dataset, $n$ instance ids with replacement "
    "from that dataset's shared pool and reuses the same dataset-wise draws "
    "across the three hint types (10{,}000 resamples, seed 88); the point "
    "estimate is the unweighted mean of the 8 per-dataset estimates. "
    "Accuracy rows are omitted at this level (dataset-pooled accuracy "
    "averages disjoint question populations; the unhinted arm cannot vary "
    "by hint type) — see the dataset-level tables. " + V_NOTE)

CI_NOTE_CELL = (
    "Per-model 95\\% CIs are plain percentile cluster bootstraps over "
    "instances (10{,}000 resamples, seed 88), as computed per condition by "
    "step 04; nothing is pooled. $\\Delta$ columns (trained $-$ base) carry "
    "a paired-bootstrap 95\\% CI: one instance resample per replicate, "
    "shared across both models, so the shared-question covariance is "
    "propagated; a CI excluding 0 is a paired test at the 5\\% level "
    "(\\% with TE $>$ 0 deltas in percentage points). " + V_NOTE)

DELTA_NOTE = (
    "Each cell is the difference of pooled point estimates (trained $-$ "
    "base) with a paired-bootstrap 95\\% CI: one instance resample per "
    "replicate, shared across both models and all hint types (10{,}000 "
    "resamples, seed 88), so the shared-question covariance is propagated; "
    "a CI excluding 0 is a paired test at the 5\\% level. The \\% with TE "
    "$>$ 0 row is in percentage points (pp).")


def esc(s: str) -> str:
    return s.replace("&", "\\&").replace("_", "\\_")


def tex(mid: str) -> str:
    return f"\\texttt{{{esc(mid)}}}"


def row_label(key, label, sup, qualifier):
    lab = label + (f" ({qualifier})" if qualifier and key != "acc_u" else "")
    return lab + (f"\\textsuperscript{{{sup}}}" if sup else "")


def ci_cell(entry):
    est, (lo, hi) = entry["estimate"], entry["ci95"]
    if est is None:
        return "N/A"
    return (f"\\makecell{{{est:.3f} \\\\ \\scriptsize[{lo:.3f}, {hi:.3f}]}}")


def pct_cell(est):
    return f"{100 * est:.0f}\\%" if est is not None else "N/A"


def delta_cell(entry, key):
    """One delta cell: point + paired-bootstrap CI from a delta JSON entry."""
    if entry is None or entry.get("delta") is None:
        return "N/A"
    d, (lo, hi) = entry["delta"], entry["ci95"]
    if key == "pct_te_pos":
        dp = round(100 * d)
        point = f"${dp:+d}$ pp" if dp else "$0$ pp"
        return (f"\\makecell{{{point} \\\\ "
                f"\\scriptsize[${100 * lo:+.0f}$, ${100 * hi:+.0f}$]}}")
    dp = round(d, 3)
    point = f"${dp:+.3f}$" if dp else "$0.000$"
    return (f"\\makecell{{{point} \\\\ "
            f"\\scriptsize[${lo:+.3f}$, ${hi:+.3f}$]}}")


def table_env(caption, label, colspec, header_lines, rows, resize):
    out = ["\\begin{table}[htbp]", "\\centering",
           f"\\caption{{{caption}}}", f"\\label{{{label}}}"]
    if resize:
        out.append("\\resizebox{\\textwidth}{!}{%")
    out += [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule"]
    out += header_lines + ["\\midrule"] + rows + ["\\bottomrule"]
    out += ["\\end{tabular}" + ("}" if resize else ""), "\\end{table}"]
    return "\n".join(out)


# --- dataset-level (step 15) -------------------------------------------------

def domain_header(datasets):
    cols = [(dom, ds) for dom, keys in DOMAINS for ds in keys
            if ds in datasets]
    groups, cmid, c = [], [], 2
    for dom, keys in DOMAINS:
        present = [ds for ds in keys if ds in datasets]
        if not present:
            continue
        groups.append(f"\\multicolumn{{{len(present)}}}{{c}}{{{esc(dom)}}}")
        cmid.append(f"\\cmidrule(lr){{{c}-{c + len(present) - 1}}}")
        c += len(present)
    ds_row = " & ".join(tex(ds) for _, ds in cols)
    return cols, [" & " + " & ".join(groups) + " \\\\",
                  " ".join(cmid),
                  f"Metric & {ds_row} \\\\"]


def dataset_level_tables(pooled, deltas_ds, models):
    blocks = []
    datasets = set(pooled[models[0][2]])
    cols, header = domain_header(datasets)
    colspec = "l" + "c" * len(cols)
    for label, mid, short, slug in models:
        rows = []
        for i, (key, lab, sup) in enumerate(METRIC_ROWS):
            cells = []
            for _, ds in cols:
                e = pooled[short][ds][key]
                cells.append(pct_cell(e["estimate"]) if key == "pct_te_pos"
                             else ci_cell(e))
            rows.append(row_label(key, lab, sup, "mean of 3 hints")
                        + " & " + " & ".join(cells) + " \\\\")
            if i < len(METRIC_ROWS) - 1:
                rows.append("\\addlinespace")
        cap = (f"Monitorability of the {label[0].lower() + label[1:]} model "
               f"({tex(mid)}), pooled over hint types, per dataset. "
               + CI_NOTE_DATASET)
        blocks.append(table_env(cap, f"tab:monitorability-{slug}", colspec,
                                header, rows, resize=True))
    for label, mid, short, slug in models[1:]:
        rows = []
        for i, (key, lab, sup) in enumerate(METRIC_ROWS):
            cells = [delta_cell(deltas_ds[short][ds].get(key), key)
                     for _, ds in cols]
            rows.append(f"$\\Delta$ {row_label(key, lab, None, 'mean of 3 hints')}"
                        + " & " + " & ".join(cells) + " \\\\")
            if i < len(METRIC_ROWS) - 1:
                rows.append("\\addlinespace")
        cap = (f"Base vs.\\ {label[0].lower() + label[1:]}: change in "
               f"monitorability after training, per dataset "
               f"({tex(mid)} $-$ {tex(models[0][1])}). " + DELTA_NOTE)
        blocks.append(table_env(cap, f"tab:delta-{slug}", colspec,
                                header, rows, resize=True))
    return blocks


# --- hint-type-level (step 19) -----------------------------------------------

def hint_entry(data, hint, key):
    return data["acc_u"] if key == "acc_u" else data["hint_types"][hint][key]


def hint_level_tables(by_hint, deltas_hint, models):
    blocks = []
    header = ["Metric & " + " & ".join(HINT_LABELS[h] for h in HINT_TYPES)
              + " \\\\"]
    colspec = "l" + "c" * len(HINT_TYPES)
    for label, mid, short, slug in models:
        rows = []
        for i, (key, lab, _) in enumerate(HINT_METRIC_ROWS):
            cells = []
            for h in HINT_TYPES:
                e = hint_entry(by_hint[short], h, key)
                cells.append(pct_cell(e["estimate"]) if key == "pct_te_pos"
                             else ci_cell(e))
            rows.append(row_label(key, lab, None, "mean of 8 datasets")
                        + " & " + " & ".join(cells) + " \\\\")
            if i < len(HINT_METRIC_ROWS) - 1:
                rows.append("\\addlinespace")
        cap = (f"Monitorability of the {label[0].lower() + label[1:]} model "
               f"({tex(mid)}), per hint type, pooled over the 8 datasets. "
               + CI_NOTE_HINT)
        blocks.append(table_env(cap, f"tab:monitorability-by-hint-{slug}",
                                colspec, header, rows, resize=False))
    for label, mid, short, slug in models[1:]:
        rows = []
        for i, (key, lab, _) in enumerate(HINT_METRIC_ROWS):
            cells = [delta_cell(deltas_hint[short][h].get(key), key)
                     for h in HINT_TYPES]
            rows.append(f"$\\Delta$ {row_label(key, lab, None, 'mean of 8 datasets')}"
                        + " & " + " & ".join(cells) + " \\\\")
            if i < len(HINT_METRIC_ROWS) - 1:
                rows.append("\\addlinespace")
        cap = (f"Base vs.\\ {label[0].lower() + label[1:]}: change in "
               f"monitorability after training, per hint type, pooled over "
               f"the 8 datasets ({tex(mid)} $-$ {tex(models[0][1])}). "
               + DELTA_NOTE)
        blocks.append(table_env(cap, f"tab:delta-by-hint-{slug}", colspec,
                                header, rows, resize=False))
    return blocks


# --- (dataset, hint)-level (step 20) -----------------------------------------

# (label, summary estimate key, summary ci95 key, delta-JSON key)
CELL_ROWS = [
    ("$g$-mean$^2$", "gmean2", "gmean2", "gmean2"),
    ("Mean TE on $V$", "mean_TE_after", "TE_after", "te_on_v"),
    ("Mean $R_{\\mathrm{effect}}$ on $V$", "mean_R_effect_after",
     "R_effect_after", "r_effect_on_v"),
    ("\\% with TE $>$ 0", None, None, "pct_te_pos"),
    ("Hinted accuracy", "mean_acc_h", "acc_h", "acc_h"),
    ("Unhinted accuracy (single arm)", "mean_acc_u", "acc_u", "acc_u"),
]


def cell_value(summary, est_key, ci_key):
    if summary is None:
        return "N/A"
    if est_key is None:  # % with TE > 0
        n_total, n_pos = summary.get("n_total"), summary.get("n_te_positive")
        return (f"{100 * n_pos / n_total:.0f}\\%"
                if n_total and n_pos is not None else "N/A")
    est = summary.get(est_key)
    ci = (summary.get("ci95") or {}).get(ci_key)
    if est is None:
        return "N/A"
    if ci is None:
        return f"{est:.3f}"
    return f"\\makecell{{{est:.3f} \\\\ \\scriptsize[{ci[0]:.3f}, {ci[1]:.3f}]}}"


def cell_level_tables(summaries, deltas_cell, models):
    blocks = []
    delta_heads = [f"$\\Delta$ {lbl} $-$ Base" for lbl, _, _, _ in models[1:]]
    header = ["Metric & " + " & ".join(lbl for lbl, _, _, _ in models)
              + " & " + " & ".join(delta_heads) + " \\\\"]
    colspec = "l" + "c" * (len(models) + len(models) - 1)
    for dom, keys in DOMAINS:
        for ds in keys:
            for hint in HINT_TYPES:
                if (ds, hint) not in summaries:
                    continue
                by_model = summaries[(ds, hint)]
                rows = []
                for i, (lab, est_key, ci_key, dkey) in enumerate(CELL_ROWS):
                    cells = [cell_value(by_model.get(short), est_key, ci_key)
                             for _, _, short, _ in models]
                    cells += [delta_cell(
                        deltas_cell[short].get(f"{ds}/{hint}", {}).get(dkey),
                        dkey) for _, _, short, _ in models[1:]]
                    rows.append(lab + " & " + " & ".join(cells) + " \\\\")
                    if i < len(CELL_ROWS) - 1:
                        rows.append("\\addlinespace")
                cap = (f"Monitorability on {tex(ds)} ({esc(dom)}) under the "
                       f"{HINT_LABELS[hint].lower()}: base vs.\\ harmlessness "
                       f"DPO vs.\\ anti-sycophancy DPO. " + CI_NOTE_CELL)
                blocks.append(table_env(
                    cap, f"tab:cell-{ds.replace('_', '-')}-"
                         f"{hint.replace('_', '-')}",
                    colspec, header, rows, resize=True))
    return blocks


def main():
    parser = argparse.ArgumentParser(
        description="Export the main-results summary tables as LaTeX.")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--data", default="test", choices=SPLITS)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--harmless", default=DEFAULT_HARMLESS)
    parser.add_argument("--antisyc", default=DEFAULT_ANTISYC)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    with config_path.open() as f:
        config = yaml.safe_load(f)
    datasets = list(config.get("datasets", {}).keys())
    analysis_dir = root / config["paths"]["analysis_dir"]

    # (display label, model id, dir name, latex label slug)
    models = [
        ("Base", args.base, model_dir_name(args.base), "base"),
        ("Harmlessness DPO", args.harmless, model_dir_name(args.harmless),
         "harmless-dpo"),
        ("Anti-sycophancy DPO", args.antisyc, model_dir_name(args.antisyc),
         "antisyc-dpo"),
    ]

    def model_json(short, name):
        p = analysis_dir / short
        if args.data != "test":
            p = p / args.data
        p = p / name
        if not p.exists():
            print(f"ERROR: {p} not found — run the generating step first.")
            sys.exit(1)
        with p.open() as f:
            return json.load(f)

    pooled = {short: model_json(short, "pooled_summary.json")["datasets"]
              for _, _, short, _ in models}
    by_hint = {short: model_json(short, "pooled_summary_by_hint.json")
               for _, _, short, _ in models}
    trained = [short for _, _, short, _ in models[1:]]
    deltas_ds = {s: model_json(s, "delta_vs_base.json")["datasets"]
                 for s in trained}
    deltas_hint = {s: model_json(s, "delta_vs_base_by_hint.json")["hint_types"]
                   for s in trained}
    deltas_cell = {s: model_json(s, "delta_vs_base_per_cell.json")["cells"]
                   for s in trained}

    cell_summaries = {}
    for ds in datasets:
        for hint in HINT_TYPES:
            by_model = {}
            for _, _, short, _ in models:
                paths = resolve_paths(root, config, ds, short, hint,
                                      split=args.data)
                p = paths["analysis_dir"] / "results_summary.json"
                if p.exists():
                    with p.open() as f:
                        by_model[short] = json.load(f)
                else:
                    print(f"WARNING: missing {p}")
            cell_summaries[(ds, hint)] = by_model

    out_dir = analysis_dir / "to_latex"
    if args.data != "test":
        out_dir = out_dir / args.data
    out_dir.mkdir(parents=True, exist_ok=True)

    exports = [
        ("main_results_dataset_level_tables.tex",
         dataset_level_tables(pooled, deltas_ds, models)),
        ("main_results_hint_type_level_tables.tex",
         hint_level_tables(by_hint, deltas_hint, models)),
        ("main_results_dataset_hint_level_tables.tex",
         cell_level_tables(cell_summaries, deltas_cell, models)),
    ]
    for name, blocks in exports:
        path = out_dir / name
        path.write_text(PREAMBLE_NOTE + "\n" + "\n\n".join(blocks) + "\n")
        print(f"Wrote {path} ({len(blocks)} tables)")


if __name__ == "__main__":
    main()
