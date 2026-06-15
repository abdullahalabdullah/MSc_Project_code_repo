"""
Hint-susceptibility analysis: correlating unhinted accuracy with hint-following.

Consumes the per-instance N-sample records produced by 07_estimate_accuracy.py
for both arms:
  - analysis/{model}/accuracy_runs.jsonl       (unhinted, N runs)
  - analysis/{model}/hint_accuracy_runs.jsonl  (hinted, N runs)

and computes, per instance:
  acc_u = P(answer == correct_answer | unhinted)
  p0    = P(answer == hint_target    | unhinted)   -- base rate
  p1    = P(answer == hint_target    | hinted)     -- hint-following rate
  te    = p1 - p0                                   -- hint effect, base-rate-adjusted

Dataset level:
  - Spearman rho (primary) and Pearson r (secondary) for (acc_u, p1) and (acc_u, te)
  - Wrong/Right/Mixed buckets (acc_u == 0 / == 1 / in between) with a
    Mann-Whitney U test comparing Wrong vs Right on p1 and te
  - A 6-bin breakdown of acc_u -> mean p1 / te, used for the figure

See documentation.md, section "Hint susceptibility vs. accuracy" for the
methodology write-up and interpretation guidance.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless runs
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Merge unhinted + hinted per-instance records
# ---------------------------------------------------------------------------

def merge_instance_records(unhinted_records: list, hinted_records: list) -> list:
    """Join unhinted and hinted N-sample records on instance_id.

    An instance is included only if both arms have a defined accuracy/p0/p1
    (i.e. at least one valid run in each arm).
    """
    u_by_id = {r["instance_id"]: r for r in unhinted_records}
    h_by_id = {r["instance_id"]: r for r in hinted_records}
    ids = sorted(set(u_by_id) & set(h_by_id))

    per_instance = []
    for iid in ids:
        u, h = u_by_id[iid], h_by_id[iid]
        acc_u, p0, p1 = u.get("accuracy"), u.get("p0"), h.get("p1")
        if acc_u is None or p0 is None or p1 is None:
            continue
        per_instance.append({
            "instance_id": iid,
            "question_id": u["question_id"],
            "correct_answer": u["correct_answer"],
            "hint_target": u["hint_target"],
            "acc_u": acc_u,
            "p0": p0,
            "p1": p1,
            "te": p1 - p0,
            "n_unhinted_valid": u["n_valid"],
            "n_hinted_valid": h["n_valid"],
        })
    return per_instance


# ---------------------------------------------------------------------------
# Correlations, buckets, Mann-Whitney, binning
# ---------------------------------------------------------------------------

def _corr(x: list, y: list) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"pearson_r": None, "pearson_p": None,
                "spearman_r": None, "spearman_p": None, "n": int(len(x))}
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    return {"pearson_r": float(pr), "pearson_p": float(pp),
            "spearman_r": float(sr), "spearman_p": float(sp), "n": int(len(x))}


def _bucket_summary(group: list) -> dict:
    if not group:
        return {"n": 0, "mean_p0": None, "mean_p1": None, "mean_te": None}
    return {
        "n": len(group),
        "mean_p0": float(np.mean([r["p0"] for r in group])),
        "mean_p1": float(np.mean([r["p1"] for r in group])),
        "mean_te": float(np.mean([r["te"] for r in group])),
    }


def _mannwhitney(group_a: list, group_b: list, key: str) -> dict:
    if not group_a or not group_b:
        return {"u": None, "p": None, "rank_biserial": None,
                "n_a": len(group_a), "n_b": len(group_b)}
    a = [r[key] for r in group_a]
    b = [r[key] for r in group_b]
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    # rank-biserial effect size: r = 1 - 2U / (n_a * n_b), in [-1, 1]
    rank_biserial = 1 - (2 * u) / (len(a) * len(b))
    return {"u": float(u), "p": float(p), "rank_biserial": float(rank_biserial),
            "n_a": len(a), "n_b": len(b)}


# acc_u bins: exact 0, exact 1, and four quartile-ish bands in between.
BIN_ORDER = ["0", "(0, 0.25]", "(0.25, 0.5]", "(0.5, 0.75]", "(0.75, 1)", "1"]


def _bin_label(acc_u: float) -> str:
    if acc_u == 0.0:
        return "0"
    if acc_u <= 0.25:
        return "(0, 0.25]"
    if acc_u <= 0.5:
        return "(0.25, 0.5]"
    if acc_u <= 0.75:
        return "(0.5, 0.75]"
    if acc_u < 1.0:
        return "(0.75, 1)"
    return "1"


def _bin_instances(per_instance: list) -> dict:
    bins = {label: [] for label in BIN_ORDER}
    for r in per_instance:
        bins[_bin_label(r["acc_u"])].append(r)

    binned = {}
    for label in BIN_ORDER:
        group = bins[label]
        if not group:
            binned[label] = {"n": 0, "mean_p1": None, "sem_p1": None,
                              "mean_te": None, "sem_te": None}
            continue
        p1_g = np.array([r["p1"] for r in group])
        te_g = np.array([r["te"] for r in group])
        binned[label] = {
            "n": len(group),
            "mean_p1": float(np.mean(p1_g)),
            "sem_p1": float(np.std(p1_g, ddof=1) / np.sqrt(len(p1_g))) if len(p1_g) > 1 else 0.0,
            "mean_te": float(np.mean(te_g)),
            "sem_te": float(np.std(te_g, ddof=1) / np.sqrt(len(te_g))) if len(te_g) > 1 else 0.0,
        }
    return binned


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def compute_hint_susceptibility(unhinted_records: list, hinted_records: list) -> dict:
    per_instance = merge_instance_records(unhinted_records, hinted_records)

    acc_vals = [r["acc_u"] for r in per_instance]
    p1_vals = [r["p1"] for r in per_instance]
    te_vals = [r["te"] for r in per_instance]

    correlations = {
        "acc_u_vs_p1": _corr(acc_vals, p1_vals),
        "acc_u_vs_te": _corr(acc_vals, te_vals),
    }

    wrong = [r for r in per_instance if r["acc_u"] == 0.0]
    right = [r for r in per_instance if r["acc_u"] == 1.0]
    mixed = [r for r in per_instance if 0.0 < r["acc_u"] < 1.0]

    buckets = {
        "wrong_unhinted": _bucket_summary(wrong),
        "right_unhinted": _bucket_summary(right),
        "mixed_unhinted": _bucket_summary(mixed),
    }

    wrong_vs_right = {
        "p1": _mannwhitney(wrong, right, "p1"),
        "te": _mannwhitney(wrong, right, "te"),
    }

    binned = _bin_instances(per_instance)

    return {
        "per_instance": per_instance,
        "n_total": len(per_instance),
        "correlations": correlations,
        "buckets": buckets,
        "wrong_vs_right": wrong_vs_right,
        "binned": binned,
        "bin_order": BIN_ORDER,
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _fmt(v, decimals: int = 4) -> str:
    return f"{v:.{decimals}f}" if v is not None else "N/A"


def write_hint_susceptibility_md(summary: dict, path: Path) -> None:
    n = summary["n_total"]
    corr = summary["correlations"]
    buckets = summary["buckets"]
    wvr = summary["wrong_vs_right"]
    binned = summary["binned"]

    lines = []
    w = lines.append

    w(f"# Hint Susceptibility vs. Accuracy: {summary['model_name']}\n")
    w(f"**N samples per arm:** {summary['n_samples']}  ")
    w(f"**Instances with valid data in both arms:** {n}\n")

    if n == 0:
        w("\nNo instances had valid data in both arms — nothing to report.\n")
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    w("---\n")
    w("## 1. Definitions\n")
    w(
        "For each instance, `n_samples` unhinted runs give `acc_u` (fraction "
        "correct) and `p0` (fraction choosing `hint_target`, the always-incorrect "
        "option the hint points to — see `src/dataset.py`); `n_samples` hinted "
        "runs give `p1` (fraction choosing `hint_target` when the hint is present). "
        "`te = p1 - p0` is the hint effect net of the base rate at which the model "
        "picks `hint_target` anyway.\n"
    )

    w("---\n")
    w("## 2. Correlations (instance level, continuous)\n")
    w("| Pair | Spearman rho | p-value | Pearson r | p-value | n |")
    w("|------|--------------|---------|-----------|---------|---|")
    for key, label in [("acc_u_vs_p1", "acc_u vs p1 (hint-following rate)"),
                        ("acc_u_vs_te", "acc_u vs te (hint effect, base-rate-adjusted)")]:
        c = corr[key]
        w(f"| {label} | {_fmt(c['spearman_r'])} | {_fmt(c['spearman_p'])} | "
          f"{_fmt(c['pearson_r'])} | {_fmt(c['pearson_p'])} | {c['n']} |")
    w("")
    w(
        "Spearman's rho is the primary measure (both variables are bounded "
        "proportions with many ties at N=32; rho only assumes a monotonic "
        "relationship). Pearson's r is reported for comparison — a large gap "
        "between the two suggests the relationship is non-linear or driven by "
        "a few extreme instances.\n"
    )
    w(
        "A **negative** rho for `acc_u vs te` means: the more accurate the model "
        "is unhinted on an item, the smaller the hint's causal pull on that item "
        "— i.e. hint-susceptibility concentrates on items the model doesn't "
        "already know.\n"
    )

    w("---\n")
    w("## 3. Wrong / Right / Mixed buckets (binary framing)\n")
    w(
        "`Wrong unhinted` = acc_u == 0 (never correct over N unhinted runs); "
        "`Right unhinted` = acc_u == 1 (always correct); `Mixed` = 0 < acc_u < 1.\n"
    )
    w("| Bucket | n | mean p0 | mean p1 | mean te |")
    w("|--------|---|---------|---------|---------|")
    for key, label in [("wrong_unhinted", "Wrong unhinted (acc_u = 0)"),
                        ("right_unhinted", "Right unhinted (acc_u = 1)"),
                        ("mixed_unhinted", "Mixed (0 < acc_u < 1)")]:
        b = buckets[key]
        w(f"| {label} | {b['n']} | {_fmt(b['mean_p0'])} | {_fmt(b['mean_p1'])} | {_fmt(b['mean_te'])} |")
    w("")

    w("### Wrong vs. Right -- Mann-Whitney U test\n")
    w(
        "Non-parametric comparison of `p1` / `te` distributions between the "
        "Wrong-unhinted and Right-unhinted buckets. `rank_biserial` is the "
        "effect size (in [-1, 1]); its sign follows the `(Wrong, Right)` "
        "argument order, i.e. positive means Wrong-unhinted instances tend to "
        "have higher values.\n"
    )
    w("| Measure | U | p-value | rank-biserial r | n (Wrong) | n (Right) |")
    w("|---------|---|---------|------------------|-----------|-----------|")
    for key, label in [("p1", "p1 (hint-following rate)"), ("te", "te (hint effect)")]:
        m = wvr[key]
        w(f"| {label} | {_fmt(m['u'], 1)} | {_fmt(m['p'])} | {_fmt(m['rank_biserial'])} | "
          f"{m['n_a']} | {m['n_b']} |")
    w("")

    w("---\n")
    w("## 4. Binned breakdown (acc_u bin -> mean p1 / te)\n")
    w("| acc_u bin | n | mean p1 | SEM(p1) | mean te | SEM(te) |")
    w("|-----------|---|---------|---------|---------|---------|")
    for label in summary["bin_order"]:
        b = binned[label]
        if b["n"] == 0:
            w(f"| {label} | 0 | N/A | N/A | N/A | N/A |")
        else:
            w(f"| {label} | {b['n']} | {_fmt(b['mean_p1'])} | {_fmt(b['sem_p1'])} | "
              f"{_fmt(b['mean_te'])} | {_fmt(b['sem_te'])} |")
    w("")

    w("---\n")
    w("## 5. Figure\n")
    w(
        f"See `{summary['plot_filename']}`: top row is a jittered scatter of "
        "acc_u vs p1 / acc_u vs te (one point per instance) with a linear trend "
        "line and the Spearman/Pearson stats from Section 2; bottom row is the "
        "binned bar chart from Section 4, with 95% CI error bars and `n` per bin.\n"
    )

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_hint_susceptibility(summary: dict, output_path: Path) -> None:
    per_instance = summary["per_instance"]
    n = summary["n_total"]
    if n == 0:
        return

    corr = summary["correlations"]
    binned = summary["binned"]
    bin_order = summary["bin_order"]

    acc = np.array([r["acc_u"] for r in per_instance])
    p1 = np.array([r["p1"] for r in per_instance])
    te = np.array([r["te"] for r in per_instance])

    rng = np.random.default_rng(0)

    def _jitter(x, amount=0.01):
        return x + rng.uniform(-amount, amount, size=len(x))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # --- Top-left: scatter acc_u vs p1 -------------------------------------
    ax = axes[0, 0]
    ax.scatter(_jitter(acc), _jitter(p1), alpha=0.35, s=20, color="#1f77b4")
    if len(acc) > 1 and np.std(acc) > 0:
        coeffs = np.polyfit(acc, p1, 1)
        xs = np.linspace(0, 1, 100)
        ax.plot(xs, np.polyval(coeffs, xs), color="#d62728", linewidth=2)
    c = corr["acc_u_vs_p1"]
    ax.set_title(f"acc_u vs p1  (Spearman rho={_fmt(c['spearman_r'])}, "
                  f"p={_fmt(c['spearman_p'])}, n={c['n']})", fontsize=10)
    ax.set_xlabel("acc_u (unhinted accuracy)")
    ax.set_ylabel("p1 (hint-following rate)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # --- Top-right: scatter acc_u vs te -------------------------------------
    ax = axes[0, 1]
    ax.scatter(_jitter(acc), _jitter(te), alpha=0.35, s=20, color="#2ca02c")
    if len(acc) > 1 and np.std(acc) > 0:
        coeffs = np.polyfit(acc, te, 1)
        xs = np.linspace(0, 1, 100)
        ax.plot(xs, np.polyval(coeffs, xs), color="#d62728", linewidth=2)
    c = corr["acc_u_vs_te"]
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"acc_u vs te  (Spearman rho={_fmt(c['spearman_r'])}, "
                  f"p={_fmt(c['spearman_p'])}, n={c['n']})", fontsize=10)
    ax.set_xlabel("acc_u (unhinted accuracy)")
    ax.set_ylabel("te = p1 - p0 (hint effect)")
    ax.set_xlim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    xs_pos = np.arange(len(bin_order))

    # --- Bottom-left: binned mean p1 -----------------------------------------
    ax = axes[1, 0]
    means = [binned[b]["mean_p1"] for b in bin_order]
    sems = [binned[b]["sem_p1"] for b in bin_order]
    ns = [binned[b]["n"] for b in bin_order]
    means_plot = [m if m is not None else 0.0 for m in means]
    errs_plot = [1.96 * s if s is not None else 0.0 for s in sems]
    ax.bar(xs_pos, means_plot, yerr=errs_plot, capsize=4, color="#1f77b4", alpha=0.8)
    for x, m, nb in zip(xs_pos, means_plot, ns):
        if nb > 0:
            ax.text(x, m + 0.02, f"n={nb}", ha="center", fontsize=8)
    ax.set_xticks(xs_pos)
    ax.set_xticklabels(bin_order, rotation=30, ha="right")
    ax.set_ylabel("mean p1 (hint-following rate)")
    ax.set_xlabel("acc_u bin")
    ax.set_ylim(0, max(means_plot, default=0) * 1.3 + 0.05)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Bottom-right: binned mean te ----------------------------------------
    ax = axes[1, 1]
    means = [binned[b]["mean_te"] for b in bin_order]
    sems = [binned[b]["sem_te"] for b in bin_order]
    ns = [binned[b]["n"] for b in bin_order]
    means_plot = [m if m is not None else 0.0 for m in means]
    errs_plot = [1.96 * s if s is not None else 0.0 for s in sems]
    ax.bar(xs_pos, means_plot, yerr=errs_plot, capsize=4, color="#2ca02c", alpha=0.8)
    for x, m, nb in zip(xs_pos, means_plot, ns):
        if nb > 0:
            offset = 0.02 if m >= 0 else -0.05
            ax.text(x, m + offset, f"n={nb}", ha="center", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(xs_pos)
    ax.set_xticklabels(bin_order, rotation=30, ha="right")
    ax.set_ylabel("mean te (hint effect)")
    ax.set_xlabel("acc_u bin")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Hint susceptibility vs. unhinted accuracy -- {summary['model_name']} (n={n})",
                  fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
