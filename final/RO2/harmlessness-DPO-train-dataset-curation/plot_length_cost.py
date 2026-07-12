"""Length-vs-cost visualisation for the curated harmlessness DPO pairs.

Visualises the "length / verbosity confound" numbers stage 05 already computes (see
report.md's "Length / verbosity confound" section): whether response length predicts Beaver
cost, both across chosen-vs-rejected and within the chosen (safe) group alone.

Called automatically from 05_report.py (writes config.LENGTH_COST_FIG_PATH on every run).
Can also be run standalone against an existing final_pairs.joblib without re-running the
scoring stages:

    python plot_length_cost.py [output_dir]   # default: config.OUTPUT_DIR
"""
import sys

import joblib
import numpy as np


def make_length_cost_plots(final_pairs, out_path):
    """Two-panel figure:

    left  — full-response length (tokens) vs Beaver cost, chosen and rejected overlaid, with
            a per-group linear fit. Visualises both the across-pair confound (rejected cluster
            sits up/right of chosen) and the within-chosen relationship (fit line through the
            chosen points alone).
    right — per-pair length delta (rejected - chosen, answer words) vs cost margin
            (rejected - chosen), the same pairing behind the report's "Pearson(cost margin,
            answer-word Δ)" stat, with a fit line and the r annotated.

    No-ops (prints and returns) if there are no pairs or matplotlib isn't installed.
    """
    if not final_pairs:
        print("[plot_length_cost] no pairs, skipping")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cho_tok = np.array([p["chosen_full_tokens"] for p in final_pairs], dtype=float)
    rej_tok = np.array([p["rejected_full_tokens"] for p in final_pairs], dtype=float)
    cho_cost = np.array([p["chosen_cost"] for p in final_pairs], dtype=float)
    rej_cost = np.array([p["rejected_cost"] for p in final_pairs], dtype=float)
    cho_w = np.array([p["chosen_answer_words"] for p in final_pairs], dtype=float)
    rej_w = np.array([p["rejected_answer_words"] for p in final_pairs], dtype=float)
    margin = rej_cost - cho_cost
    w_delta = rej_w - cho_w

    def fit_line(x, y):
        if len(x) < 2 or np.std(x) == 0:
            return None
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        return xs, slope * xs + intercept, float(np.corrcoef(x, y)[0, 1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ── left: length vs cost, chosen vs rejected ────────────────────────────
    ax1.scatter(cho_tok, cho_cost, s=6, alpha=0.25, color="#4C72B0", label="chosen (safe)")
    ax1.scatter(rej_tok, rej_cost, s=6, alpha=0.25, color="#C44E52", label="rejected (unsafe)")
    fit = fit_line(cho_tok, cho_cost)
    if fit:
        xs, ys, r = fit
        ax1.plot(xs, ys, color="#1C3D6E", linewidth=2, label=f"chosen fit (r={r:+.2f})")
    fit = fit_line(rej_tok, rej_cost)
    if fit:
        xs, ys, r = fit
        ax1.plot(xs, ys, color="#7A1F24", linewidth=2, label=f"rejected fit (r={r:+.2f})")
    ax1.set_xlabel("Full response length (tokens)")
    ax1.set_ylabel("Beaver cost (lower = safer)")
    ax1.set_title("Length vs cost — chosen vs rejected")
    ax1.legend(fontsize=8, loc="best")

    # ── right: per-pair length delta vs cost margin ─────────────────────────
    ax2.scatter(w_delta, margin, s=6, alpha=0.3, color="#55A868")
    fit = fit_line(w_delta, margin)
    if fit:
        xs, ys, r = fit
        ax2.plot(xs, ys, color="#2C5C34", linewidth=2, label=f"fit (r={r:+.3f})")
        ax2.legend(fontsize=9, loc="best")
    ax2.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("Answer-word Δ (rejected − chosen)")
    ax2.set_ylabel("Cost margin (rejected − chosen)")
    ax2.set_title("Per-pair length Δ vs cost margin")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[plot_length_cost] wrote {out_path}")


if __name__ == "__main__":
    import config

    if len(sys.argv) > 1:
        import os
        output_dir = sys.argv[1]
        final_pairs_path = os.path.join(output_dir, "final_pairs.joblib")
        out_path = os.path.join(output_dir, "length_cost_scatter.png")
    else:
        final_pairs_path = config.FINAL_PAIRS_PATH
        out_path = config.LENGTH_COST_FIG_PATH

    final_pairs = joblib.load(final_pairs_path)
    print(f"Loaded {len(final_pairs)} final pairs from {final_pairs_path}")
    make_length_cost_plots(final_pairs, out_path)
