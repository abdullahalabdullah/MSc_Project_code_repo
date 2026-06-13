"""
Visualisation for the g-mean^2 monitorability evaluation.

Produces:
  - TE distribution with the Level-1 boundary (TE = 0) marked
  - g-mean^2 distribution over Level-1-passing instances
  - TPR_mc vs sqrt(TNR_X=0 * TNR_X=1) scatter
  - CoT token length distribution (hinted vs unhinted)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless runs
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# TE distribution with Level-1 boundary
# ---------------------------------------------------------------------------

def plot_te_distribution(per_instance: list, output_path: Path, bin_width: float = 0.2):
    """
    Histogram of per-instance TE values using fixed-width bins (default 0.2).

    Bins span [-1, 1]. Bars are coloured red for TE < 0 (fails Level-1) and
    blue for TE > 0 (passes the TE part of Level-1). Instances with TE exactly
    0 also fail Level-1, so they are pulled out of the histogram into a separate
    black column centred on TE = 0. A vertical dashed line marks TE = 0, the
    Level-1 boundary (instances also need p1_i > 0 to pass).
    """
    te = np.array([r["te_i"] for r in per_instance])
    n = len(te)

    # Exact-zero TE fails Level-1; show it as its own column rather than
    # letting it fall into the adjacent [0, bin_width) "TE > 0" bin.
    zero_mask = np.isclose(te, 0.0, atol=1e-9)
    n_zero = int(zero_mask.sum())
    te_nonzero = te[~zero_mask]

    edges = np.arange(-1.0, 1.0 + bin_width + 1e-9, bin_width)
    counts, edges = np.histogram(te_nonzero, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2

    colors = ["#d62728" if c < 0 else "#1f77b4" for c in centers]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(centers, counts, width=bin_width * 0.88,
                   color=colors, alpha=0.80, linewidth=0.5, edgecolor="white")

    # Dedicated black column for TE == 0, sitting on the boundary.
    zero_bar = ax.bar([0.0], [n_zero], width=bin_width * 0.5,
                      color="black", alpha=0.90, linewidth=0.5,
                      edgecolor="white", zorder=5)

    for bar, count in zip(list(bars) + list(zero_bar),
                          list(counts) + [n_zero]):
        if count == 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(count),
            ha="center", va="bottom", fontsize=9, zorder=6,
        )

    n_level1 = int((te > 0).sum())
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1.5, zorder=1,
                label=f"Level-1 boundary (TE = 0)  —  {n_level1}/{n} instances have TE > 0")

    import matplotlib.patches as mpatches
    ax.legend(
        handles=[
            mpatches.Patch(color="#d62728", alpha=0.80, label="TE < 0 (fails Level-1)"),
            mpatches.Patch(color="black", alpha=0.90, label="TE = 0 (fails Level-1)"),
            mpatches.Patch(color="#1f77b4", alpha=0.80, label="TE > 0"),
            ax.get_legend_handles_labels()[0][0],
        ],
        fontsize=9,
    )

    ax.set_xlabel("TE per instance", fontsize=12)
    ax.set_ylabel("Number of instances", fontsize=12)
    ax.set_title(f"Distribution of per-instance treatment effects (n = {n}, bin width = {bin_width})",
                  fontsize=13)
    ax.set_xlim(edges[0] - bin_width * 0.5, edges[-1] + bin_width * 0.5)
    ax.set_xticks(edges)
    ax.set_xticklabels([f"{e:.1f}" for e in edges], fontsize=8, rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# g-mean^2 distribution
# ---------------------------------------------------------------------------

def plot_gmean2_distribution(per_instance: list, output_path: Path):
    """
    Histogram of per-instance g-mean^2 values, restricted to instances that
    pass the Level-1 filter and have a defined g-mean^2 (i.e. TNR_X=1 is defined).
    """
    vals = [r["gmean2_i"] for r in per_instance if r.get("level1_pass") and r["gmean2_i"] is not None]

    if not vals:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 21)
    ax.hist(vals, bins=bins, color="#2ca02c", alpha=0.75, edgecolor="white")
    ax.axvline(x=float(np.mean(vals)), color="black", linestyle="--", linewidth=1.5,
                label=f"mean = {np.mean(vals):.3f}")

    ax.set_xlabel("g-mean^2 per instance", fontsize=12)
    ax.set_ylabel("Number of instances", fontsize=12)
    ax.set_title(f"Distribution of g-mean^2 (n = {len(vals)} Level-1-passing instances)", fontsize=13)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# TPR_mc vs TNR scatter
# ---------------------------------------------------------------------------

def plot_tpr_tnr_scatter(per_instance: list, output_path: Path):
    """
    Scatter of TPR_mc_i (x) vs sqrt(TNR_X=0_i * TNR_X=1_i) (y) for Level-1-passing
    instances with a defined g-mean^2. The g-mean^2_i value (the product of the
    two axes) is encoded as point colour. The y=x reference line shows where the
    two components are balanced.

    TPR_mc_i and TNR_i are often exactly 0 or 1 (small per-instance sample counts),
    so many instances land on identical coordinates. A small fixed-seed jitter is
    added to (x, y) purely for display, so overlapping points remain visible as a
    cluster rather than collapsing into one marker.
    """
    x, y, c = [], [], []
    for r in per_instance:
        if not r.get("level1_pass") or r["gmean2_i"] is None:
            continue
        x.append(r["tpr_mc_i"])
        y.append(float(np.sqrt(r["tnr_x0_i"] * r["tnr_x1_i"])))
        c.append(r["gmean2_i"])

    if not x:
        return

    rng = np.random.default_rng(0)
    jitter_x = rng.normal(0, 0.012, size=len(x))
    jitter_y = rng.normal(0, 0.012, size=len(y))
    x_jit = np.clip(np.array(x) + jitter_x, -0.05, 1.05)
    y_jit = np.clip(np.array(y) + jitter_y, -0.05, 1.05)

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(x_jit, y_jit, c=c, cmap="viridis", alpha=0.5, s=30,
                     edgecolors="black", linewidths=0.3, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label="g-mean^2")

    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1.2, label="y = x")

    ax.set_xlabel("TPR_mc (jittered)", fontsize=12)
    ax.set_ylabel("sqrt(TNR_X=0 * TNR_X=1) (jittered)", fontsize=12)
    ax.set_title(f"TPR_mc vs TNR component (n = {len(x)})", fontsize=13)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# CoT length distribution
# ---------------------------------------------------------------------------

def plot_cot_length_distribution(per_instance: list, output_path: Path):
    """
    Histogram of mean CoT token lengths per instance, hinted vs unhinted overlay.
    """
    h_lens = [r["mean_cot_tokens_hinted"]   for r in per_instance if r.get("mean_cot_tokens_hinted")   is not None]
    u_lens = [r["mean_cot_tokens_unhinted"]  for r in per_instance if r.get("mean_cot_tokens_unhinted") is not None]

    if not h_lens and not u_lens:
        return

    all_vals = h_lens + u_lens
    bins = np.linspace(min(all_vals), max(all_vals), 30)

    fig, ax = plt.subplots(figsize=(10, 5))
    if h_lens:
        ax.hist(h_lens, bins=bins, alpha=0.6, color="#1f77b4",
                label=f"Hinted  (mean={np.mean(h_lens):.1f}, std={np.std(h_lens):.1f})")
    if u_lens:
        ax.hist(u_lens, bins=bins, alpha=0.6, color="#ff7f0e",
                label=f"Unhinted (mean={np.mean(u_lens):.1f}, std={np.std(u_lens):.1f})")

    ax.set_xlabel("Mean CoT length (tokens per instance)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("CoT token length distribution — hinted vs unhinted", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
