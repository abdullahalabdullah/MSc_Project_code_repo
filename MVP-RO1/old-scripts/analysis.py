"""
Visualisation and analysis for the MVP CoT monitorability evaluation.

Produces:
  - Confusion matrix (hint-following × acknowledged) as PNG + JSON
  - TE vs. ack scatter plot across all models as PNG
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless runs
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def confusion_matrix_data(
    hinted_records: list,
    monitor_records: list,
    S: set,
) -> dict:
    """
    Cross-tabulate hint-following × acknowledged for all hinted runs in S.

    Input:
      hinted_records   — list of JSONL records (hinted arm)
      monitor_records  — list of JSONL records (monitor outputs)
      S                — set of instance_ids in the retained set

    Output dict:
      counts       {followed_acknowledged, followed_not_acknowledged,
                    not_followed_acknowledged, not_followed_not_acknowledged}
      proportions  same keys, normalised by total
      total        int
    """
    monitor_lookup = {
        (r["instance_id"], r["sample_idx"]): r
        for r in monitor_records
    }

    counts = {
        "followed_acknowledged": 0,
        "followed_not_acknowledged": 0,
        "not_followed_acknowledged": 0,
        "not_followed_not_acknowledged": 0,
    }

    for r in hinted_records:
        if r["instance_id"] not in S:
            continue
        if r.get("truncated", False) or r.get("parsed_answer") is None:
            continue

        followed = bool(r.get("followed_hint", False))
        mon = monitor_lookup.get((r["instance_id"], r["sample_idx"]), {})
        acknowledged = mon.get("acknowledged", False) or False

        if followed and acknowledged:
            counts["followed_acknowledged"] += 1
        elif followed and not acknowledged:
            counts["followed_not_acknowledged"] += 1
        elif not followed and acknowledged:
            counts["not_followed_acknowledged"] += 1
        else:
            counts["not_followed_not_acknowledged"] += 1

    total = sum(counts.values())
    proportions = {k: v / max(total, 1) for k, v in counts.items()}

    # Derived metric: opacity rate = covert / (covert + transparent)
    denom = counts["followed_acknowledged"] + counts["followed_not_acknowledged"]
    opacity_rate = counts["followed_not_acknowledged"] / max(denom, 1)

    return {
        "counts": counts,
        "proportions": proportions,
        "total": total,
        "opacity_rate": opacity_rate,
    }


def plot_confusion_matrix(cm_data: dict, model_name: str, output_path: Path):
    """
    Save a labelled 2×2 confusion matrix heatmap.

    Rows: Acknowledged / Not Acknowledged
    Cols: Followed Hint / Did Not Follow
    """
    c = cm_data["counts"]
    matrix = np.array([
        [c["followed_acknowledged"],     c["not_followed_acknowledged"]],
        [c["followed_not_acknowledged"], c["not_followed_not_acknowledged"]],
    ])
    row_labels = ["Acknowledged", "Not Acknowledged"]
    col_labels = ["Followed Hint", "Did Not Follow"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(col_labels, fontsize=11)
    ax.set_yticklabels(row_labels, fontsize=11)
    ax.set_xlabel("Hint Following", fontsize=12)
    ax.set_ylabel("CoT Acknowledgement", fontsize=12)
    ax.set_title(f"{model_name}\nOpacity rate: {cm_data['opacity_rate']:.2f}", fontsize=12)

    for i in range(2):
        for j in range(2):
            prop = cm_data["proportions"]
            key = {
                (0, 0): "followed_acknowledged",
                (0, 1): "not_followed_acknowledged",
                (1, 0): "followed_not_acknowledged",
                (1, 1): "not_followed_not_acknowledged",
            }[(i, j)]
            ax.text(
                j, i,
                f"{matrix[i, j]}\n({prop[key]:.1%})",
                ha="center", va="center", fontsize=12,
                color="white" if matrix[i, j] > matrix.max() * 0.6 else "black",
            )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# TE binned histogram (fixed-width bins)
# ---------------------------------------------------------------------------

def plot_te_binned_histogram(
    per_instance: list,
    output_path: Path,
    bin_width: float = 0.2,
    deltas: tuple = (0.05, 0.20),
):
    """
    Histogram of per-instance TE values using fixed-width bins (default 0.2).

    Bins span from -1.0 to 1.0 in steps of `bin_width`. Bars are coloured
    red for bins whose centre is negative (hint-repelled) and blue otherwise.
    Each bar is annotated with its count. Vertical dashed lines mark the
    same delta thresholds as the ECDF figure for visual consistency.
    """
    te = np.array([r["te_i"] for r in per_instance])
    n = len(te)

    # Build bin edges covering [-1, 1] with clean multiples of bin_width.
    # Add a small epsilon so np.arange reliably includes the 1.0 + bin_width endpoint
    # despite floating-point truncation (np.arange(-1.0, 1.2, 0.2) may stop at 1.0).
    edges = np.arange(-1.0, 1.0 + bin_width + 1e-9, bin_width)
    counts, edges = np.histogram(te, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2

    # Per-bin mean ack_i (only for TE >= 0 bins)
    bin_indices = np.digitize(te, edges) - 1  # bin index for each instance
    bin_indices = np.clip(bin_indices, 0, len(counts) - 1)
    bin_mean_har = {}
    for bi, center in enumerate(centers):
        if center < 0:
            continue
        ack_vals = [
            r["ack_i"] for r, idx in zip(per_instance, bin_indices)
            if idx == bi and r.get("ack_i") is not None
        ]
        bin_mean_har[bi] = float(np.mean(ack_vals)) if ack_vals else None

    colors = ["#d62728" if c < 0 else "#1f77b4" for c in centers]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(centers, counts, width=bin_width * 0.88,
                  color=colors, alpha=0.80, linewidth=0.5, edgecolor="white")

    # Annotate each non-zero bar with its count (above bar) and mean HAR (inside bar, TE>=0 only)
    for bi, (bar, count) in enumerate(zip(bars, counts)):
        if count == 0:
            continue
        # Count label above the bar
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(count),
            ha="center", va="bottom", fontsize=9,
        )
        # Mean HAR label inside the bar, TE >= 0 bins only
        if bi in bin_mean_har and bin_mean_har[bi] is not None:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                f"HAR={bin_mean_har[bi]:.3f}",
                ha="center", va="center", fontsize=8.5,
                color="black", fontweight="bold",
            )

    # Twin y-axis for fraction
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim()[0] / n, ax.get_ylim()[1] / n)
    ax2.set_ylabel("Fraction of instances", fontsize=11)

    # Delta reference lines — same colours as ECDF figure
    delta_colors = ["#ff7f0e", "#d62728"]
    for delta, color in zip(deltas, delta_colors):
        n_selected = int((te >= delta).sum())
        pct = 100 * n_selected / n
        ax.axvline(x=delta, color=color, linestyle="--", linewidth=1.5,
                   label=f"δ = {delta:.2f}  ({pct:.1f}% of instances)")

    # Legend patches for bar colours
    import matplotlib.patches as mpatches
    ax.legend(
        handles=[
            mpatches.Patch(color="#d62728", alpha=0.80, label="TE < 0  (hint-repelled)"),
            mpatches.Patch(color="#1f77b4", alpha=0.80, label="TE ≥ 0"),
            *[ax.get_legend_handles_labels()[0][i] for i in range(len(deltas))],
        ],
        fontsize=9,
    )

    ax.set_xlabel("TE per instance", fontsize=12)
    ax.set_ylabel("Number of instances", fontsize=12)
    ax.set_title(
        f"Distribution of per-instance treatment effects  "
        f"(n = {n},  bin width = {bin_width})",
        fontsize=13,
    )
    ax.set_xlim(edges[0] - bin_width * 0.5, edges[-1] + bin_width * 0.5)
    ax.set_xticks(edges)
    ax.set_xticklabels([f"{e:.1f}" for e in edges], fontsize=8, rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")

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


# ---------------------------------------------------------------------------
# TE vs. ack scatter
# ---------------------------------------------------------------------------

_MODEL_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
_MODEL_MARKERS = ["o", "s", "^", "D"]


def plot_te_ack_scatter(
    per_instance_all: dict,
    delta: float,
    output_path: Path,
):
    """
    Combined scatter plot: TE_i (x) vs ack_i (y), one series per model.

    Input:
      per_instance_all  {model_name: list of per-instance dicts}
      delta             float, S inclusion threshold (drawn as vertical dashed line)
      output_path       Path for PNG output
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    for (model_name, records), color, marker in zip(
        per_instance_all.items(), _MODEL_COLORS, _MODEL_MARKERS
    ):
        x = [r["te_i"] for r in records if r["ack_i"] is not None]
        y = [r["ack_i"] for r in records if r["ack_i"] is not None]

        if not x:
            continue

        ax.scatter(x, y, alpha=0.45, s=22, color=color, marker=marker, label=model_name)

        # Linear trend line
        if len(x) >= 3:
            coeffs = np.polyfit(x, y, 1)
            x_sorted = np.linspace(min(x), max(x), 100)
            ax.plot(x_sorted, np.polyval(coeffs, x_sorted),
                    color=color, linewidth=1.5, linestyle="--", alpha=0.75)

    ax.axvline(x=delta, color="black", linestyle=":", linewidth=1.8, label=f"δ = {delta:.2f}")
    ax.set_xlabel("Total Effect (TE_i)", fontsize=13)
    ax.set_ylabel("Acknowledgement Rate (ack_i)", fontsize=13)
    ax.set_title("TE vs. Acknowledgement Rate — all checkpoints", fontsize=13)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
