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
