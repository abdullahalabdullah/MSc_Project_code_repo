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
# TE distribution
# ---------------------------------------------------------------------------

def plot_te_distribution(
    per_instance: list,
    output_path: Path,
    delta_5pct: float = 0.05,
    delta_25pct: float = 0.20,
):
    """
    Bar chart of TE_i per example, sorted ascending by TE value.
    Horizontal lines mark the 5% and 25% thresholds.
    """
    te_values = sorted(r["te_i"] for r in per_instance)
    n = len(te_values)
    mean_te = float(np.mean(te_values))
    var_te  = float(np.var(te_values))
    std_te  = float(np.std(te_values))

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(range(n), te_values, width=1.0, color="#1f77b4", alpha=0.70, linewidth=0)
    ax.axhline(y=delta_5pct,  color="#ff7f0e", linestyle="--", linewidth=1.6,
               label=f"5% threshold (δ={delta_5pct:.2f})")
    ax.axhline(y=delta_25pct, color="#d62728", linestyle="-",  linewidth=1.6,
               label=f"25% threshold (δ={delta_25pct:.2f})")
    ax.axhline(y=mean_te,     color="black",   linestyle=":",  linewidth=1.4,
               label=f"Population mean TE = {mean_te:.3f}")

    ax.set_xlabel("Example (sorted by TE)", fontsize=12)
    ax.set_ylabel("TE$_i$", fontsize=12)
    ax.set_title(
        f"TE distribution over all examples  "
        f"(n={n},  mean={mean_te:.3f},  var={var_te:.4f},  std={std_te:.4f})",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.set_xlim(-1, n)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()

    return {"mean_TE": mean_te, "var_TE": var_te, "std_TE": std_te}


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
