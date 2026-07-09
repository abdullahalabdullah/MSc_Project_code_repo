#!/usr/bin/env python3
"""
Compare monitor outputs across all datasets for three monitors:
  - monitor_outputs-1  (gpt-4o)
  - monitor_outputs-2  (gpt-5-mini)
  - qwen3_1_7b         (local professor monitor)

Note the two layouts:
  monitor_outputs-{1,2}/{dataset}/{MODEL}/monitor_results.jsonl
  qwen3_1_7b/{dataset}/{hint_type}/monitor_results.jsonl

For each pair of monitors and each dataset:
  1. Align records by (instance_id, sample_idx, arm); report removed counts.
  2. Compute exact-match agreement on `followed_hint` and `acknowledged`.

Saves results to monitor-validation/monitor_comparison.md.
"""

import json
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).parent

MODEL = "qwen3_1_7b"      # subject model whose outputs are being monitored
HINT_TYPE = "professor"   # hint-type subdir under the local-monitor layout

# Each monitor: display name -> function mapping a dataset to its results file.
# Local monitors under qwen3_1_7b-new use the layout {dataset}/{HINT_TYPE}/{monitor}.
NEW_ROOT = ROOT / "qwen3_1_7b-new"
MONITORS = {
    "gpt-5-mini (mon-2)": lambda ds: ROOT / "monitor_outputs-2" / ds / MODEL / "monitor_results.jsonl",
    "gemma-4-E4B (local)": lambda ds: NEW_ROOT / ds / HINT_TYPE / "gemma_e4b" / "monitor_results.jsonl",
    "qwen3_8b (local)": lambda ds: NEW_ROOT / ds / HINT_TYPE / "qwen3_8b" / "monitor_results.jsonl",
}


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def key(record: dict) -> tuple:
    return (record["instance_id"], record["sample_idx"], record["arm"])


def find_datasets() -> list[str]:
    # Datasets present for the new local monitors (the newest, possibly-partial set).
    return sorted(
        {p.parent.parent.parent.name
         for p in NEW_ROOT.glob(f"*/{HINT_TYPE}/*/monitor_results.jsonl")}
    )


def compare_dataset(name_a: str, name_b: str, dataset: str) -> dict:
    pa = MONITORS[name_a](dataset)
    pb = MONITORS[name_b](dataset)

    recs_a = {key(r): r for r in load_jsonl(pa)}
    recs_b = {key(r): r for r in load_jsonl(pb)}

    common = recs_a.keys() & recs_b.keys()
    n = len(common)

    fh_matches = sum(recs_a[k]["followed_hint"] == recs_b[k]["followed_hint"] for k in common)
    ack_matches = sum(recs_a[k]["acknowledged"] == recs_b[k]["acknowledged"] for k in common)

    return {
        "dataset": dataset,
        "n_a": len(recs_a),
        "n_b": len(recs_b),
        "only_in_a": len(recs_a) - n,
        "only_in_b": len(recs_b) - n,
        "n_compared": n,
        "followed_hint_matches": fh_matches,
        "followed_hint_accuracy": fh_matches / n if n else float("nan"),
        "acknowledged_matches": ack_matches,
        "acknowledged_accuracy": ack_matches / n if n else float("nan"),
    }


def render_pair(name_a: str, name_b: str, datasets: list[str]) -> tuple[list[str], dict]:
    results = [compare_dataset(name_a, name_b, ds) for ds in datasets]

    total_n = sum(r["n_compared"] for r in results)
    total_fh = sum(r["followed_hint_matches"] for r in results)
    total_ack = sum(r["acknowledged_matches"] for r in results)
    total_only_a = sum(r["only_in_a"] for r in results)
    total_only_b = sum(r["only_in_b"] for r in results)

    lines = [
        f"## `{name_a}` vs `{name_b}`",
        "",
        "### Overlap & removal",
        "",
        f"| Dataset | {name_a} records | {name_b} records | Only in A | Only in B | Compared |",
        "| ------- | ---------------- | ---------------- | --------- | --------- | -------- |",
    ]
    for r in results:
        lines.append(
            f"| {r['dataset']} | {r['n_a']} | {r['n_b']} "
            f"| {r['only_in_a']} | {r['only_in_b']} | {r['n_compared']} |"
        )
    lines += [
        f"| **Total** | {sum(r['n_a'] for r in results)} | {sum(r['n_b'] for r in results)} "
        f"| {total_only_a} | {total_only_b} | **{total_n}** |",
        "",
        "### Agreement per dataset",
        "",
        "| Dataset | followed_hint agreement | acknowledged agreement |",
        "| ------- | ----------------------- | ---------------------- |",
    ]
    for r in results:
        lines.append(
            f"| {r['dataset']} "
            f"| {r['followed_hint_accuracy']:.1%} ({r['followed_hint_matches']}/{r['n_compared']}) "
            f"| {r['acknowledged_accuracy']:.1%} ({r['acknowledged_matches']}/{r['n_compared']}) |"
        )
    avg_fh = total_fh / total_n if total_n else float("nan")
    avg_ack = total_ack / total_n if total_n else float("nan")
    lines += [
        f"| **Average** | **{avg_fh:.1%}** ({total_fh}/{total_n}) "
        f"| **{avg_ack:.1%}** ({total_ack}/{total_n}) |",
        "",
    ]

    summary = {
        "pair": f"{name_a} vs {name_b}",
        "n": total_n,
        "fh": avg_fh,
        "ack": avg_ack,
    }
    return lines, summary


def main():
    datasets = find_datasets()

    lines = [
        "# Monitor Comparison",
        "",
        f"Subject model: `{MODEL}`",
        "",
        "Monitors compared: " + ", ".join(f"`{n}`" for n in MONITORS),
        "",
    ]

    summaries = []
    pair_blocks = []
    for name_a, name_b in combinations(MONITORS, 2):
        block, summary = render_pair(name_a, name_b, datasets)
        pair_blocks.append(block)
        summaries.append(summary)

    # Headline summary table across all pairs.
    lines += [
        "## Summary (overall agreement)",
        "",
        "| Pair | followed_hint | acknowledged | Compared |",
        "| ---- | ------------- | ------------ | -------- |",
    ]
    for s in summaries:
        lines.append(
            f"| {s['pair']} | {s['fh']:.1%} | {s['ack']:.1%} | {s['n']} |"
        )
    lines.append("")

    for block in pair_blocks:
        lines += block

    md = "\n".join(lines)
    OUT_MD = ROOT / "monitor_comparison.md"
    OUT_MD.write_text(md, encoding="utf-8")
    print(md)
    print(f"\nSaved to {OUT_MD}")


if __name__ == "__main__":
    main()
