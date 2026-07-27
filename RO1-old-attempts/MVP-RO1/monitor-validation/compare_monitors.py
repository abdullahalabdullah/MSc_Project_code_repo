#!/usr/bin/env python3
"""
Compare two sets of monitor outputs (monitor_outputs-1 vs monitor_outputs-2)
across all datasets.

For each dataset's monitor_results.jsonl pair:
  1. Align records by (instance_id, sample_idx, arm); report removed counts.
  2. Compute exact-match agreement on `followed_hint` and `acknowledged`.

Saves results to monitor-validation/monitor_comparison.md.
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent
MON1 = ROOT / "monitor_outputs-1"
MON2 = ROOT / "monitor_outputs-2"
OUT_MD = ROOT / "monitor_comparison.md"

MODEL = "qwen3_1_7b"


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def key(record: dict) -> tuple:
    return (record["instance_id"], record["sample_idx"], record["arm"])


def compare_dataset(dataset: str) -> dict:
    p1 = MON1 / dataset / MODEL / "monitor_results.jsonl"
    p2 = MON2 / dataset / MODEL / "monitor_results.jsonl"

    recs1 = {key(r): r for r in load_jsonl(p1)}
    recs2 = {key(r): r for r in load_jsonl(p2)}

    only_in_1 = len(recs1) - len(recs1.keys() & recs2.keys())
    only_in_2 = len(recs2) - len(recs1.keys() & recs2.keys())
    common_keys = recs1.keys() & recs2.keys()
    n = len(common_keys)

    fh_matches = sum(
        recs1[k]["followed_hint"] == recs2[k]["followed_hint"] for k in common_keys
    )
    ack_matches = sum(
        recs1[k]["acknowledged"] == recs2[k]["acknowledged"] for k in common_keys
    )

    return {
        "dataset": dataset,
        "n1": len(recs1),
        "n2": len(recs2),
        "removed_from_1": only_in_1,
        "removed_from_2": only_in_2,
        "n_compared": n,
        "followed_hint_matches": fh_matches,
        "followed_hint_accuracy": fh_matches / n if n else float("nan"),
        "acknowledged_matches": ack_matches,
        "acknowledged_accuracy": ack_matches / n if n else float("nan"),
    }


def main():
    datasets = sorted(
        {p.parent.parent.name for p in MON1.glob(f"*/{MODEL}/monitor_results.jsonl")}
    )

    results = [compare_dataset(ds) for ds in datasets]

    total_n = sum(r["n_compared"] for r in results)
    total_fh = sum(r["followed_hint_matches"] for r in results)
    total_ack = sum(r["acknowledged_matches"] for r in results)
    total_removed_1 = sum(r["removed_from_1"] for r in results)
    total_removed_2 = sum(r["removed_from_2"] for r in results)

    lines = [
        "# Monitor Comparison: monitor_outputs-1 vs monitor_outputs-2",
        "",
        f"Model: `{MODEL}`",
        "",
        "## Overlap & removal",
        "",
        "| Dataset | Mon-1 records | Mon-2 records | Removed from Mon-1 | Removed from Mon-2 | Compared |",
        "| ------- | ------------- | ------------- | ------------------ | ------------------ | -------- |",
    ]
    for r in results:
        lines.append(
            f"| {r['dataset']} | {r['n1']} | {r['n2']} "
            f"| {r['removed_from_1']} | {r['removed_from_2']} | {r['n_compared']} |"
        )
    lines += [
        f"| **Total** | {sum(r['n1'] for r in results)} | {sum(r['n2'] for r in results)} "
        f"| {total_removed_1} | {total_removed_2} | **{total_n}** |",
        "",
        "## Agreement per dataset",
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

    md = "\n".join(lines)
    OUT_MD.write_text(md, encoding="utf-8")
    print(md)
    print(f"\nSaved to {OUT_MD}")


if __name__ == "__main__":
    main()
