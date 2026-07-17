#!/usr/bin/env python3
"""
Length-bucketed g-mean^2 comparison between two models, POOLED over all
datasets and hint types.

Motivation
----------
The harmlessness-DPO model produces longer CoTs than base, and longer CoTs tend
to be more monitorable. This script disentangles that length confound from any
genuine training effect: it pools every per-instance g-mean^2 that ENTERS the
metric (Level-1 pass AND gmean2_i defined) across all dataset/hint slices, then
buckets instances by the mean token length of their 8 hinted responses
(`mean_cot_tokens_hinted`). Within each length bucket the two models are at ~the
same response length, so a difference there is closer to a pure training effect.

Only (dataset, hint) slices that exist for BOTH models are included, so the two
columns are always computed over the same slices.

Usage
-----
    python monitorability-eval/scripts/10_length_bucket_gmean2.py [BUCKET_WIDTH]

    BUCKET_WIDTH  bucket size in tokens (default: 500)

Run from the RO1 project root (the folder containing outputs-and-results/).
"""

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

# Make the metric module importable regardless of CWD.
_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))
from gmean2_metrics import compute_gmean2, load_jsonl, resolve_jsonl  # noqa: E402

# --- config -----------------------------------------------------------------
BASE_MODEL = "Qwen_Qwen3-4B"
HARM_MODEL = "er-abd_harmless-dpo-qwen3-4B-step-735"

ROOT = Path(__file__).resolve().parents[2]  # RO1/
OUT = ROOT / "outputs-and-results"
INF = OUT / "inference-outputs"
MON = OUT / "monitor_outputs"


def slice_exists(model: str, dataset: str, hint: str) -> bool:
    """A (dataset, hint) slice is usable if the model has hinted + unhinted
    inference (plain or .gz) AND a monitor_results.jsonl for it."""
    idir = INF / model / dataset / hint
    hinted_ok = (idir / "hinted.jsonl").exists() or (idir / "hinted.jsonl.gz").exists()
    unhinted_ok = (idir / "unhinted.jsonl").exists() or (idir / "unhinted.jsonl.gz").exists()
    monitor_ok = (MON / model / dataset / hint / "monitor_results.jsonl").exists()
    return hinted_ok and unhinted_ok and monitor_ok


def entering_instances(model: str, dataset: str, hint: str):
    """Return [(mean_hinted_tokens, gmean2_i), ...] for instances that enter the
    g-mean^2 average (Level-1 pass and gmean2_i defined) in this slice."""
    idir = INF / model / dataset / hint
    h = load_jsonl(resolve_jsonl(idir / "hinted.jsonl"))
    u = load_jsonl(resolve_jsonl(idir / "unhinted.jsonl"))
    m = load_jsonl(MON / model / dataset / hint / "monitor_results.jsonl")
    res = compute_gmean2(h, u, m)
    out = []
    for r in res["per_instance"]:
        if r["level1_pass"] and r["gmean2_i"] is not None and r["mean_cot_tokens_hinted"] is not None:
            out.append((r["mean_cot_tokens_hinted"], r["gmean2_i"]))
    return out


def discover_slices():
    """All (dataset, hint) slices present for BOTH models."""
    base_dir = MON / BASE_MODEL
    datasets = sorted(p.name for p in base_dir.iterdir() if p.is_dir())
    slices, skipped = [], []
    for ds in datasets:
        for hint in sorted(p.name for p in (base_dir / ds).iterdir() if p.is_dir()):
            if slice_exists(BASE_MODEL, ds, hint) and slice_exists(HARM_MODEL, ds, hint):
                slices.append((ds, hint))
            else:
                skipped.append((ds, hint))
    return slices, skipped


def main():
    width = int(sys.argv[1]) if len(sys.argv) > 1 else 500

    slices, skipped = discover_slices()
    print(f"Bucket width: {width} tokens")
    print(f"Included slices (present in both models): {len(slices)}")
    for ds, hint in slices:
        print(f"    {ds}/{hint}")
    if skipped:
        print(f"Skipped slices (missing in one model): {len(skipped)}")
        for ds, hint in skipped:
            print(f"    {ds}/{hint}")

    base, harm = [], []
    for ds, hint in slices:
        base.extend(entering_instances(BASE_MODEL, ds, hint))
        harm.extend(entering_instances(HARM_MODEL, ds, hint))

    def bucketize(data):
        b = defaultdict(list)
        for L, g in data:
            b[int(L // width) * width].append(g)
        return b

    bb, hh = bucketize(base), bucketize(harm)
    keys = sorted(set(bb) | set(hh))

    print(f"\n{'bucket (tok)':>15} | {'BASE n':>6} | {'BASE g2':>8} | {'HARM n':>6} | {'HARM g2':>8}")
    print("-" * 62)
    for k in keys:
        bn = len(bb.get(k, []))
        bg = f"{np.mean(bb[k]):.3f}" if k in bb else "  --  "
        hn = len(hh.get(k, []))
        hg = f"{np.mean(hh[k]):.3f}" if k in hh else "  --  "
        print(f"{k:>6}-{k + width:<8} | {bn:>6} | {bg:>8} | {hn:>6} | {hg:>8}")

    def summarize(data):
        L = np.array([x[0] for x in data])
        g = np.array([x[1] for x in data])
        return len(data), np.median(L), np.mean(g)

    bn, bmed, bg = summarize(base)
    hn, hmed, hg = summarize(harm)
    print("-" * 62)
    print(f"\nOverall: BASE n={bn}, median len={bmed:.0f}, g2={bg:.4f} "
          f"· HARM n={hn}, median len={hmed:.0f}, g2={hg:.4f}")


if __name__ == "__main__":
    main()
