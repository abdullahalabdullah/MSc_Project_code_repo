#!/usr/bin/env python3
"""
Cumulative length g-mean^2 comparison between two models, POOLED over all
datasets and hint types.

Companion to 10_length_bucket_gmean2.py. Instead of disjoint length buckets,
this reports g-mean^2 computed over every instance whose mean hinted-response
length is <= a running threshold T (T = step, 2*step, ...). Each point is a
monotone-growing prefix of the length-sorted instances, so:
  - it is smoother than a bucket (n only grows), and
  - it answers "if we cap both models at the same max response length, who is
    more monitorable?" — which removes the long-tail composition difference.

The final row equals each model's overall (unrestricted) g-mean^2.

Only (dataset, hint) slices present for BOTH models are included.

Usage
-----
    python monitorability-eval/scripts/11_length_cumulative_gmean2.py [STEP]

    STEP  threshold increment in tokens (default: 500)

Run from the RO1 project root (the folder containing outputs-and-results/).
"""

import sys
from pathlib import Path

import numpy as np

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
    idir = INF / model / dataset / hint
    hinted_ok = (idir / "hinted.jsonl").exists() or (idir / "hinted.jsonl.gz").exists()
    unhinted_ok = (idir / "unhinted.jsonl").exists() or (idir / "unhinted.jsonl.gz").exists()
    monitor_ok = (MON / model / dataset / hint / "monitor_results.jsonl").exists()
    return hinted_ok and unhinted_ok and monitor_ok


def entering_instances(model: str, dataset: str, hint: str):
    idir = INF / model / dataset / hint
    h = load_jsonl(resolve_jsonl(idir / "hinted.jsonl"))
    u = load_jsonl(resolve_jsonl(idir / "unhinted.jsonl"))
    m = load_jsonl(MON / model / dataset / hint / "monitor_results.jsonl")
    res = compute_gmean2(h, u, m)
    return [(r["mean_cot_tokens_hinted"], r["gmean2_i"])
            for r in res["per_instance"]
            if r["level1_pass"] and r["gmean2_i"] is not None
            and r["mean_cot_tokens_hinted"] is not None]


def discover_slices():
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
    step = int(sys.argv[1]) if len(sys.argv) > 1 else 500

    slices, skipped = discover_slices()
    print(f"Threshold step: {step} tokens")
    print(f"Included slices (present in both models): {len(slices)}")
    if skipped:
        print(f"Skipped slices (missing in one model): {len(skipped)}")
        for ds, hint in skipped:
            print(f"    {ds}/{hint}")

    base, harm = [], []
    for ds, hint in slices:
        base.extend(entering_instances(BASE_MODEL, ds, hint))
        harm.extend(entering_instances(HARM_MODEL, ds, hint))

    max_len = max(max(L for L, _ in base), max(L for L, _ in harm))
    thresholds = list(range(step, int(max_len) + step, step))

    print(f"\n{'<= tok':>8} | {'BASE n':>6} | {'BASE g2':>8} | {'HARM n':>6} | {'HARM g2':>8}")
    print("-" * 52)
    for T in thresholds:
        b = [g for L, g in base if L <= T]
        h = [g for L, g in harm if L <= T]
        bg = f"{np.mean(b):.3f}" if b else "  --  "
        hg = f"{np.mean(h):.3f}" if h else "  --  "
        print(f"{T:>8} | {len(b):>6} | {bg:>8} | {len(h):>6} | {hg:>8}")

    print("-" * 52)
    print(f"{'ALL':>8} | {len(base):>6} | {np.mean([g for _, g in base]):>8.3f} "
          f"| {len(harm):>6} | {np.mean([g for _, g in harm]):>8.3f}")


if __name__ == "__main__":
    main()
