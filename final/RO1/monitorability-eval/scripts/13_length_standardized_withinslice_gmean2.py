#!/usr/bin/env python3
"""
Within-slice length-standardized g-mean^2 comparison (strictest confound
control), with bootstrap confidence intervals.

This is the stricter sibling of 12_length_standardized_gmean2.py. The pooled
version (12) adjusts for response LENGTH only; a pooled length bin still mixes
instances from different datasets/hints, which have different baseline
monitorability, so dataset composition can leak into a within-bin comparison.
This script closes that loophole: it standardizes length *inside each
(dataset, hint) slice* and then aggregates, so BOTH length AND dataset/hint are
held fixed. It answers: "within the same task, at the same response length, does
the monitorability difference remain?"

Method
------
For each (dataset, hint) slice, over entering instances (Level-1 pass and
gmean2_i defined):
  base_raw_slice = mean g2 over BASE instances in the slice
  harm_raw_slice = mean g2 over HARM instances in the slice
  harm_std_slice = HARM's g2 reweighted to BASE's WITHIN-SLICE length distribution
                 = sum over length bins of base_weight_in_slice[bin]*harm_mean[bin]
                   (bins where both models have instances; base weights renormalised
                    over covered bins -> per-slice coverage reported)

Aggregate across slices with weight w_slice = base_n_slice / total_base_n (so the
reference is BASE's joint (slice, length) population):
  base_raw   = sum w_slice * base_raw_slice           (= pooled base mean)
  harm_std   = sum w_slice * harm_std_slice           (HARM at base's joint mix)
  harm_raw   = pooled HARM mean (its own natural mix)

Decomposition of the raw gap:
  total gap            = harm_raw - base_raw
  confound-driven      = harm_raw - harm_std   (length + dataset/hint composition)
  training-driven      = harm_std - base_raw   (remains with slice AND length fixed)

Bootstrap: STRATIFIED resample — within each (model, slice) resample instances
with replacement keeping the per-slice n fixed — recompute, 2.5/97.5 percentiles.

Only (dataset, hint) slices present for BOTH models are included.

Usage
-----
    python monitorability-eval/scripts/13_length_standardized_withinslice_gmean2.py [BIN_WIDTH] [N_BOOT]

    BIN_WIDTH  within-slice length bin width in tokens (default: 500)
    N_BOOT     bootstrap resamples (default: 10000)

Run from the RO1 project root (the folder containing outputs-and-results/).
"""

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))
from gmean2_metrics import compute_gmean2, load_jsonl, resolve_jsonl  # noqa: E402

BASE_MODEL = "Qwen_Qwen3-4B"
HARM_MODEL = "er-abd_harmless-dpo-qwen3-4B-step-735"
BOOT_SEED = 88

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs-and-results"
INF = OUT / "inference-outputs"
MON = OUT / "monitor_outputs"


def slice_exists(model, dataset, hint):
    idir = INF / model / dataset / hint
    hinted_ok = (idir / "hinted.jsonl").exists() or (idir / "hinted.jsonl.gz").exists()
    unhinted_ok = (idir / "unhinted.jsonl").exists() or (idir / "unhinted.jsonl.gz").exists()
    monitor_ok = (MON / model / dataset / hint / "monitor_results.jsonl").exists()
    return hinted_ok and unhinted_ok and monitor_ok


def entering_instances(model, dataset, hint):
    idir = INF / model / dataset / hint
    h = load_jsonl(resolve_jsonl(idir / "hinted.jsonl"))
    u = load_jsonl(resolve_jsonl(idir / "unhinted.jsonl"))
    m = load_jsonl(MON / model / dataset / hint / "monitor_results.jsonl")
    res = compute_gmean2(h, u, m)
    rows = [(r["mean_cot_tokens_hinted"], r["gmean2_i"])
            for r in res["per_instance"]
            if r["level1_pass"] and r["gmean2_i"] is not None
            and r["mean_cot_tokens_hinted"] is not None]
    return np.asarray(rows, dtype=float).reshape(-1, 2)


def discover_slices():
    base_dir = MON / BASE_MODEL
    datasets = sorted(p.name for p in base_dir.iterdir() if p.is_dir())
    slices = []
    for ds in datasets:
        for hint in sorted(p.name for p in (base_dir / ds).iterdir() if p.is_dir()):
            if slice_exists(BASE_MODEL, ds, hint) and slice_exists(HARM_MODEL, ds, hint):
                slices.append((ds, hint))
    return slices


def standardize(src, ref, width):
    """Mean g2 of `src` reweighted to `ref`'s length distribution.
    Returns (value, coverage). coverage = fraction of ref weight in bins where
    src also has instances."""
    src_bin = defaultdict(list)
    for L, g in src:
        src_bin[int(L // width) * width].append(g)
    src_mean = {k: np.mean(v) for k, v in src_bin.items()}
    ref_cnt = defaultdict(int)
    for L, _ in ref:
        ref_cnt[int(L // width) * width] += 1
    n_ref = len(ref)
    num = cov = 0.0
    for k, c in ref_cnt.items():
        if k in src_mean:
            w = c / n_ref
            num += w * src_mean[k]
            cov += w
    return (num / cov if cov > 0 else np.nan), cov


def compute_all(base_by_slice, harm_by_slice, width):
    """base_by_slice / harm_by_slice: dict slice_key -> ndarray[(L,g)].

    All decomposition quantities are computed over the COVERED slice set (slices
    where HARM has instances and a within-slice standardized value is defined),
    with base slice weights renormalised over that set. This keeps base_raw,
    harm_raw and harm_std on the SAME slices — otherwise base_raw would include
    a slice (e.g. mmlu_pro/unethical) that harm_std cannot cover, biasing the gap.
    slice_coverage reports the base-weight fraction that is covered."""
    total_base = sum(len(v) for v in base_by_slice.values())

    # First pass: per-slice pieces on covered slices only.
    covered = []  # (slice_key, base_n, base_raw_slice, harm_raw_slice, harm_std_slice, cov_len)
    for k, b in base_by_slice.items():
        h = harm_by_slice.get(k)
        if len(b) == 0 or h is None or len(h) == 0:
            continue
        hstd, cov_len = standardize(h, b, width)
        if np.isnan(hstd):
            continue
        covered.append((k, len(b), b[:, 1].mean(), h[:, 1].mean(), hstd, cov_len, len(h)))

    slice_coverage = sum(c[1] for c in covered) / total_base if total_base else np.nan
    if not covered:
        return {"base_raw": np.nan, "harm_raw": np.nan, "harm_std": np.nan,
                "slice_coverage": 0.0, "length_coverage": np.nan,
                "total_gap": np.nan, "confound_part": np.nan, "training_part": np.nan}

    base_w = np.array([c[1] for c in covered], dtype=float)
    base_w /= base_w.sum()                       # base slice weights over covered set
    harm_n = np.array([c[6] for c in covered], dtype=float)
    harm_w = harm_n / harm_n.sum()               # harm's own slice weights over covered set

    base_raw = float(np.sum(base_w * np.array([c[2] for c in covered])))
    harm_raw = float(np.sum(harm_w * np.array([c[3] for c in covered])))
    harm_std = float(np.sum(base_w * np.array([c[4] for c in covered])))
    len_cov = float(np.sum(base_w * np.array([c[5] for c in covered])))

    return {
        "base_raw": base_raw, "harm_raw": harm_raw, "harm_std": harm_std,
        "slice_coverage": slice_coverage, "length_coverage": len_cov,
        "total_gap": harm_raw - base_raw,
        "confound_part": harm_raw - harm_std,
        "training_part": harm_std - base_raw,
    }


def main():
    width = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    n_boot = int(sys.argv[2]) if len(sys.argv) > 2 else 10000

    slices = discover_slices()
    base_by_slice, harm_by_slice = {}, {}
    for ds, hint in slices:
        base_by_slice[(ds, hint)] = entering_instances(BASE_MODEL, ds, hint)
        harm_by_slice[(ds, hint)] = entering_instances(HARM_MODEL, ds, hint)

    nb = sum(len(v) for v in base_by_slice.values())
    nh = sum(len(v) for v in harm_by_slice.values())
    print(f"Within-slice length bin width: {width} tokens")
    print(f"Bootstrap resamples: {n_boot}")
    print(f"Slices pooled (both models): {len(slices)}")
    print(f"Entering instances: BASE n={nb}, HARM n={nh}")

    pt = compute_all(base_by_slice, harm_by_slice, width)
    print(f"Slice coverage: {pt['slice_coverage']*100:.1f}% of base weight "
          f"| mean within-slice length coverage: {pt['length_coverage']*100:.1f}%")

    # stratified bootstrap
    rng = np.random.default_rng(BOOT_SEED)
    keys = ["base_raw", "harm_raw", "harm_std", "total_gap", "confound_part", "training_part"]
    samp = {k: np.empty(n_boot) for k in keys}
    bkeys = list(base_by_slice)
    for i in range(n_boot):
        bb, hh = {}, {}
        for k in bkeys:
            b = base_by_slice[k]
            h = harm_by_slice[k]
            bb[k] = b[rng.integers(0, len(b), len(b))]
            hh[k] = h[rng.integers(0, len(h), len(h))] if len(h) else h
        r = compute_all(bb, hh, width)
        for kk in keys:
            samp[kk][i] = r[kk]

    def ci(a):
        a = a[~np.isnan(a)]
        return np.percentile(a, 2.5), np.percentile(a, 97.5)

    print(f"\n{'quantity':<44}{'point':>8}   95% CI")
    print("-" * 74)
    for k, lab in [("base_raw", "BASE raw g2 (covered slices)"),
                   ("harm_raw", "HARM raw g2 (covered slices)"),
                   ("harm_std", "HARM standardized -> BASE (slice,length)")]:
        lo, hi = ci(samp[k])
        print(f"{lab:<44}{pt[k]:>8.3f}   [{lo:.3f}, {hi:.3f}]")
    print("-" * 74)
    print("Decomposition of the raw gap (HARM raw - BASE raw):")
    for k, lab in [("total_gap",     "  total gap"),
                   ("confound_part", "  confound-driven (length+dataset/hint)"),
                   ("training_part", "  training-driven (slice+length fixed)")]:
        lo, hi = ci(samp[k])
        flag = "  <-- excludes 0" if (lo > 0 or hi < 0) else "  (includes 0)"
        print(f"{lab:<42}{pt[k]:>+7.3f}   [{lo:+.3f}, {hi:+.3f}]{flag}")


if __name__ == "__main__":
    main()
