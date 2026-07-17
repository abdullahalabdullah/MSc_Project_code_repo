#!/usr/bin/env python3
"""
Length-standardized g-mean^2 comparison between two models, POOLED over all
datasets and hint types, with bootstrap confidence intervals.

This is the rigorous way to separate a response-LENGTH confound from a genuine
training effect. The harmlessness-DPO model writes longer CoTs than base, and
longer CoTs tend to be more monitorable, so a raw g-mean^2 gap could be pure
length composition. Standardization removes that: it computes what the harmless
model WOULD score if its responses had base's length distribution.

Method (direct standardization / post-stratification)
-----------------------------------------------------
1. Bin every entering instance (Level-1 pass and gmean2_i defined) by
   `mean_cot_tokens_hinted` into fixed-width length bins.
2. HARM standardized-to-BASE = sum over bins of
       [ base's fraction of instances in the bin ] * [ HARM's mean g2 in the bin ]
   i.e. HARM's per-length monitorability, but weighted by BASE's length mix.
   (Only bins where BOTH models have instances contribute; the base weights are
   renormalised over those bins and the coverage is reported.)
3. Decompose the raw gap:
       total gap      = harm_raw - base_raw
       length-driven  = harm_raw - harm_std   (attributable to HARM's longer mix)
       training-driven= harm_std - base_raw    (remains at matched length)

Bootstrap: resample instances (the independent unit; each row already pools 8
samples into one gmean2_i) with replacement WITHIN each model, recompute
everything, take 2.5/97.5 percentiles. The base weights are re-estimated each
resample, so their sampling variability is included.

Only (dataset, hint) slices present for BOTH models are included.

Usage
-----
    python monitorability-eval/scripts/12_length_standardized_gmean2.py [BIN_WIDTH] [N_BOOT]

    BIN_WIDTH  standardization bin width in tokens (default: 500)
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

# --- config -----------------------------------------------------------------
BASE_MODEL = "Qwen_Qwen3-4B"
HARM_MODEL = "er-abd_harmless-dpo-qwen3-4B-step-735"
BOOT_SEED = 88

ROOT = Path(__file__).resolve().parents[2]  # RO1/
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
    return [(r["mean_cot_tokens_hinted"], r["gmean2_i"])
            for r in res["per_instance"]
            if r["level1_pass"] and r["gmean2_i"] is not None
            and r["mean_cot_tokens_hinted"] is not None]


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
    Returns (standardized_value, coverage) where coverage is the fraction of
    ref's weight that falls in bins where src also has instances."""
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


def compute_all(base, harm, width):
    base = np.asarray(base)
    harm = np.asarray(harm)
    base_raw = base[:, 1].mean()
    harm_raw = harm[:, 1].mean()
    harm_std, cov = standardize(harm, base, width)          # HARM at BASE's lengths
    base_std, cov_rev = standardize(base, harm, width)      # BASE at HARM's lengths (robustness)
    return {
        "base_raw": base_raw, "harm_raw": harm_raw,
        "harm_std": harm_std, "coverage": cov,
        "base_std": base_std, "coverage_rev": cov_rev,
        "total_gap": harm_raw - base_raw,
        "length_part": harm_raw - harm_std,
        "training_part": harm_std - base_raw,
    }


def main():
    width = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    n_boot = int(sys.argv[2]) if len(sys.argv) > 2 else 10000

    slices = discover_slices()
    base, harm = [], []
    for ds, hint in slices:
        base.extend(entering_instances(BASE_MODEL, ds, hint))
        harm.extend(entering_instances(HARM_MODEL, ds, hint))
    base = np.asarray(base)
    harm = np.asarray(harm)

    print(f"Standardization bin width: {width} tokens")
    print(f"Bootstrap resamples: {n_boot}")
    print(f"Slices pooled (both models): {len(slices)}")
    print(f"Entering instances: BASE n={len(base)}, HARM n={len(harm)}")

    pt = compute_all(base, harm, width)
    print(f"Base-weight coverage (harm->base): {pt['coverage']*100:.1f}% "
          f"| reverse (base->harm): {pt['coverage_rev']*100:.1f}%")

    # bootstrap
    rng = np.random.default_rng(BOOT_SEED)
    nb, nh = len(base), len(harm)
    keys = ["base_raw", "harm_raw", "harm_std", "base_std",
            "total_gap", "length_part", "training_part"]
    samp = {k: np.empty(n_boot) for k in keys}
    for i in range(n_boot):
        bb = base[rng.integers(0, nb, nb)]
        hh = harm[rng.integers(0, nh, nh)]
        r = compute_all(bb, hh, width)
        for k in keys:
            samp[k][i] = r[k]

    def ci(a):
        a = a[~np.isnan(a)]
        return np.percentile(a, 2.5), np.percentile(a, 97.5)

    print(f"\n{'quantity':<38}{'point':>8}   95% CI")
    print("-" * 70)
    labels = [
        ("base_raw",      "BASE raw g2"),
        ("harm_raw",      "HARM raw g2"),
        ("harm_std",      "HARM standardized -> BASE lengths"),
        ("base_std",      "BASE standardized -> HARM lengths"),
    ]
    for k, lab in labels:
        lo, hi = ci(samp[k])
        print(f"{lab:<38}{pt[k]:>8.3f}   [{lo:.3f}, {hi:.3f}]")
    print("-" * 70)
    print("Decomposition of the raw gap (HARM raw - BASE raw):")
    for k, lab in [("total_gap",     "  total gap"),
                   ("length_part",   "  length-driven  (harm_raw - harm_std)"),
                   ("training_part", "  training-driven (harm_std - base_raw)")]:
        lo, hi = ci(samp[k])
        flag = "  <-- excludes 0" if (lo > 0 or hi < 0) else "  (includes 0)"
        print(f"{lab:<40}{pt[k]:>+7.3f}   [{lo:+.3f}, {hi:+.3f}]{flag}")


if __name__ == "__main__":
    main()
