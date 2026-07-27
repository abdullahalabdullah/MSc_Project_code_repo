# Re-run (2026-07-27): length-standardized g-mean² — Base vs Harmless-DPO (step 735), full 24/24 slices

**Why this re-run exists.** The original scripts-12/13 analyses for base vs
harmless-735 ([pooled](standardized_length_base_vs_harmless_step735.md),
[within-slice](withinslice_standardized_base_vs_harmless_step735.md)) were run
**before** the `mmlu_pro × unethical` monitor gap was repaired, so HARM
contributed no instances there (23 covered slices; HARM n = 2,451). The cell has
since been repaired (unhinted monitor verdicts restored ~2026-07-20 by reusing
the hint-type-invariant unhinted arm; the cell now scores **g² = 0.9442,
n = 187 Level-1 instances**, and the 2026-07-21 `main_results_*` tables already
include it). This note re-runs both analyses on the complete data
(**24/24 slices; HARM n = 2,638**). **Quote these numbers in the write-up, not
the 23-slice ones.**

## Method fidelity

Identical methodology, executed through the repo's own scripts
(`12_length_standardized_gmean2.py`, `13_length_standardized_withinslice_gmean2.py`;
seed 88, 10,000 resamples, same slice iteration order). The per-instance
`(mean_cot_tokens_hinted, gmean2_i)` inputs were taken from each cell's
`results_per_instance.csv` — written by `04_compute_gmean2.py` from the same
`compute_gmean2` per-instance rows the scripts recompute from raw jsonl — with
only the two I/O functions (`discover_slices`, `entering_instances`) redirected
to that cache. Harness validated by exact reproduction:

- **Anti-syc anchor (unchanged data), script 13 @ 500/10k:** reproduces the
  published note **digit-for-digit including all CIs** (0.768 [0.757, 0.779] /
  0.793 [0.782, 0.804] / 0.767 [0.750, 0.785]; +0.025 [+0.010, +0.041] /
  +0.026 [+0.012, +0.040] / −0.001 [−0.020, +0.019]).
- **Pre-repair harmless state (HARM slice emptied), scripts 12+13:** reproduces
  both old notes' point estimates exactly (13: 0.754/0.789/0.773,
  +0.035/+0.016/+0.019, 92.8%/96.7%; 12: 0.768/0.789/0.793/0.770,
  +0.021/−0.004/+0.025, BASE n=2750 / HARM n=2451).

Since the on-disk data is now complete, running the scripts directly on the repo
(`python3 monitorability-eval/scripts/13_length_standardized_withinslice_gmean2.py 500 10000`,
same for 1000, and script 12) should reproduce the numbers below from raw jsonl.

## Results — script 13, within-slice (strictest control), width 500, 10,000 boots

Slices 24/24 (100.0% of base weight); mean within-slice length coverage 96.6%.
Entering instances: BASE n = 2,750, HARM n = 2,638.

| quantity | point | 95% CI |
|---|---|---|
| BASE raw g² (covered slices) | 0.768 | [0.757, 0.779] |
| HARM raw g² (covered slices) | 0.800 | [0.789, 0.811] |
| **HARM standardized → BASE (slice, length)** | **0.784** | **[0.769, 0.800]** |

**Decomposition of the raw gap (HARM − BASE = +0.032):**

| component | point | 95% CI | verdict |
|---|---|---|---|
| total gap | **+0.032** | [+0.017, +0.048] | excludes 0 |
| confound-driven (length + dataset/hint) | **+0.016** | [+0.003, +0.029] | excludes 0 |
| **training-driven (slice + length fixed)** | **+0.016** | **[−0.003, +0.036]** | **includes 0 (borderline)** |

Robustness at width 1000 (10,000 boots; length coverage 98.9%): total +0.032
[+0.017, +0.048], confound +0.016 [+0.005, +0.027], **training +0.017
[−0.001, +0.034]** — same picture.

## Results — script 12, pooled standardization, width 500, 10,000 boots

Base-weight coverage (harm→base) 99.6%; reverse 99.8%.

| quantity | point | 95% CI |
|---|---|---|
| BASE raw g² | 0.768 | [0.755, 0.781] |
| HARM raw g² | 0.800 | [0.788, 0.812] |
| **HARM standardized → BASE lengths** | **0.803** | **[0.790, 0.816]** |
| BASE standardized → HARM lengths (robustness) | 0.770 | [0.756, 0.784] |

**Decomposition of the raw gap (HARM − BASE = +0.032):**

| component | point | 95% CI | verdict |
|---|---|---|---|
| total gap | **+0.032** | [+0.015, +0.051] | excludes 0 |
| length-driven (harm_raw − harm_std) | **−0.003** | [−0.009, +0.003] | includes 0 |
| **training-driven (harm_std − base_raw)** | **+0.035** | **[+0.017, +0.054]** | excludes 0 |

## Old (23-slice) vs new (24/24) at a glance

| quantity | old | new (quote this) |
|---|---|---|
| 13 within-slice: total gap | +0.035 [+0.018, +0.052] (covered set) | +0.032 [+0.017, +0.048] |
| 13 within-slice: confound-driven | +0.016 [+0.002, +0.030] | +0.016 [+0.003, +0.029] |
| 13 within-slice: **training-driven** | +0.019 [−0.001, +0.039] | **+0.016 [−0.003, +0.036]** |
| 12 pooled: total gap | +0.021 [+0.003, +0.040] | +0.032 [+0.015, +0.051] |
| 12 pooled: length-driven | −0.004 [−0.010, +0.003] | −0.003 [−0.009, +0.003] |
| 12 pooled: **training-driven** | +0.025 [+0.006, +0.044] | **+0.035 [+0.017, +0.054]** |
| HARM entering instances | 2,451 | 2,638 |
| Slice coverage (13) | 92.8% (23/24) | 100% (24/24) |

## Interpretation

1. **The conclusions are unchanged, now on complete data.** Length explains
   essentially none of the raw gap (pooled length-driven −0.003, tight CI around
   0, despite the ~1.2–1.3× median-length inflation).
2. **The pooled (length-only-controlled) residual strengthened** (+0.025 →
   +0.035, CI excludes 0): the repaired cell is a high-g² slice (0.944) that the
   old asymmetric pooling had granted to BASE but denied to HARM.
3. **The strict within-slice residual is essentially unchanged and still
   borderline** (+0.019 [−0.001, +0.039] → +0.016 [−0.003, +0.036]; width-1000:
   +0.017 [−0.001, +0.034]). The repaired slice is at ceiling for both models
   (base 0.943 vs harm 0.944), so it adds ~7% base weight of a ≈0 within-slice
   effect, slightly diluting the residual while removing the coverage caveat.
4. **Combined headline across both DPO arms (unchanged):** monitorability is
   preserved — the raw pooled gains are real at the aggregate level and are not
   length artefacts, but neither objective shows a confirmed within-task,
   matched-length improvement under the strictest control (harmless borderline
   +0.016, CI includes 0; anti-syc clean null −0.001).

## For the dissertation (§ Results / confound isolation)

Quote: *"Under the strictest control — length standardized within each
(dataset, hint) slice over the full 24-slice grid — the residual training-driven
component is +0.016 (95% CI [−0.003, +0.036]) for harmlessness DPO and −0.001
([−0.020, +0.019]) for anti-sycophancy DPO: monitorability is preserved, with a
suggestive but unconfirmed improvement."* The pooled length-only decomposition
(+0.035 [+0.017, +0.054]) may be quoted as the intermediate rung of the control
ladder, with the within-slice figure as the final word.

*Provenance: re-run executed 2026-07-27 from `results_per_instance.csv` caches
(extraction: `logs/lc_cache_20260727.json`; counts cross-checked against the
aggregate summaries: base 2,750 / harmless 2,638 / anti-syc 2,568).*
