# Length-standardized g-mean² analysis — Base vs Harmless-DPO (step 735)

**The rigorous version.** This is the method that most cleanly isolates the
response-length confound and gives a single length-adjusted number with a
confidence interval. Companion to the two descriptive views:
[bucket](bucket_analysis_base_vs_harmless_step735.md) (local, matched-length) and
[cumulative](cumulative_length_base_vs_harmless_step735.md) (robustness-to-cap).

**Question.** The harmlessness-DPO model (`er-abd_harmless-dpo-qwen3-4B-step-735`)
writes longer CoTs than base (`Qwen_Qwen3-4B`); longer CoTs tend to be more
monitorable. Its raw g-mean² is a little higher. Is that a genuine training effect,
or just because it has more (more-monitorable) long responses?

**Method — direct standardization (post-stratification).**
1. Take every per-instance g-mean² that *enters* the metric (Level-1 pass and
   `gmean2_i` defined). Bin by mean hinted-response length
   (`mean_cot_tokens_hinted`).
2. **HARM standardized → BASE lengths** = Σ over bins of
   `[BASE's fraction of instances in the bin] × [HARM's mean g² in the bin]`.
   That is HARM's per-length monitorability, reweighted to base's length mix — so
   length is held fixed and can no longer explain any difference.
3. Decompose the raw gap:
   - `total gap      = harm_raw − base_raw`
   - `length-driven  = harm_raw − harm_std`   (attributable to HARM's longer mix)
   - `training-driven= harm_std − base_raw`   (what remains at matched length)
4. **Bootstrap CIs**: resample instances (each row already pools 8 samples into one
   `gmean2_i`) with replacement within each model, 10 000 times; base weights are
   re-estimated each resample.

Pooled over **all 24 `(dataset, hint)` slices** (8 datasets × 3 hint types), all
present in both models.

- Script: `monitorability-eval/scripts/12_length_standardized_gmean2.py`
- Command: `python3 monitorability-eval/scripts/12_length_standardized_gmean2.py 500 10000`
- Bin width: **500 tokens** (base-weight coverage 99.6%); **identical at 1000 tok**
- Entering instances: **BASE n = 2750**, **HARM n = 2451**

## Results (pooled, bin width 500, 10 000 bootstrap resamples)

| quantity | point | 95% CI |
|---|---|---|
| BASE raw g² | 0.768 | [0.755, 0.781] |
| HARM raw g² | 0.789 | [0.776, 0.803] |
| **HARM standardized → BASE lengths** | **0.793** | **[0.779, 0.807]** |
| BASE standardized → HARM lengths (robustness) | 0.770 | [0.755, 0.784] |

**Decomposition of the raw gap (HARM raw − BASE raw = +0.021):**

| component | point | 95% CI | verdict |
|---|---|---|---|
| total gap | **+0.021** | [+0.003, +0.040] | excludes 0 |
| length-driven (harm_raw − harm_std) | **−0.004** | [−0.010, +0.003] | **includes 0** |
| training-driven (harm_std − base_raw) | **+0.025** | [+0.006, +0.044] | excludes 0 |

Robustness: at bin width **1000 tok** the numbers are unchanged (harm_std = 0.793,
length-driven −0.004, training-driven +0.025), and the reverse standardization
(base → harm lengths) barely moves base (0.768 → 0.770). Coverage is ~100%, so the
adjustment is not resting on unmatched bins.

## Interpretation

1. **There IS a small but real monitorability difference.** Total gap +0.021 with
   CI [+0.003, +0.040] excluding zero. Pooling to ~2.5k instances resolves what was
   borderline on a single slice: the harmless model is genuinely a little more
   monitorable.

2. **Length explains essentially none of it.** The length-driven component is
   **−0.004 with a tight CI [−0.010, +0.003] that straddles zero.** Reweighting the
   harmless model onto base's (shorter) length distribution *does not* pull its
   score down — if anything it nudges up. So the fact that HARM writes longer
   responses is **not** why it looks more monitorable.

3. **The whole gap is training-driven.** After matching length, the difference is
   **+0.025, CI [+0.006, +0.044]**, still excluding zero — statistically as strong
   as (slightly larger than) the raw gap. Holding length fixed does not shrink the
   effect; it isolates it.

4. **Robust.** Same result at 500 and 1000-tok bins, ~100% coverage, and the
   symmetric check (base → harm lengths moves base by only +0.002) all agree. The
   conclusion is not an artifact of binning choice.

### What this confirms

- **The harmlessness-DPO model is slightly more monitorable than base — and this is
  a property of the training, not of its longer responses.** ✅
- **The length confound is formally ruled out:** the length-attributable part of the
  gap is ≈0 (CI [−0.010, +0.003]); the entire +0.021 raw gap survives as a
  training-driven +0.025 at matched length. ✅
- This is the quantitative, CI-backed version of what the bucket table (no
  systematic within-length gap) and cumulative curve (HARM ahead at every length
  cap) showed qualitatively — all three methods agree.

## Caveats

- **Standardizes on length, pooled across slices.** It adjusts for response length,
  not for dataset/hint composition; both models are pooled identically over the same
  24 slices, so the comparison is fair, but a pooled bin is not "monitorability at
  length L" for any single task. A stricter design would standardize within each
  `(dataset, hint)` slice and aggregate — expected to agree, given per-slice results.
- **g-mean² is granular per instance** (8 samples → values like 0, 0.5, 0.667, 1),
  which is why we pool and bootstrap rather than trust any one bin.
- **Effect size is small** (~0.02–0.03 on a 0–1 scale). It is statistically distinct
  from zero at this n but modest in practical terms; report the CI, not just the
  point.

## Appendix — how the standardization is computed (step by step)

### The intuition (a schools analogy)

Two schools take the same test. School B's students score higher on average. But
School B also has far more **seniors**, and seniors naturally score higher. Is
School B actually better, or does it just *have more seniors*?

The fair comparison is **not** the raw averages. Instead: *"If School B had the
same grade mix as School A, would it still score higher?"* You take School B's
score **within each grade** and recombine those using **School A's grade
proportions**. Now both schools are judged on the same student mix, so grade level
cannot explain any remaining difference.

Map that onto this analysis:

| analogy | this study |
|---|---|
| school | model (BASE vs HARM) |
| grade level | response-length bin (500-tok wide) |
| test score | monitorability (g-mean² of an instance) |
| "more seniors" | HARM writes more long responses |

HARM *looks* better partly the way School B does — it has more of the
"high-scoring type" (long responses). Standardization removes that by scoring HARM
on **base's** length mix.

### The algorithm (exactly what the script does)

1. **Collect entering instances.** For each of the 24 slices, run `compute_gmean2`
   and keep every instance that passes Level-1 (`TE_i>0 and p1_i>0`) and has a
   defined `gmean2_i`. Each instance contributes one pair: `(mean hinted-response
   length, gmean2_i)`. Pool across slices → BASE 2750 pairs, HARM 2451 pairs.

2. **Bin by length.** Assign each instance to a 500-token bin: `bin = floor(L/500)*500`.

3. **Per-bin means.** For each bin, compute HARM's mean g² among HARM instances in
   that bin (`harm_mean[bin]`), and count BASE instances in that bin
   (`base_count[bin]`).

4. **Base length weights.** `base_weight[bin] = base_count[bin] / 2750` — the
   fraction of BASE instances at that length.

5. **The standardized number — the key swap.** HARM's raw average is
   `Σ harm_weight[bin] × harm_mean[bin]` (HARM's own length weights). To standardize,
   swap in **base's** weights but keep **HARM's** per-bin scores:

   ```
   harm_std = Σ  base_weight[bin] × harm_mean[bin]
   ```

   (Restricted to bins where HARM also has instances; base_weight is renormalised
   over those bins. Here coverage = 99.6%, so almost nothing is dropped.)

6. **Decompose the gap.**
   - `length-driven  = harm_raw − harm_std` → moving HARM from its own to base's
     length mix; whatever this changes is attributable to the length difference.
   - `training-driven = harm_std − base_raw` → both now on base's length mix, so
     what remains is not length.

7. **Bootstrap.** Resample the 2750 BASE and 2451 HARM instances with replacement
   (independently, within model), recompute steps 3–6, repeat 10 000×, take the
   2.5th/97.5th percentiles. Because base counts are recomputed each resample, the
   uncertainty in base's length weights is included in the CI.

### Worked example (real pooled numbers, first two bins)

Using the pooled 500-tok bins (BASE total n = 2750):

| bin (tok) | BASE n | base_weight | HARM mean g² | contribution to `harm_std` = base_weight × HARM g² |
|---|---|---|---|---|
| 500–1000 | 330 | 330/2750 = 0.120 | 0.840 | 0.120 × 0.840 = 0.101 |
| 1000–1500 | 419 | 419/2750 = 0.152 | 0.744 | 0.152 × 0.744 = 0.113 |
| … | … | … | … | … |

Summing that last column over **all** bins gives `harm_std = 0.793`. Compare to
HARM's own raw average, which uses HARM's weights instead
(`Σ harm_weight × harm_mean = 0.789`). The two are almost identical — that is the
finding: swapping HARM onto base's length distribution barely changes its score
(0.789 → 0.793), so length is not what is producing HARM's monitorability.

Contrast with a *toy* case where standardization would matter: if short responses
scored 0.5 and long scored 0.9 for **both** models, then BASE (mostly short) would
average ~0.58 and HARM (mostly long) ~0.82 — a big raw gap. Standardizing HARM to
base's mostly-short mix would pull it back to ~0.58, revealing the gap as pure
length. That collapse does **not** happen in our data (−0.004), which is exactly
why we conclude the effect is training-driven.
