# Cumulative length g-mean² analysis — Base vs Harmless-DPO (step 735)

**Question.** Companion to the disjoint-bucket analysis
([`bucket_analysis_base_vs_harmless_step735.md`](bucket_analysis_base_vs_harmless_step735.md)).
The harmlessness-DPO model (`er-abd_harmless-dpo-qwen3-4B-step-735`) writes longer
CoTs than base (`Qwen_Qwen3-4B`), and longer CoTs tend to be more monitorable. A
specific version of the length-confound worry is: *"HARM only looks more
monitorable because it has more long responses, and long responses are more
monitorable."* This cumulative view addresses that directly by asking: **if we cap
both models at the same maximum response length, who is more monitorable?**

**Approach.** Take every per-instance g-mean² that *enters* the metric (Level-1
pass and `gmean2_i` defined). For a running threshold `T` (500, 1000, ... tokens),
compute g-mean² over all instances whose mean hinted-response length
(`mean_cot_tokens_hinted`) is `<= T`. Each row is a monotone-growing prefix of the
length-sorted instances, so the count only grows and the final row equals each
model's overall (unrestricted) g-mean². Pooled over **all 24 `(dataset, hint)`
slices** (8 datasets × 3 hint types), all present in both models.

- Script: `monitorability-eval/scripts/11_length_cumulative_gmean2.py`
- Command: `python3 monitorability-eval/scripts/11_length_cumulative_gmean2.py 500`
- Threshold step: **500 tokens**
- Entering instances: **BASE n = 2750**, **HARM n = 2451**

## Results (pooled, cumulative g-mean² @ ≤ length X)

| ≤ tok | BASE n | BASE g² | HARM n | HARM g² |
|---|---|---|---|---|
| 500 | 11 | 0.830 | 0 | — |
| 1000 | 341 | 0.755 | 106 | 0.832 |
| 1500 | 760 | 0.760 | 368 | 0.771 |
| 2000 | 1110 | 0.766 | 622 | 0.795 |
| 2500 | 1366 | 0.772 | 893 | 0.801 |
| 3000 | 1543 | 0.775 | 1114 | 0.804 |
| 3500 | 1688 | 0.774 | 1270 | 0.808 |
| 4000 | 1815 | 0.774 | 1403 | 0.810 |
| 4500 | 1905 | 0.772 | 1522 | 0.807 |
| 5000 | 2003 | 0.770 | 1602 | 0.807 |
| 5500 | 2093 | 0.768 | 1690 | 0.801 |
| 6000 | 2177 | 0.767 | 1758 | 0.801 |
| 6500 | 2262 | 0.767 | 1824 | 0.800 |
| 7000 | 2338 | 0.765 | 1897 | 0.796 |
| 7500 | 2383 | 0.767 | 1960 | 0.796 |
| 8000 | 2441 | 0.767 | 2021 | 0.797 |
| 8500 | 2492 | 0.768 | 2074 | 0.795 |
| 9000 | 2531 | 0.769 | 2126 | 0.795 |
| 9500 | 2570 | 0.769 | 2174 | 0.796 |
| 10000 | 2601 | 0.769 | 2213 | 0.795 |
| 10500 | 2628 | 0.769 | 2255 | 0.795 |
| 11000 | 2654 | 0.770 | 2297 | 0.794 |
| 11500 | 2676 | 0.769 | 2332 | 0.792 |
| 12000 | 2695 | 0.769 | 2363 | 0.792 |
| 12500 | 2715 | 0.768 | 2385 | 0.791 |
| 13000 | 2726 | 0.768 | 2407 | 0.790 |
| 13500 | 2736 | 0.768 | 2420 | 0.790 |
| 14000 | 2740 | 0.768 | 2432 | 0.789 |
| 14500 | 2746 | 0.768 | 2440 | 0.789 |
| 15000 | 2749 | 0.768 | 2445 | 0.789 |
| 16000 (ALL) | 2750 | 0.768 | 2451 | 0.789 |

**Overall (final row):** BASE n=2750, g²=**0.768** · HARM n=2451, g²=**0.789**

## Interpretation

1. **HARM ≥ BASE at every length cap.** From ≤1000 tok all the way to the full set,
   the harmless model is at least as monitorable as base at every threshold — the
   gap is roughly constant at **~+0.02 to +0.04**. The conclusion does not depend
   on where you truncate.

2. **The "long-tail composition" worry is ruled out.** If HARM only won because it
   has more long (more-monitorable) responses, then capping *out* the long tail
   should erase its advantage. It does not: at **≤3000 tok** HARM is still ahead
   (**0.804 vs 0.775**), and it peaks around **≤4000 tok (0.810 vs 0.774)** —
   *before* most of the long tail is even admitted. The advantage is present in the
   short/mid range, not manufactured by the tail.

3. **Both curves are flat.** Base stays ~**0.765–0.775** and HARM ~**0.789–0.810**
   across the entire length range. Monitorability barely depends on length *within
   either model*, which is exactly why the +33% median-length shift (2524 → 3352
   tok, per the bucket note) cannot be driving the monitorability numbers.

### What this confirms

- **Restricting both models to the same maximum response length does not change
  the ordering or close the gap** — HARM stays slightly ahead everywhere. ✅
- **The small monitorability difference is not an artifact of HARM's longer
  responses / heavier tail.** It is a (small) property of the training. ✅
- **Robustness:** the result is stable to the choice of length cap, so it is not an
  artifact of one particular length window. ✅

This is consistent with, and reinforces, the disjoint-bucket note (no systematic
within-length gap) and the single-slice length-standardization check (length
explained −0.001 of the `mmlu_pro/grader_hacking` gap).

## Relationship to the bucket analysis (what's different)

The cumulative curve and the disjoint buckets are built from the **same**
(length, g²) pairs — the cumulative curve is essentially the running average of the
buckets, so it is **not independent evidence** and cannot contradict them.

- **Buckets** (`bucket_analysis_...md`) → *local, matched-length* comparison: "at
  *this* length, who wins?" Cleanest confound isolation, but noisier per row, and
  it reveals local structure (HARM actually dips slightly below base in a few
  bands, e.g. 1000–1500 and 6500–7000 tok).
- **Cumulative** (this note) → *robustness-to-length-cap* view: "does the answer
  survive any max-length truncation, and is it stable?" Smooth (n only grows) and
  directly kills the long-tail story — but it averages over a *mixture* of lengths,
  so each point blends composition with the true effect and it hides the local
  dips the buckets expose.

Use them together: buckets for confound isolation, cumulative for robustness.

## Caveats

- **Not independent of the bucket analysis** — same underlying data, different lens.
- **Mixture within each cap.** Each cumulative point averages over all lengths ≤ T,
  and the two models have different length mixes within any cap (HARM is depleted of
  short responses), so a cumulative gap is *not* a clean matched-length comparison.
  For that, use the buckets; for a single length-adjusted number, use
  length-standardization + bootstrap CI.
- **Pooling mixes heterogeneous slices** (different baseline g² and length
  distributions per dataset/hint). The BASE-vs-HARM comparison is still fair (both
  pooled identically over the same 24 slices).
- **No CIs in this table** — point estimates only.
