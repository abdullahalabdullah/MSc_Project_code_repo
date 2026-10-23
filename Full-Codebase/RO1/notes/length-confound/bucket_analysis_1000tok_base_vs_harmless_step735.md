# Length-confound bucket analysis (1000-token bins) — Base vs Harmless-DPO (step 735)

> **Variant note.** Same analysis as
> [`bucket_analysis_base_vs_harmless_step735.md`](bucket_analysis_base_vs_harmless_step735.md)
> with **1000-token bins and an open-ended 12000+ bin** instead of uniform
> 500-token buckets, so the thin tail is pooled rather than read at single-digit
> counts. One thing to know before comparing the two notes:
>
> **This note is computed on the repaired 24/24 grid** (`HARM n = 2,638`),
> whereas the original 500-token note predates the `mmlu_pro × unethical`
> monitor repair and covers 23 slices (`HARM n = 2,451`, overall g² 0.7893).
> The overall numbers here match the
> [2026-07-27 re-run](rerun_2026-07-27_base_vs_harmless_step735_24slices.md)
> (BASE 0.768, HARM 0.800) and are the ones to quote.

**Question.** The harmlessness-DPO model (`er-abd_harmless-dpo-qwen3-4B-step-735`)
generates longer CoTs than base (`Qwen_Qwen3-4B`). Longer CoTs tend to be more
monitorable. So if the two models differ in monitorability (g-mean²), is that a
**genuine training effect** or just an artifact of **response length**?

**Approach.** Take every per-instance g-mean² that *enters* the metric (Level-1
pass: `TE_i > 0 and p1_i > 0`, and `gmean2_i` defined). For each instance use the
mean token length of its 8 **hinted** responses (`mean_cot_tokens_hinted`) and
bucket by length. Within a bucket both models sit at ~the same length, so a
within-bucket difference is closer to a pure training effect. Pooled over **all
24 `(dataset, hint)` slices** (8 datasets × 3 hint types: `grader_hacking`,
`professor`, `unethical`) — all present in both models, none skipped.

- Script: `monitorability-eval/scripts/10_length_bucket_gmean2.py` (bin edges
  applied post hoc to the same `(mean_cot_tokens_hinted, gmean2_i)` rows the
  script pools, read from each cell's `results_per_instance.csv`)
- Bin width: **1000 tokens**, with all instances above 12000 pooled into one bin
- Entering instances: **BASE n = 2750**, **HARM n = 2638**

## Results (pooled, 1000-token bins)

| bin (tok) | BASE n | BASE g² | HARM n | HARM g² | HARM − BASE |
|---|---|---|---|---|---|
| 0–1000 | 341 | 0.755 | 123 | 0.843 | +0.087 |
| 1000–2000 | 769 | 0.771 | 578 | 0.803 | +0.032 |
| 2000–3000 | 433 | 0.797 | 522 | 0.826 | +0.029 |
| 3000–4000 | 272 | 0.769 | 303 | 0.838 | +0.070 |
| 4000–5000 | 188 | 0.735 | 205 | 0.790 | +0.054 |
| 5000–6000 | 174 | 0.727 | 171 | 0.749 | +0.022 |
| 6000–7000 | 161 | 0.741 | 147 | 0.740 | −0.001 |
| 7000–8000 | 103 | 0.813 | 134 | 0.827 | +0.014 |
| 8000–9000 | 90 | 0.807 | 112 | 0.778 | −0.029 |
| 9000–10000 | 70 | 0.782 | 91 | 0.798 | +0.016 |
| 10000–11000 | 53 | 0.801 | 90 | 0.778 | −0.022 |
| 11000–12000 | 41 | 0.724 | 66 | 0.711 | −0.013 |
| 12000+ | 55 | 0.721 | 96 | 0.747 | +0.025 |

**Overall:** BASE n=2750, median len=2524, g²=**0.7679** · HARM n=2638, median
len=3317, g²=**0.8002**

## Interpretation

1. **The length shift is real and substantial.** Median hinted length rises from
   **2524 → 3317 tokens** (+~31%), and the whole HARM distribution is shifted
   right: base holds 341 instances below 1000 tokens against HARM's 123, while
   HARM has more mass in almost every bin above 2000. This confirms the original
   observation: harmlessness-DPO training makes the model write longer.

2. **The raw monitorability gap is small.** Overall g-mean² is **0.768 (base) vs
   0.800 (harmless)** — a gap of **+0.032**. Even before controlling for length,
   the harmless model is not dramatically more monitorable.

3. **The gap is not manufactured by length composition.** If the raw difference
   were purely a length artifact — HARM looking better *only* because it has more
   long, more-monitorable responses — HARM would sit level with base *within*
   every bin. Instead HARM is higher in 9 of 13 bins, and the advantage is
   concentrated below 6000 tokens where most of both models' mass sits. Note also
   that base's own g² is not monotone in length (0.727 at 5000–6000, its lowest;
   0.813 at 7000–8000), so "longer is more monitorable" is a weaker regularity
   than the confound story assumes.

4. **The within-bin advantage is not uniform, and the tail is mixed.** The gap is
   largest in the sparsest low bin (+0.087 at 0–1000, n=123 for HARM) and in
   3000–5000 (+0.070, +0.054), but the sign alternates above 6000 tokens
   (−0.029, +0.016, −0.022, −0.013, +0.025). A single "HARM is better at matched
   length" summary would overstate a pattern that is concentrated in the middle
   of the distribution and noisy in the tail.

5. **Read bin width as a resolution knob, not a result.** At 500 tokens the sign
   alternates more; pooled into six wider bins it is positive throughout. Same
   data, different smoothing. None of these settles the question, which is why
   the standardized analyses exist.

### What this confirms

- **Harmlessness-DPO does inflate CoT length** (median +~31%). ✅
- **The length inflation alone does not account for the monitorability gap.**
  Holding length approximately fixed leaves a positive difference in most bins,
  concentrated below 6000 tokens. ✅
- **It does not establish that the residual is training-driven.** Pooled bins
  still mix `(dataset, hint)` slices with very different baseline g², so
  composition can masquerade as a within-length effect here. The within-slice
  standardization
  ([re-run](rerun_2026-07-27_base_vs_harmless_step735_24slices.md)) controls
  length *and* slice, and finds the training-driven component at
  **+0.016 [−0.003, +0.036]** — borderline, including zero. **That is the
  quotable number, not this table.** ⚠️

## Caveats

- **Pooling mixes heterogeneous slices.** Datasets/hint-types have different
  baseline g² and different length distributions, so a single pooled bin can
  blend composition across slices. The BASE-vs-HARM comparison is still fair
  (both pooled identically over the same 24 slices), but a pooled bin mean is
  not a clean "monitorability at length L" for any one task.
- **No CIs in this table.** Bin means are point estimates. For a quotable
  headline number with an interval, use the length-standardization +
  bootstrap-CI approach (scripts 12/13).
- **Upper bins are thin** (41–103 base instances per bin above 7000 tokens) and
  the sign there is unstable; read them loosely. The 12000+ bin is open-ended
  and spans a far wider range of lengths than any other.
