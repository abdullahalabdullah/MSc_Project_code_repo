# Length-confound bucket analysis (1000-token bins) — Base vs Anti-syc-DPO (step 520)

> **Variant note.** Same analysis as
> [`bucket_analysis_base_vs_antisyc_step520.md`](bucket_analysis_base_vs_antisyc_step520.md)
> with **1000-token bins and an open-ended 12000+ bin** instead of uniform
> 500-token buckets, so the thin tail is pooled rather than read at single-digit
> counts. The underlying data is unchanged (this model was never affected by the
> `mmlu_pro × unethical` monitor gap), and the overall figures reproduce the
> original note exactly: BASE n = 2750 / g² 0.7679, ANTI-SYC n = 2568 / g² 0.7929.

**Question.** The anti-sycophancy-DPO model (`er-abd_anti-syc-dpo-qwen3-4B-step-520`)
generates longer CoTs than base (`Qwen_Qwen3-4B`). Longer CoTs tend to be more
monitorable. So is the raw g-mean² gap a **genuine training effect** or an
artifact of **response length**? (Same method as
[`bucket_analysis_1000tok_base_vs_harmless_step735.md`](bucket_analysis_1000tok_base_vs_harmless_step735.md).)

**Approach.** Pool every per-instance g-mean² that *enters* the metric (Level-1
pass and `gmean2_i` defined) across all **24 `(dataset, hint)` slices** (8
datasets × 3 hints — all present in both models, none skipped; the
`mmlu_pro/unethical` monitor gap from the harmless run has since been repaired).
Bucket by `mean_cot_tokens_hinted` (mean token length of the 8 hinted responses).

- Script: `monitorability-eval/scripts/10_length_bucket_gmean2.py` with
  `LC_TRAINED_MODEL=er-abd_anti-syc-dpo-qwen3-4B-step-520` (bin edges applied
  post hoc to the same `(mean_cot_tokens_hinted, gmean2_i)` rows the script
  pools, read from each cell's `results_per_instance.csv`)
- Bin width: **1000 tokens**, with all instances above 12000 pooled into one bin
- Entering instances: **BASE n = 2750**, **ANTI-SYC n = 2568**

## Results (pooled, 1000-token bins)

| bin (tok) | BASE n | BASE g² | AS n | AS g² | AS − BASE |
|---|---|---|---|---|---|
| 0–1000 | 341 | 0.755 | 99 | 0.786 | +0.031 |
| 1000–2000 | 769 | 0.771 | 494 | 0.803 | +0.032 |
| 2000–3000 | 433 | 0.797 | 480 | 0.818 | +0.021 |
| 3000–4000 | 272 | 0.769 | 348 | 0.832 | +0.063 |
| 4000–5000 | 188 | 0.735 | 223 | 0.805 | +0.070 |
| 5000–6000 | 174 | 0.727 | 155 | 0.723 | −0.004 |
| 6000–7000 | 161 | 0.741 | 134 | 0.752 | +0.011 |
| 7000–8000 | 103 | 0.813 | 158 | 0.737 | −0.076 |
| 8000–9000 | 90 | 0.807 | 107 | 0.792 | −0.015 |
| 9000–10000 | 70 | 0.782 | 104 | 0.862 | +0.080 |
| 10000–11000 | 53 | 0.801 | 83 | 0.744 | −0.057 |
| 11000–12000 | 41 | 0.724 | 65 | 0.753 | +0.029 |
| 12000+ | 55 | 0.721 | 118 | 0.724 | +0.003 |

**Overall:** BASE n=2750, median len=2524, g²=**0.7679** · ANTI-SYC n=2568,
median len=3543, g²=**0.7929**

## Interpretation

1. **The length shift is real and larger than the harmless model's.** Median
   hinted length rises **2524 → 3543 tokens** (+~40%; harmless was +31% to 3317),
   with the whole distribution shifted right (base holds 341 instances below
   1000 tokens against anti-syc's 99).

2. **The raw monitorability gap is small:** 0.768 vs 0.793 (**+0.025**),
   slightly smaller than the harmless model's +0.032.

3. **Within matched-length bins the gap does not vanish, but it is
   band-dependent.** Anti-syc is higher in 9 of 13 bins, with the advantage
   concentrated in 3000–5000 tokens (+0.063 and +0.070) where a large share of
   its mass sits, and a flat +0.02 to +0.03 below 3000. So length alone does not
   explain the raw gap at the bin level.

4. **The tail is noisy and the sign alternates there.** Above 5000 tokens the
   differences swing between −0.076 (7000–8000) and +0.080 (9000–10000) on bins
   holding 41–103 base instances each. Those swings are consistent with sampling
   noise rather than a length-dependent effect, and no reading of the upper bins
   should be load-bearing.

5. **None of this shows the residual is training-driven.** Pooled bins hold
   length approximately fixed but leave `(dataset, hint)` composition
   uncontrolled, and those slices differ enormously in baseline g². The
   within-slice standardization
   ([`withinslice_standardized_base_vs_antisyc_step520.md`](withinslice_standardized_base_vs_antisyc_step520.md))
   controls length *and* slice, and the residual disappears:
   training-driven **−0.001 [−0.020, +0.019]**, with essentially the entire raw
   gap attributed to confounds (+0.026 [+0.012, +0.040]). **That is the quotable
   number, not this table.** ⚠️

## Caveats

- **Pooling mixes heterogeneous slices.** A single pooled bin blends composition
  across datasets and hint types, so it is not a clean "monitorability at
  length L" for any one task. The comparison is still fair between models (both
  pooled identically over the same 24 slices).
- **No CIs in this table.** Bin means are point estimates; use scripts 12/13 for
  an interval.
- **Upper bins are thin** (41–103 base instances per bin above 7000 tokens) and
  unstable in sign. The 12000+ bin is open-ended and spans a far wider range of
  lengths than any other.
- **Bin width changes the apparent pattern.** At 500 tokens the sign alternates
  more; pooled into six wider bins it is positive in five of six. Same data at
  different resolution, and neither settles the question.
