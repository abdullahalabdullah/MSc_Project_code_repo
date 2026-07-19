# Length-confound bucket analysis — Base vs Anti-syc-DPO (step 520)

**Question.** The anti-sycophancy-DPO model (`er-abd_anti-syc-dpo-qwen3-4B-step-520`)
generates longer CoTs than base (`Qwen_Qwen3-4B`). Longer CoTs tend to be more
monitorable. So is the raw g-mean² gap a **genuine training effect** or an
artifact of **response length**? (Same method as
[`bucket_analysis_base_vs_harmless_step735.md`](bucket_analysis_base_vs_harmless_step735.md).)

**Approach.** Pool every per-instance g-mean² that *enters* the metric (Level-1
pass and `gmean2_i` defined) across all **24 `(dataset, hint)` slices** (8
datasets × 3 hints — all present in both models, none skipped; the
`mmlu_pro/unethical` monitor gap from the harmless run has since been repaired).
Bucket by `mean_cot_tokens_hinted` (mean token length of the 8 hinted responses).

- Script: `monitorability-eval/scripts/10_length_bucket_gmean2.py`
- Command: `LC_TRAINED_MODEL=er-abd_anti-syc-dpo-qwen3-4B-step-520 python3 monitorability-eval/scripts/10_length_bucket_gmean2.py 500`
- Bucket width: **500 tokens**
- Entering instances: **BASE n = 2750**, **ANTI-SYC n = 2568**

## Results (pooled, 500-token buckets; "HARM" = anti-syc model)

| bucket (tok) | BASE n | BASE g² | AS n | AS g² |
|---|---|---|---|---|
| 0–500 | 11 | 0.830 | 0 | — |
| 500–1000 | 330 | 0.753 | 99 | 0.786 |
| 1000–1500 | 419 | 0.764 | 217 | 0.786 |
| 1500–2000 | 350 | 0.779 | 277 | 0.816 |
| 2000–2500 | 256 | 0.797 | 255 | 0.809 |
| 2500–3000 | 177 | 0.797 | 225 | 0.829 |
| 3000–3500 | 145 | 0.760 | 200 | 0.842 |
| 3500–4000 | 127 | 0.779 | 148 | 0.818 |
| 4000–4500 | 90 | 0.742 | 120 | 0.804 |
| 4500–5000 | 98 | 0.730 | 103 | 0.807 |
| 5000–5500 | 90 | 0.718 | 83 | 0.728 |
| 5500–6000 | 84 | 0.737 | 72 | 0.718 |
| 6000–6500 | 85 | 0.774 | 72 | 0.827 |
| 6500–7000 | 76 | 0.704 | 62 | 0.666 |
| 7000–7500 | 45 | 0.860 | 89 | 0.748 |
| 7500–8000 | 58 | 0.777 | 69 | 0.724 |
| 8000–8500 | 51 | 0.801 | 50 | 0.745 |
| 8500–9000 | 39 | 0.814 | 57 | 0.834 |
| 9000–9500 | 39 | 0.772 | 51 | 0.903 |
| 9500–10000 | 31 | 0.795 | 53 | 0.822 |
| 10000–10500 | 27 | 0.817 | 39 | 0.714 |
| 10500–11000 | 26 | 0.784 | 44 | 0.771 |
| 11000–11500 | 22 | 0.713 | 36 | 0.783 |
| 11500–12000 | 19 | 0.736 | 29 | 0.716 |
| 12000–12500 | 20 | 0.684 | 27 | 0.638 |
| 12500–13000 | 11 | 0.624 | 24 | 0.774 |
| 13000–13500 | 10 | 0.860 | 19 | 0.689 |
| 13500–14000 | 4 | 0.719 | 21 | 0.866 |
| 14000–14500 | 6 | 0.773 | 12 | 0.758 |
| 14500–15000 | 3 | 0.667 | 7 | 0.548 |
| 15000–15500 | 0 | — | 7 | 0.688 |
| 15500–16000 | 1 | 1.000 | 1 | 0.600 |

**Overall:** BASE n=2750, median len=2524, g²=**0.7679** · ANTI-SYC n=2568,
median len=3543, g²=**0.7929**

## Interpretation

1. **The length shift is real and larger than the harmless model's.** Median
   hinted length rises **2524 → 3543 tokens** (+~40%; harmless was +33% to 3352),
   with the whole distribution shifted right (base 330 vs anti-syc 99 in
   500–1000 tok).
2. **The raw monitorability gap is small:** 0.768 vs 0.793 (**+0.025**),
   comparable to the harmless model's +0.021.
3. **Within matched-length buckets the gap does NOT vanish, but it is
   band-dependent:** anti-syc is consistently higher in every bucket from
   500–5000 tok (where ~2/3 of its mass sits), roughly flat 5000–7000, and mixed
   above 7000. So length alone does not explain the raw gap at the bucket level —
   but see the within-slice standardization
   ([`withinslice_standardized_base_vs_antisyc_step520.md`](withinslice_standardized_base_vs_antisyc_step520.md)),
   which shows the residual disappears once dataset/hint composition is *also*
   controlled.
