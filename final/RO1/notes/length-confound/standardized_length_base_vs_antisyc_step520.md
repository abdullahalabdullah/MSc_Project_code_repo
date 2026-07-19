# Length-standardized g-mean² — Base vs Anti-syc-DPO (step 520)

**Pooled direct standardization (post-stratification) with bootstrap CIs** —
the rigorous single-number version of the length-confound check. Computes what
the anti-syc model *would* score if its responses had base's length
distribution, then decomposes the raw gap into a length-driven and a
training-driven part. Same method as
[`standardized_length_base_vs_harmless_step735.md`](standardized_length_base_vs_harmless_step735.md).

- Script: `monitorability-eval/scripts/12_length_standardized_gmean2.py`
- Command: `LC_TRAINED_MODEL=er-abd_anti-syc-dpo-qwen3-4B-step-520 python3 monitorability-eval/scripts/12_length_standardized_gmean2.py 500 10000`
- Slices: all **24**; entering instances BASE n=2750, ANTI-SYC n=2568
- Bin coverage (anti-syc→base): 99.6% (100.0% at width 1000)

## Results (width 500, 10 000 bootstrap; "AS" = anti-syc model)

| quantity | point | 95% CI |
|---|---|---|
| BASE raw g² | 0.768 | [0.755, 0.781] |
| AS raw g² | 0.793 | [0.780, 0.806] |
| **AS standardized → BASE lengths** | **0.791** | **[0.776, 0.806]** |
| BASE standardized → AS lengths | 0.771 | [0.756, 0.785] |

**Decomposition of the raw gap (AS − BASE = +0.025):**

| component | point | 95% CI | verdict |
|---|---|---|---|
| total gap | **+0.025** | [+0.006, +0.044] | excludes 0 |
| length-driven (as_raw − as_std) | **+0.002** | [−0.006, +0.009] | includes 0 |
| training-driven (as_std − base_raw) | **+0.024** | [+0.003, +0.044] | excludes 0 |

Robust to bin width: at 1000 tok the split is −0.000 length / +0.025
"training", same picture.

## Interpretation

Identical qualitative result to the harmless model: **response length explains
essentially none of the raw gap** (+0.002, tight CI around 0) despite the
anti-syc model's +40% median-length inflation, and the length-adjusted residual
(+0.024, CI excludes 0) looks like a training effect *at this level of
control*. **But** — as with the harmless model — the stricter within-slice
standardization
([`withinslice_standardized_base_vs_antisyc_step520.md`](withinslice_standardized_base_vs_antisyc_step520.md))
shows this residual is actually dataset/hint composition, not training: pooled
length bins mix slices with very different baseline g². Quote the within-slice
numbers, not these, as the headline.
