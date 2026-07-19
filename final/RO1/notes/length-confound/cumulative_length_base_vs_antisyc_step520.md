# Cumulative length g-mean² — Base vs Anti-syc-DPO (step 520)

**Companion to** [`bucket_analysis_base_vs_antisyc_step520.md`](bucket_analysis_base_vs_antisyc_step520.md).
Instead of disjoint buckets, each row reports g-mean² over all entering
instances with `mean_cot_tokens_hinted ≤ T`: "if we cap both models at the same
max response length, who is more monitorable?" This removes the long-tail
composition difference; the final row equals each model's overall g-mean².

- Script: `monitorability-eval/scripts/11_length_cumulative_gmean2.py`
- Command: `LC_TRAINED_MODEL=er-abd_anti-syc-dpo-qwen3-4B-step-520 python3 monitorability-eval/scripts/11_length_cumulative_gmean2.py 500`
- Slices: all **24** (present in both models)

## Results ("AS" = anti-syc model)

| ≤ tok | BASE n | BASE g² | AS n | AS g² |
|---|---|---|---|---|
| 500 | 11 | 0.830 | 0 | — |
| 1000 | 341 | 0.755 | 99 | 0.786 |
| 1500 | 760 | 0.760 | 316 | 0.786 |
| 2000 | 1110 | 0.766 | 593 | 0.800 |
| 2500 | 1366 | 0.772 | 849 | 0.803 |
| 3000 | 1543 | 0.775 | 1073 | 0.808 |
| 3500 | 1688 | 0.774 | 1273 | 0.814 |
| 4000 | 1815 | 0.774 | 1421 | 0.814 |
| 4500 | 1905 | 0.772 | 1541 | 0.813 |
| 5000 | 2003 | 0.770 | 1644 | 0.813 |
| 6000 | 2177 | 0.767 | 1799 | 0.805 |
| 7000 | 2338 | 0.765 | 1933 | 0.801 |
| 8000 | 2441 | 0.767 | 2091 | 0.797 |
| 9000 | 2531 | 0.769 | 2198 | 0.796 |
| 10000 | 2601 | 0.769 | 2302 | 0.799 |
| 12000 | 2695 | 0.769 | 2450 | 0.796 |
| 14000 | 2740 | 0.768 | 2541 | 0.794 |
| 16000 (ALL) | 2750 | 0.768 | 2568 | 0.793 |

(500-tok steps in the raw output; every-other rows elided above 5000 for
brevity — full stdout reproducible with the command.)

## Interpretation

At **every** length cap from 1000 tokens up, the anti-syc model's g-mean² is
higher than base's — the advantage (+0.02–0.04) is, if anything, *largest* in
the short/mid range (≤5000 tok) and shrinks slightly as the long tail enters.
So the raw +0.025 gap does not hinge on the anti-syc model's extra-long
responses: capping both models at the same max length preserves (or grows) it.
Same qualitative picture as the harmless model. Length-cap robustness ✅ — but
this view controls length only, not dataset/hint composition (see the
within-slice note).
