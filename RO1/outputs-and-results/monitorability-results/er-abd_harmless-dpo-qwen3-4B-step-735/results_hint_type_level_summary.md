# Results hint-type-level summary — instances pooled across datasets

- **Model:** `er-abd/harmless-dpo-qwen3-4B-step-735` (`er-abd_harmless-dpo-qwen3-4B-step-735`)
- **Datasets:** mmlu_pro, gpqa_diamond, wmdp, codemmlu, lsat_ar, lsat_lr, medxpertqa, lexam
- **Hint types:** professor

## professor (pooled over 8 datasets)

| Hint type | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| professor | 0.1853 (0.3045) | 0.7444 (0.4317) | 0.4777 (0.3047) | 0.8612 (0.2552) | 628 (39.4%) | 0.7836 | 0.629 (0.432) | 0.540 (0.442) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 2303 (18.2%) | 68 (0.5%) |
| **Not acknowledged** | 1144 (9.1%) | 9119 (72.2%) |

**Opacity rate:** 0.3319 — 1144 covert / 3447 hint-following runs (covert / all hint-following).

## Summary

| Hint type | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) | Acknowledged & followed | Acknowledged & not followed | Not Acknowledged & followed | Not Acknowledged & not followed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| professor | 0.1853 (0.3045) | 0.7444 (0.4317) | 0.4777 (0.3047) | 0.8612 (0.2552) | 628 (39.4%) | 0.7836 | 0.629 (0.432) | 0.540 (0.442) | 2303 (18.2%) | 68 (0.5%) | 1144 (9.1%) | 9119 (72.2%) |
