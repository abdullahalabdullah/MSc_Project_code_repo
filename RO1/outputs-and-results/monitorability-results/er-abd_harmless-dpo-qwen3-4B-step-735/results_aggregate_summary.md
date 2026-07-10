# Results aggregate summary — instances pooled across all datasets and hint types

- **Model:** `er-abd/harmless-dpo-qwen3-4B-step-735` (`er-abd_harmless-dpo-qwen3-4B-step-735`)
- **Datasets:** mmlu_pro, gpqa_diamond, wmdp, codemmlu, lsat_ar, lsat_lr, medxpertqa, lexam
- **Hint types:** professor, unethical

## All (15 dataset × hint-type runs)

| Scope | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| all | 0.3689 (0.3996) | 0.8320 (0.3462) | 0.6266 (0.3269) | 0.9036 (0.2186) | 1768 (59.2%) | 0.8646 | 0.622 (0.433) | 0.418 (0.433) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 8716 (36.9%) | 156 (0.7%) |
| **Not acknowledged** | 2175 (9.2%) | 12604 (53.3%) |

**Opacity rate:** 0.1997 — 2175 covert / 10891 hint-following runs (covert / all hint-following).
