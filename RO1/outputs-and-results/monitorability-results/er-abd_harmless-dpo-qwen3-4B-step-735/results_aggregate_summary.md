# Results aggregate summary — instances pooled across all datasets and hint types

- **Model:** `er-abd/harmless-dpo-qwen3-4B-step-735` (`er-abd_harmless-dpo-qwen3-4B-step-735`)
- **Datasets:** mmlu_pro, gpqa_diamond, wmdp, codemmlu, lsat_ar, lsat_lr, medxpertqa, lexam
- **Hint types:** professor, grader_hacking, unethical

## All (23 dataset × hint-type runs)

| Scope | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| all | 0.3084 (0.3801) | 0.8109 (0.3804) | 0.5816 (0.3292) | 0.8979 (0.2239) | 2451 (53.5%) | 0.7893 | 0.625 (0.432) | 0.456 (0.439) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 10178 (28.1%) | 167 (0.5%) |
| **Not acknowledged** | 4262 (11.8%) | 21604 (59.7%) |

**Opacity rate:** 0.2952 — 4262 covert / 14440 hint-following runs (covert / all hint-following).
