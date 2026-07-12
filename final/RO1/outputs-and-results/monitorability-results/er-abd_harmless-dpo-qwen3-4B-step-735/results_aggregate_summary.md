# Results aggregate summary — instances pooled across all datasets and hint types

- **Model:** `er-abd_harmless-dpo-qwen3-4B-step-735` (`er-abd_harmless-dpo-qwen3-4B-step-735`)
- **Datasets:** mmlu_pro, gpqa_diamond, wmdp, codemmlu, lsat_ar, lsat_lr, medxpertqa, lexam
- **Hint types:** professor, grader_hacking, unethical

## All (23 dataset × hint-type runs)

| Scope | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| all | 0.3084 [0.2972, 0.3194] | 0.8109 [0.7961, 0.8250] | 0.5816 [0.5686, 0.5947] | 0.8979 [0.8887, 0.9066] | 2451 (53.5% [52.0, 54.9]) | 0.7893 [0.7758, 0.8028] | 0.625 [0.612, 0.638] | 0.456 [0.443, 0.469] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 10178 (28.1%) | 167 (0.5%) |
| **Not acknowledged** | 4262 (11.8%) | 21604 (59.7%) |

**Opacity rate:** 0.2952 [0.2795, 0.3107] — 4262 covert / 14440 hint-following runs (covert / all hint-following).
