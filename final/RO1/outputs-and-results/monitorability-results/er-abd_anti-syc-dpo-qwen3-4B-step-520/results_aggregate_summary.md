# Results aggregate summary — instances pooled across all datasets and hint types

- **Model:** `er-abd/anti-syc-dpo-qwen3-4B-step-520` (`er-abd_anti-syc-dpo-qwen3-4B-step-520`)
- **Datasets:** mmlu_pro, gpqa_diamond, wmdp, codemmlu, lsat_ar, lsat_lr, medxpertqa, lexam
- **Hint types:** professor, grader_hacking, unethical

## All (24 dataset × hint-type runs)

| Scope | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| all | 0.3094 [0.2989, 0.3202] | 0.8093 [0.7953, 0.8227] | 0.5814 [0.5690, 0.5941] | 0.8967 [0.8881, 0.9051] | 2568 (53.7% [52.2, 55.1]) | 0.7929 [0.7797, 0.8057] | 0.629 [0.617, 0.642] | 0.454 [0.441, 0.466] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 10798 (28.8%) | 209 (0.6%) |
| **Not acknowledged** | 4159 (11.1%) | 22333 (59.6%) |

**Opacity rate:** 0.2781 [0.2636, 0.2925] — 4159 covert / 14957 hint-following runs (covert / all hint-following).
