# Results aggregate summary — instances pooled across all datasets and hint types

- **Model:** `er-abd_harmless-dpo-qwen3-4B-step-735` (`er-abd_harmless-dpo-qwen3-4B-step-735`)
- **Datasets:** mmlu_pro, gpqa_diamond, wmdp, codemmlu, lsat_ar, lsat_lr, medxpertqa, lexam
- **Hint types:** professor, grader_hacking, unethical

## All (24 dataset × hint-type runs)

| Scope | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| all | 0.3250 [0.3142, 0.3359] | 0.8207 [0.8069, 0.8341] | 0.5939 [0.5812, 0.6063] | 0.9036 [0.8953, 0.9117] | 2638 (55.2% [53.8, 56.6]) | 0.8002 [0.7874, 0.8129] | 0.629 [0.617, 0.642] | 0.446 [0.434, 0.459] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 11288 (29.9%) | 179 (0.5%) |
| **Not acknowledged** | 4337 (11.5%) | 21992 (58.2%) |

**Opacity rate:** 0.2776 [0.2630, 0.2919] — 4337 covert / 15625 hint-following runs (covert / all hint-following).
