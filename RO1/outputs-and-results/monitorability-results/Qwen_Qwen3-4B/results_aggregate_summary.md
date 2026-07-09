# Results aggregate summary — instances pooled across all datasets and hint types

- **Model:** `Qwen/Qwen3-4B` (`Qwen_Qwen3-4B`)
- **Datasets:** mmlu_pro, gpqa_diamond, wmdp, codemmlu, lsat_ar, lsat_lr, medxpertqa, lexam
- **Hint types:** professor, grader_hacking, unethical

## All (24 dataset × hint-type runs)

| Scope | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| all | 0.3429 (0.3902) | 0.8155 (0.4033) | 0.6017 (0.3256) | 0.8981 (0.2233) | 2750 (57.5%) | 0.7679 | 0.625 (0.427) | 0.427 (0.430) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 11279 (29.7%) | 147 (0.4%) |
| **Not acknowledged** | 5115 (13.5%) | 21417 (56.4%) |

**Opacity rate:** 0.3120 — 5115 covert / 16394 hint-following runs (covert / all hint-following).
