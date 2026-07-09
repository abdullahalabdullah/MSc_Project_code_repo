# Results aggregate summary — instances pooled across all datasets and hint types

- **Model:** `er-abd_harmless-dpo-qwen3-4B-pilot` (`er-abd_harmless-dpo-qwen3-4B-pilot`)
- **Datasets:** mmlu_pro, gpqa_diamond, wmdp, codemmlu, lsat_ar, lsat_lr, medxpertqa, lexam
- **Hint types:** professor

## All (8 dataset × hint-type runs)

| Scope | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| all | 0.2035 (0.3384) | 0.7662 (0.4032) | 0.5977 (0.2902) | 0.9215 (0.1860) | 139 (34.9%) | 0.7276 | 0.632 (0.442) | 0.531 (0.448) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 267 (16.8%) | 7 (0.4%) |
| **Not acknowledged** | 183 (11.5%) | 1130 (71.2%) |

**Opacity rate:** 0.4067 — 183 covert / 450 hint-following runs (covert / all hint-following).
