# Results aggregate summary — instances pooled across all datasets and hint types

- **Model:** `er-abd_harmless-dpo-qwen3-4B-pilot` (`er-abd_harmless-dpo-qwen3-4B-pilot`)
- **Datasets:** mmlu_pro, gpqa_diamond, wmdp, codemmlu
- **Hint types:** professor

## All (4 dataset × hint-type runs)

| Scope | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| all | 0.2315 (0.3585) | 0.7713 (0.4095) | 0.6126 (0.2936) | 0.9188 (0.1935) | 77 (38.9%) | 0.7864 | 0.665 (0.431) | 0.548 (0.442) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 165 (20.9%) | 2 (0.3%) |
| **Not acknowledged** | 85 (10.8%) | 536 (68.0%) |

**Opacity rate:** 0.3400 — 85 covert / 250 hint-following runs (covert / all hint-following).
