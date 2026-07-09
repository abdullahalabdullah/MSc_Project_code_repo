# Results aggregate summary — instances pooled across all datasets and hint types

- **Model:** `Qwen/Qwen3-4B` (`Qwen_Qwen3-4B`)
- **Datasets:** mmlu_pro, medxpertqa, lexam
- **Hint types:** professor

## All (3 dataset × hint-type runs)

| Scope | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| all | 0.2550 (0.3600) | 0.8200 (0.3270) | 0.5977 (0.3138) | 0.8841 (0.2406) | 64 (42.7%) | 0.6497 | 0.522 (0.462) | 0.407 (0.453) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 106 (17.7%) | 3 (0.5%) |
| **Not acknowledged** | 90 (15.1%) | 399 (66.7%) |

**Opacity rate:** 0.4592 — 90 covert / 196 hint-following runs (covert / all hint-following).
