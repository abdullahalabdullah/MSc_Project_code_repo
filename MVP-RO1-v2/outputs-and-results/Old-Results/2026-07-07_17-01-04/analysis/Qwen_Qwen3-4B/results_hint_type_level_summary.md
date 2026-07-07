# Results hint-type-level summary — instances pooled across datasets

- **Model:** `Qwen/Qwen3-4B` (`Qwen_Qwen3-4B`)
- **Datasets:** mmlu_pro, medxpertqa, lexam
- **Hint types:** professor

## professor (pooled over 3 datasets)

| Hint type | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| professor | 0.2550 (0.3600) | 0.8200 (0.3270) | 0.5977 (0.3138) | 0.8841 (0.2406) | 64 (42.7%) | 0.6497 | 0.522 (0.462) | 0.407 (0.453) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 106 (17.7%) | 7 (1.2%) |
| **Not acknowledged** | 90 (15.1%) | 395 (66.1%) |

**Opacity rate:** 0.4592 — 90 covert / 196 hint-following runs (covert / all hint-following).

## Summary

| Hint type | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) | Acknowledged & followed | Acknowledged & not followed | Not Acknowledged & followed | Not Acknowledged & not followed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| professor | 0.2550 (0.3600) | 0.8200 (0.3270) | 0.5977 (0.3138) | 0.8841 (0.2406) | 64 (42.7%) | 0.6497 | 0.522 (0.462) | 0.407 (0.453) | 106 (17.7%) | 7 (1.2%) | 90 (15.1%) | 395 (66.1%) |
