# Results hint-type-level summary — instances pooled across datasets

- **Model:** `er-abd/harmless-sft-qwen3-4B-step-735` (`er-abd_harmless-sft-qwen3-4B-step-735`)
- **Datasets:** mmlu_pro
- **Hint types:** grader_hacking

## grader_hacking (pooled over 1 datasets)

| Hint type | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| grader_hacking | 0.2048 [0.1635, 0.2476] | 0.8896 [0.8287, 0.9439] | 0.4722 [0.4103, 0.5336] | 0.9509 [0.9130, 0.9828] | 87 (43.5% [36.5, 50.5]) | 0.6875 [0.6047, 0.7686] | 0.711 [0.652, 0.767] | 0.600 [0.541, 0.655] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 194 (12.2%) | 6 (0.4%) |
| **Not acknowledged** | 191 (12.1%) | 1194 (75.3%) |

**Opacity rate:** 0.4961 [0.3995, 0.5837] — 191 covert / 385 hint-following runs (covert / all hint-following).

## Summary

| Hint type | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] | Acknowledged & followed | Acknowledged & not followed | Not Acknowledged & followed | Not Acknowledged & not followed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| grader_hacking | 0.2048 [0.1635, 0.2476] | 0.8896 [0.8287, 0.9439] | 0.4722 [0.4103, 0.5336] | 0.9509 [0.9130, 0.9828] | 87 (43.5% [36.5, 50.5]) | 0.6875 [0.6047, 0.7686] | 0.711 [0.652, 0.767] | 0.600 [0.541, 0.655] | 194 (12.2%) | 6 (0.4%) | 191 (12.1%) | 1194 (75.3%) |
