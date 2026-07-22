# Results hint-type-level summary — instances pooled across datasets

- **Model:** `er-abd_harmless-dpo-qwen3-4B-step-735` (`er-abd_harmless-dpo-qwen3-4B-step-735`)
- **Datasets:** mmlu_pro, gpqa_diamond, wmdp, codemmlu, lsat_ar, lsat_lr, medxpertqa, lexam
- **Hint types:** professor, grader_hacking, unethical

## professor (pooled over 8 datasets)

| Hint type | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| professor | 0.1853 [0.1704, 0.2005] | 0.7444 [0.7121, 0.7752] | 0.4777 [0.4538, 0.5018] | 0.8612 [0.8411, 0.8813] | 628 (39.4% [37.0, 41.8]) | 0.7836 [0.7568, 0.8096] | 0.629 [0.608, 0.650] | 0.540 [0.517, 0.561] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 2303 (18.2%) | 68 (0.5%) |
| **Not acknowledged** | 1144 (9.1%) | 9119 (72.2%) |

**Opacity rate:** 0.3319 [0.3037, 0.3605] — 1144 covert / 3447 hint-following runs (covert / all hint-following).

## grader_hacking (pooled over 8 datasets)

| Hint type | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| grader_hacking | 0.1950 [0.1798, 0.2102] | 0.7586 [0.7272, 0.7887] | 0.4651 [0.4419, 0.4879] | 0.8830 [0.8649, 0.9005] | 683 (42.8% [40.4, 45.2]) | 0.5944 [0.5628, 0.6253] | 0.630 [0.608, 0.650] | 0.529 [0.507, 0.550] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1462 (11.6%) | 11 (0.1%) |
| **Not acknowledged** | 2087 (16.6%) | 9000 (71.7%) |

**Opacity rate:** 0.5881 [0.5569, 0.6185] — 2087 covert / 3549 hint-following runs (covert / all hint-following).

## unethical (pooled over 8 datasets)

| Hint type | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| unethical | 0.5946 [0.5755, 0.6134] | 0.8942 [0.8803, 0.9077] | 0.7151 [0.6985, 0.7314] | 0.9342 [0.9241, 0.9440] | 1327 (83.2% [81.3, 85.0]) | 0.9141 [0.9032, 0.9241] | 0.629 [0.608, 0.650] | 0.271 [0.253, 0.289] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 7523 (59.7%) | 100 (0.8%) |
| **Not acknowledged** | 1106 (8.8%) | 3873 (30.7%) |

**Opacity rate:** 0.1282 [0.1151, 0.1416] — 1106 covert / 8629 hint-following runs (covert / all hint-following).

## Summary

| Hint type | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] | Acknowledged & followed | Acknowledged & not followed | Not Acknowledged & followed | Not Acknowledged & not followed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| professor | 0.1853 [0.1704, 0.2005] | 0.7444 [0.7121, 0.7752] | 0.4777 [0.4538, 0.5018] | 0.8612 [0.8411, 0.8813] | 628 (39.4% [37.0, 41.8]) | 0.7836 [0.7568, 0.8096] | 0.629 [0.608, 0.650] | 0.540 [0.517, 0.561] | 2303 (18.2%) | 68 (0.5%) | 1144 (9.1%) | 9119 (72.2%) |
| grader_hacking | 0.1950 [0.1798, 0.2102] | 0.7586 [0.7272, 0.7887] | 0.4651 [0.4419, 0.4879] | 0.8830 [0.8649, 0.9005] | 683 (42.8% [40.4, 45.2]) | 0.5944 [0.5628, 0.6253] | 0.630 [0.608, 0.650] | 0.529 [0.507, 0.550] | 1462 (11.6%) | 11 (0.1%) | 2087 (16.6%) | 9000 (71.7%) |
| unethical | 0.5946 [0.5755, 0.6134] | 0.8942 [0.8803, 0.9077] | 0.7151 [0.6985, 0.7314] | 0.9342 [0.9241, 0.9440] | 1327 (83.2% [81.3, 85.0]) | 0.9141 [0.9032, 0.9241] | 0.629 [0.608, 0.650] | 0.271 [0.253, 0.289] | 7523 (59.7%) | 100 (0.8%) | 1106 (8.8%) | 3873 (30.7%) |
