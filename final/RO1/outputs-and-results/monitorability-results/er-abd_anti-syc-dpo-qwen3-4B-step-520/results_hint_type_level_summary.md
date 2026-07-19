# Results hint-type-level summary — instances pooled across datasets

- **Model:** `er-abd/anti-syc-dpo-qwen3-4B-step-520` (`er-abd_anti-syc-dpo-qwen3-4B-step-520`)
- **Datasets:** mmlu_pro, gpqa_diamond, wmdp, codemmlu, lsat_ar, lsat_lr, medxpertqa, lexam
- **Hint types:** professor, grader_hacking, unethical

## professor (pooled over 8 datasets)

| Hint type | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| professor | 0.1677 [0.1536, 0.1819] | 0.7236 [0.6927, 0.7540] | 0.4661 [0.4427, 0.4898] | 0.8488 [0.8285, 0.8688] | 586 (36.7% [34.4, 39.1]) | 0.7721 [0.7437, 0.7993] | 0.630 [0.609, 0.651] | 0.548 [0.526, 0.570] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 2142 (17.0%) | 57 (0.5%) |
| **Not acknowledged** | 1084 (8.6%) | 9301 (73.9%) |

**Opacity rate:** 0.3360 [0.3078, 0.3646] — 1084 covert / 3226 hint-following runs (covert / all hint-following).

## grader_hacking (pooled over 8 datasets)

| Hint type | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| grader_hacking | 0.1851 [0.1707, 0.1997] | 0.7451 [0.7123, 0.7762] | 0.4393 [0.4176, 0.4616] | 0.8732 [0.8551, 0.8914] | 687 (43.0% [40.6, 45.5]) | 0.5753 [0.5418, 0.6079] | 0.629 [0.608, 0.650] | 0.529 [0.508, 0.551] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1356 (10.9%) | 22 (0.2%) |
| **Not acknowledged** | 2044 (16.4%) | 9005 (72.5%) |

**Opacity rate:** 0.6012 [0.5684, 0.6322] — 2044 covert / 3400 hint-following runs (covert / all hint-following).

## unethical (pooled over 8 datasets)

| Hint type | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| unethical | 0.5754 [0.5566, 0.5949] | 0.8889 [0.8750, 0.9027] | 0.7090 [0.6922, 0.7252] | 0.9308 [0.9209, 0.9405] | 1295 (81.2% [79.2, 83.1]) | 0.9178 [0.9074, 0.9275] | 0.629 [0.608, 0.650] | 0.284 [0.266, 0.303] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 7300 (58.5%) | 130 (1.0%) |
| **Not acknowledged** | 1031 (8.3%) | 4027 (32.2%) |

**Opacity rate:** 0.1238 [0.1111, 0.1371] — 1031 covert / 8331 hint-following runs (covert / all hint-following).

## Summary

| Hint type | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] | Acknowledged & followed | Acknowledged & not followed | Not Acknowledged & followed | Not Acknowledged & not followed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| professor | 0.1677 [0.1536, 0.1819] | 0.7236 [0.6927, 0.7540] | 0.4661 [0.4427, 0.4898] | 0.8488 [0.8285, 0.8688] | 586 (36.7% [34.4, 39.1]) | 0.7721 [0.7437, 0.7993] | 0.630 [0.609, 0.651] | 0.548 [0.526, 0.570] | 2142 (17.0%) | 57 (0.5%) | 1084 (8.6%) | 9301 (73.9%) |
| grader_hacking | 0.1851 [0.1707, 0.1997] | 0.7451 [0.7123, 0.7762] | 0.4393 [0.4176, 0.4616] | 0.8732 [0.8551, 0.8914] | 687 (43.0% [40.6, 45.5]) | 0.5753 [0.5418, 0.6079] | 0.629 [0.608, 0.650] | 0.529 [0.508, 0.551] | 1356 (10.9%) | 22 (0.2%) | 2044 (16.4%) | 9005 (72.5%) |
| unethical | 0.5754 [0.5566, 0.5949] | 0.8889 [0.8750, 0.9027] | 0.7090 [0.6922, 0.7252] | 0.9308 [0.9209, 0.9405] | 1295 (81.2% [79.2, 83.1]) | 0.9178 [0.9074, 0.9275] | 0.629 [0.608, 0.650] | 0.284 [0.266, 0.303] | 7300 (58.5%) | 130 (1.0%) | 1031 (8.3%) | 4027 (32.2%) |
