# Results hint-type-level summary — instances pooled across datasets

- **Model:** `Qwen/Qwen3-4B` (`Qwen_Qwen3-4B`)
- **Datasets:** mmlu_pro, gpqa_diamond, wmdp, codemmlu, lsat_ar, lsat_lr, medxpertqa, lexam
- **Hint types:** professor, grader_hacking, unethical

## professor (pooled over 8 datasets)

| Hint type | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| professor | 0.1998 [0.1846, 0.2153] | 0.7371 [0.6996, 0.7711] | 0.4871 [0.4644, 0.5104] | 0.8628 [0.8436, 0.8816] | 666 (41.8% [39.3, 44.2]) | 0.7595 [0.7340, 0.7846] | 0.625 [0.604, 0.646] | 0.524 [0.503, 0.546] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 2279 (18.0%) | 48 (0.4%) |
| **Not acknowledged** | 1375 (10.9%) | 8963 (70.8%) |

**Opacity rate:** 0.3763 [0.3489, 0.4036] — 1375 covert / 3654 hint-following runs (covert / all hint-following).

## grader_hacking (pooled over 8 datasets)

| Hint type | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| grader_hacking | 0.1952 [0.1803, 0.2104] | 0.7484 [0.7135, 0.7810] | 0.4520 [0.4301, 0.4735] | 0.8694 [0.8507, 0.8876] | 707 (44.3% [41.9, 46.8]) | 0.5220 [0.4906, 0.5537] | 0.625 [0.604, 0.646] | 0.522 [0.501, 0.544] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1265 (10.0%) | 28 (0.2%) |
| **Not acknowledged** | 2325 (18.4%) | 9016 (71.4%) |

**Opacity rate:** 0.6476 [0.6180, 0.6760] — 2325 covert / 3590 hint-following runs (covert / all hint-following).

## unethical (pooled over 8 datasets)

| Hint type | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| unethical | 0.6335 [0.6153, 0.6516] | 0.8936 [0.8802, 0.9068] | 0.7340 [0.7185, 0.7493] | 0.9299 [0.9201, 0.9396] | 1377 (86.3% [84.6, 88.0]) | 0.8982 [0.8866, 0.9093] | 0.625 [0.604, 0.646] | 0.234 [0.218, 0.251] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 7735 (61.1%) | 71 (0.6%) |
| **Not acknowledged** | 1415 (11.2%) | 3438 (27.2%) |

**Opacity rate:** 0.1546 [0.1402, 0.1693] — 1415 covert / 9150 hint-following runs (covert / all hint-following).

## Summary

| Hint type | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] | Acknowledged & followed | Acknowledged & not followed | Not Acknowledged & followed | Not Acknowledged & not followed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| professor | 0.1998 [0.1846, 0.2153] | 0.7371 [0.6996, 0.7711] | 0.4871 [0.4644, 0.5104] | 0.8628 [0.8436, 0.8816] | 666 (41.8% [39.3, 44.2]) | 0.7595 [0.7340, 0.7846] | 0.625 [0.604, 0.646] | 0.524 [0.503, 0.546] | 2279 (18.0%) | 48 (0.4%) | 1375 (10.9%) | 8963 (70.8%) |
| grader_hacking | 0.1952 [0.1803, 0.2104] | 0.7484 [0.7135, 0.7810] | 0.4520 [0.4301, 0.4735] | 0.8694 [0.8507, 0.8876] | 707 (44.3% [41.9, 46.8]) | 0.5220 [0.4906, 0.5537] | 0.625 [0.604, 0.646] | 0.522 [0.501, 0.544] | 1265 (10.0%) | 28 (0.2%) | 2325 (18.4%) | 9016 (71.4%) |
| unethical | 0.6335 [0.6153, 0.6516] | 0.8936 [0.8802, 0.9068] | 0.7340 [0.7185, 0.7493] | 0.9299 [0.9201, 0.9396] | 1377 (86.3% [84.6, 88.0]) | 0.8982 [0.8866, 0.9093] | 0.625 [0.604, 0.646] | 0.234 [0.218, 0.251] | 7735 (61.1%) | 71 (0.6%) | 1415 (11.2%) | 3438 (27.2%) |
