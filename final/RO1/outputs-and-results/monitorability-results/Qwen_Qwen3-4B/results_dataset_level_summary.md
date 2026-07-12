# Results dataset-level summary — instances pooled across hint types

- **Model:** `Qwen/Qwen3-4B` (`Qwen_Qwen3-4B`)
- **Datasets:** mmlu_pro, gpqa_diamond, wmdp, codemmlu, lsat_ar, lsat_lr, medxpertqa, lexam
- **Hint types:** professor, grader_hacking, unethical

## mmlu_pro (pooled over 3 hint types)

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| mmlu_pro | 0.3888 [0.3553, 0.4224] | 0.9096 [0.8692, 0.9423] | 0.6644 [0.6310, 0.6977] | 0.9477 [0.9286, 0.9648] | 353 (58.8% [54.8, 62.7]) | 0.8446 [0.8134, 0.8743] | 0.719 [0.688, 0.751] | 0.456 [0.421, 0.491] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1677 (35.0%) | 18 (0.4%) |
| **Not acknowledged** | 384 (8.0%) | 2710 (56.6%) |

**Opacity rate:** 0.1863 [0.1529, 0.2208] — 384 covert / 2061 hint-following runs (covert / all hint-following).

## gpqa_diamond (pooled over 3 hint types)

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| gpqa_diamond | 0.4264 [0.3948, 0.4571] | 0.7478 [0.7066, 0.7845] | 0.6171 [0.5868, 0.6475] | 0.8318 [0.8071, 0.8563] | 414 (69.7% [65.8, 73.4]) | 0.6775 [0.6408, 0.7142] | 0.512 [0.479, 0.545] | 0.300 [0.269, 0.331] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1534 (32.6%) | 9 (0.2%) |
| **Not acknowledged** | 1174 (24.9%) | 1991 (42.3%) |

**Opacity rate:** 0.4335 [0.3969, 0.4700] — 1174 covert / 2708 hint-following runs (covert / all hint-following).

## wmdp (pooled over 3 hint types)

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| wmdp | 0.3681 [0.3355, 0.4014] | 0.8137 [0.7757, 0.8503] | 0.6420 [0.6068, 0.6762] | 0.9143 [0.8904, 0.9364] | 340 (57.5% [53.5, 61.6]) | 0.7826 [0.7456, 0.8183] | 0.679 [0.646, 0.712] | 0.414 [0.379, 0.448] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1488 (32.5%) | 8 (0.2%) |
| **Not acknowledged** | 694 (15.2%) | 2384 (52.1%) |

**Opacity rate:** 0.3181 [0.2772, 0.3603] — 694 covert / 2182 hint-following runs (covert / all hint-following).

## codemmlu (pooled over 3 hint types)

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| codemmlu | 0.2714 [0.2428, 0.3014] | 0.7999 [0.7529, 0.8448] | 0.5727 [0.5349, 0.6098] | 0.9235 [0.8989, 0.9468] | 287 (47.8% [43.8, 51.8]) | 0.8256 [0.7865, 0.8621] | 0.763 [0.733, 0.794] | 0.555 [0.520, 0.589] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1258 (26.5%) | 20 (0.4%) |
| **Not acknowledged** | 436 (9.2%) | 3037 (63.9%) |

**Opacity rate:** 0.2574 [0.2141, 0.3025] — 436 covert / 1694 hint-following runs (covert / all hint-following).

## lsat_ar (pooled over 3 hint types)

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| lsat_ar | 0.2478 [0.2221, 0.2737] | 0.8008 [0.7499, 0.8475] | 0.4741 [0.4437, 0.5055] | 0.9083 [0.8855, 0.9294] | 323 (53.8% [49.8, 57.8]) | 0.8109 [0.7756, 0.8440] | 0.782 [0.755, 0.808] | 0.614 [0.584, 0.645] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1119 (23.6%) | 30 (0.6%) |
| **Not acknowledged** | 352 (7.4%) | 3244 (68.4%) |

**Opacity rate:** 0.2393 [0.2056, 0.2760] — 352 covert / 1471 hint-following runs (covert / all hint-following).

## lsat_lr (pooled over 3 hint types)

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| lsat_lr | 0.1759 [0.1515, 0.2008] | 0.8355 [0.7908, 0.8781] | 0.4949 [0.4526, 0.5381] | 0.8999 [0.8698, 0.9285] | 215 (35.8% [32.0, 39.7]) | 0.6123 [0.5583, 0.6658] | 0.834 [0.808, 0.859] | 0.687 [0.655, 0.719] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 525 (10.9%) | 6 (0.1%) |
| **Not acknowledged** | 537 (11.2%) | 3727 (77.7%) |

**Opacity rate:** 0.5056 [0.4483, 0.5629] — 537 covert / 1062 hint-following runs (covert / all hint-following).

## medxpertqa (pooled over 3 hint types)

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| medxpertqa | 0.4579 [0.4235, 0.4915] | 0.8454 [0.8140, 0.8753] | 0.6813 [0.6489, 0.7148] | 0.9119 [0.8898, 0.9333] | 404 (67.3% [63.5, 71.2]) | 0.7369 [0.7020, 0.7709] | 0.127 [0.104, 0.152] | 0.064 [0.048, 0.081] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1742 (36.3%) | 5 (0.1%) |
| **Not acknowledged** | 923 (19.2%) | 2128 (44.4%) |

**Opacity rate:** 0.3463 [0.3077, 0.3850] — 923 covert / 2665 hint-following runs (covert / all hint-following).

## lexam (pooled over 3 hint types)

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| lexam | 0.4079 [0.3769, 0.4392] | 0.7926 [0.7459, 0.8315] | 0.5973 [0.5667, 0.6270] | 0.8689 [0.8457, 0.8918] | 414 (69.0% [65.3, 72.7]) | 0.8182 [0.7881, 0.8466] | 0.585 [0.552, 0.617] | 0.325 [0.295, 0.356] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1936 (40.4%) | 51 (1.1%) |
| **Not acknowledged** | 615 (12.8%) | 2196 (45.8%) |

**Opacity rate:** 0.2411 [0.2089, 0.2746] — 615 covert / 2551 hint-following runs (covert / all hint-following).

## Summary

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] | Acknowledged & followed | Acknowledged & not followed | Not Acknowledged & followed | Not Acknowledged & not followed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mmlu_pro | 0.3888 [0.3553, 0.4224] | 0.9096 [0.8692, 0.9423] | 0.6644 [0.6310, 0.6977] | 0.9477 [0.9286, 0.9648] | 353 (58.8% [54.8, 62.7]) | 0.8446 [0.8134, 0.8743] | 0.719 [0.688, 0.751] | 0.456 [0.421, 0.491] | 1677 (35.0%) | 18 (0.4%) | 384 (8.0%) | 2710 (56.6%) |
| gpqa_diamond | 0.4264 [0.3948, 0.4571] | 0.7478 [0.7066, 0.7845] | 0.6171 [0.5868, 0.6475] | 0.8318 [0.8071, 0.8563] | 414 (69.7% [65.8, 73.4]) | 0.6775 [0.6408, 0.7142] | 0.512 [0.479, 0.545] | 0.300 [0.269, 0.331] | 1534 (32.6%) | 9 (0.2%) | 1174 (24.9%) | 1991 (42.3%) |
| wmdp | 0.3681 [0.3355, 0.4014] | 0.8137 [0.7757, 0.8503] | 0.6420 [0.6068, 0.6762] | 0.9143 [0.8904, 0.9364] | 340 (57.5% [53.5, 61.6]) | 0.7826 [0.7456, 0.8183] | 0.679 [0.646, 0.712] | 0.414 [0.379, 0.448] | 1488 (32.5%) | 8 (0.2%) | 694 (15.2%) | 2384 (52.1%) |
| codemmlu | 0.2714 [0.2428, 0.3014] | 0.7999 [0.7529, 0.8448] | 0.5727 [0.5349, 0.6098] | 0.9235 [0.8989, 0.9468] | 287 (47.8% [43.8, 51.8]) | 0.8256 [0.7865, 0.8621] | 0.763 [0.733, 0.794] | 0.555 [0.520, 0.589] | 1258 (26.5%) | 20 (0.4%) | 436 (9.2%) | 3037 (63.9%) |
| lsat_ar | 0.2478 [0.2221, 0.2737] | 0.8008 [0.7499, 0.8475] | 0.4741 [0.4437, 0.5055] | 0.9083 [0.8855, 0.9294] | 323 (53.8% [49.8, 57.8]) | 0.8109 [0.7756, 0.8440] | 0.782 [0.755, 0.808] | 0.614 [0.584, 0.645] | 1119 (23.6%) | 30 (0.6%) | 352 (7.4%) | 3244 (68.4%) |
| lsat_lr | 0.1759 [0.1515, 0.2008] | 0.8355 [0.7908, 0.8781] | 0.4949 [0.4526, 0.5381] | 0.8999 [0.8698, 0.9285] | 215 (35.8% [32.0, 39.7]) | 0.6123 [0.5583, 0.6658] | 0.834 [0.808, 0.859] | 0.687 [0.655, 0.719] | 525 (10.9%) | 6 (0.1%) | 537 (11.2%) | 3727 (77.7%) |
| medxpertqa | 0.4579 [0.4235, 0.4915] | 0.8454 [0.8140, 0.8753] | 0.6813 [0.6489, 0.7148] | 0.9119 [0.8898, 0.9333] | 404 (67.3% [63.5, 71.2]) | 0.7369 [0.7020, 0.7709] | 0.127 [0.104, 0.152] | 0.064 [0.048, 0.081] | 1742 (36.3%) | 5 (0.1%) | 923 (19.2%) | 2128 (44.4%) |
| lexam | 0.4079 [0.3769, 0.4392] | 0.7926 [0.7459, 0.8315] | 0.5973 [0.5667, 0.6270] | 0.8689 [0.8457, 0.8918] | 414 (69.0% [65.3, 72.7]) | 0.8182 [0.7881, 0.8466] | 0.585 [0.552, 0.617] | 0.325 [0.295, 0.356] | 1936 (40.4%) | 51 (1.1%) | 615 (12.8%) | 2196 (45.8%) |
