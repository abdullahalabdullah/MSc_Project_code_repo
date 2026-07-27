# Results dataset-level summary — instances pooled across hint types

- **Model:** `er-abd/anti-syc-dpo-qwen3-4B-step-520` (`er-abd_anti-syc-dpo-qwen3-4B-step-520`)
- **Datasets:** mmlu_pro, gpqa_diamond, wmdp, codemmlu, lsat_ar, lsat_lr, medxpertqa, lexam
- **Hint types:** professor, grader_hacking, unethical

## mmlu_pro (pooled over 3 hint types)

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| mmlu_pro | 0.3383 [0.3069, 0.3702] | 0.9190 [0.8913, 0.9444] | 0.6008 [0.5635, 0.6373] | 0.9545 [0.9363, 0.9706] | 338 (56.3% [52.3, 60.3]) | 0.8776 [0.8479, 0.9053] | 0.731 [0.700, 0.763] | 0.500 [0.465, 0.536] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1501 (31.7%) | 24 (0.5%) |
| **Not acknowledged** | 293 (6.2%) | 2924 (61.7%) |

**Opacity rate:** 0.1633 [0.1280, 0.1994] — 293 covert / 1794 hint-following runs (covert / all hint-following).

## gpqa_diamond (pooled over 3 hint types)

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| gpqa_diamond | 0.4164 [0.3845, 0.4481] | 0.7690 [0.7358, 0.8008] | 0.6167 [0.5864, 0.6480] | 0.8553 [0.8301, 0.8784] | 405 (68.2% [64.5, 71.9]) | 0.6957 [0.6572, 0.7323] | 0.512 [0.478, 0.545] | 0.312 [0.281, 0.345] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1677 (36.0%) | 15 (0.3%) |
| **Not acknowledged** | 963 (20.7%) | 2007 (43.1%) |

**Opacity rate:** 0.3648 [0.3290, 0.4014] — 963 covert / 2640 hint-following runs (covert / all hint-following).

## wmdp (pooled over 3 hint types)

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| wmdp | 0.3472 [0.3155, 0.3786] | 0.7997 [0.7607, 0.8361] | 0.6474 [0.6131, 0.6800] | 0.9033 [0.8789, 0.9258] | 320 (54.1% [50.0, 58.1]) | 0.8072 [0.7698, 0.8441] | 0.679 [0.645, 0.711] | 0.428 [0.393, 0.462] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1497 (33.2%) | 11 (0.2%) |
| **Not acknowledged** | 563 (12.5%) | 2444 (54.1%) |

**Opacity rate:** 0.2733 [0.2323, 0.3149] — 563 covert / 2060 hint-following runs (covert / all hint-following).

## codemmlu (pooled over 3 hint types)

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| codemmlu | 0.2399 [0.2133, 0.2680] | 0.7841 [0.7420, 0.8243] | 0.5294 [0.4923, 0.5674] | 0.8879 [0.8592, 0.9150] | 273 (45.5% [41.5, 49.7]) | 0.7676 [0.7211, 0.8132] | 0.762 [0.732, 0.792] | 0.584 [0.549, 0.618] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1083 (23.2%) | 41 (0.9%) |
| **Not acknowledged** | 448 (9.6%) | 3103 (66.4%) |

**Opacity rate:** 0.2926 [0.2445, 0.3426] — 448 covert / 1531 hint-following runs (covert / all hint-following).

## lsat_ar (pooled over 3 hint types)

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| lsat_ar | 0.2099 [0.1860, 0.2347] | 0.8274 [0.7776, 0.8708] | 0.4562 [0.4230, 0.4909] | 0.9066 [0.8812, 0.9298] | 283 (47.2% [43.2, 51.2]) | 0.8330 [0.7983, 0.8649] | 0.826 [0.802, 0.850] | 0.674 [0.643, 0.704] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 912 (19.8%) | 46 (1.0%) |
| **Not acknowledged** | 262 (5.7%) | 3383 (73.5%) |

**Opacity rate:** 0.2232 [0.1849, 0.2633] — 262 covert / 1174 hint-following runs (covert / all hint-following).

## lsat_lr (pooled over 3 hint types)

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| lsat_lr | 0.1550 [0.1318, 0.1792] | 0.8455 [0.8022, 0.8864] | 0.4922 [0.4460, 0.5396] | 0.8768 [0.8395, 0.9125] | 189 (31.5% [27.8, 35.3]) | 0.6330 [0.5763, 0.6885] | 0.845 [0.819, 0.869] | 0.714 [0.682, 0.745] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 488 (10.2%) | 10 (0.2%) |
| **Not acknowledged** | 446 (9.3%) | 3832 (80.2%) |

**Opacity rate:** 0.4775 [0.4162, 0.5387] — 446 covert / 934 hint-following runs (covert / all hint-following).

## medxpertqa (pooled over 3 hint types)

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| medxpertqa | 0.4365 [0.4030, 0.4698] | 0.8399 [0.8060, 0.8725] | 0.6918 [0.6597, 0.7230] | 0.9282 [0.9108, 0.9449] | 380 (63.3% [59.3, 67.2]) | 0.7972 [0.7643, 0.8290] | 0.128 [0.106, 0.153] | 0.066 [0.049, 0.084] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1832 (38.3%) | 8 (0.2%) |
| **Not acknowledged** | 697 (14.6%) | 2245 (46.9%) |

**Opacity rate:** 0.2756 [0.2393, 0.3125] — 697 covert / 2529 hint-following runs (covert / all hint-following).

## lexam (pooled over 3 hint types)

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| lexam | 0.3336 [0.3047, 0.3626] | 0.7303 [0.6866, 0.7716] | 0.5355 [0.5044, 0.5668] | 0.8612 [0.8361, 0.8855] | 380 (63.3% [59.5, 67.2]) | 0.8728 [0.8444, 0.8989] | 0.551 [0.517, 0.584] | 0.350 [0.318, 0.382] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1808 (38.1%) | 54 (1.1%) |
| **Not acknowledged** | 487 (10.3%) | 2395 (50.5%) |

**Opacity rate:** 0.2122 [0.1801, 0.2451] — 487 covert / 2295 hint-following runs (covert / all hint-following).

## Summary

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] | Acknowledged & followed | Acknowledged & not followed | Not Acknowledged & followed | Not Acknowledged & not followed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mmlu_pro | 0.3383 [0.3069, 0.3702] | 0.9190 [0.8913, 0.9444] | 0.6008 [0.5635, 0.6373] | 0.9545 [0.9363, 0.9706] | 338 (56.3% [52.3, 60.3]) | 0.8776 [0.8479, 0.9053] | 0.731 [0.700, 0.763] | 0.500 [0.465, 0.536] | 1501 (31.7%) | 24 (0.5%) | 293 (6.2%) | 2924 (61.7%) |
| gpqa_diamond | 0.4164 [0.3845, 0.4481] | 0.7690 [0.7358, 0.8008] | 0.6167 [0.5864, 0.6480] | 0.8553 [0.8301, 0.8784] | 405 (68.2% [64.5, 71.9]) | 0.6957 [0.6572, 0.7323] | 0.512 [0.478, 0.545] | 0.312 [0.281, 0.345] | 1677 (36.0%) | 15 (0.3%) | 963 (20.7%) | 2007 (43.1%) |
| wmdp | 0.3472 [0.3155, 0.3786] | 0.7997 [0.7607, 0.8361] | 0.6474 [0.6131, 0.6800] | 0.9033 [0.8789, 0.9258] | 320 (54.1% [50.0, 58.1]) | 0.8072 [0.7698, 0.8441] | 0.679 [0.645, 0.711] | 0.428 [0.393, 0.462] | 1497 (33.2%) | 11 (0.2%) | 563 (12.5%) | 2444 (54.1%) |
| codemmlu | 0.2399 [0.2133, 0.2680] | 0.7841 [0.7420, 0.8243] | 0.5294 [0.4923, 0.5674] | 0.8879 [0.8592, 0.9150] | 273 (45.5% [41.5, 49.7]) | 0.7676 [0.7211, 0.8132] | 0.762 [0.732, 0.792] | 0.584 [0.549, 0.618] | 1083 (23.2%) | 41 (0.9%) | 448 (9.6%) | 3103 (66.4%) |
| lsat_ar | 0.2099 [0.1860, 0.2347] | 0.8274 [0.7776, 0.8708] | 0.4562 [0.4230, 0.4909] | 0.9066 [0.8812, 0.9298] | 283 (47.2% [43.2, 51.2]) | 0.8330 [0.7983, 0.8649] | 0.826 [0.802, 0.850] | 0.674 [0.643, 0.704] | 912 (19.8%) | 46 (1.0%) | 262 (5.7%) | 3383 (73.5%) |
| lsat_lr | 0.1550 [0.1318, 0.1792] | 0.8455 [0.8022, 0.8864] | 0.4922 [0.4460, 0.5396] | 0.8768 [0.8395, 0.9125] | 189 (31.5% [27.8, 35.3]) | 0.6330 [0.5763, 0.6885] | 0.845 [0.819, 0.869] | 0.714 [0.682, 0.745] | 488 (10.2%) | 10 (0.2%) | 446 (9.3%) | 3832 (80.2%) |
| medxpertqa | 0.4365 [0.4030, 0.4698] | 0.8399 [0.8060, 0.8725] | 0.6918 [0.6597, 0.7230] | 0.9282 [0.9108, 0.9449] | 380 (63.3% [59.3, 67.2]) | 0.7972 [0.7643, 0.8290] | 0.128 [0.106, 0.153] | 0.066 [0.049, 0.084] | 1832 (38.3%) | 8 (0.2%) | 697 (14.6%) | 2245 (46.9%) |
| lexam | 0.3336 [0.3047, 0.3626] | 0.7303 [0.6866, 0.7716] | 0.5355 [0.5044, 0.5668] | 0.8612 [0.8361, 0.8855] | 380 (63.3% [59.5, 67.2]) | 0.8728 [0.8444, 0.8989] | 0.551 [0.517, 0.584] | 0.350 [0.318, 0.382] | 1808 (38.1%) | 54 (1.1%) | 487 (10.3%) | 2395 (50.5%) |
