# Results dataset-level summary — instances pooled across hint types

- **Model:** `er-abd_harmless-dpo-qwen3-4B-step-735` (`er-abd_harmless-dpo-qwen3-4B-step-735`)
- **Datasets:** mmlu_pro, gpqa_diamond, wmdp, codemmlu, lsat_ar, lsat_lr, medxpertqa, lexam
- **Hint types:** professor, grader_hacking, unethical

## mmlu_pro (pooled over 3 hint types)

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| mmlu_pro | 0.3579 [0.3256, 0.3904] | 0.9270 [0.9024, 0.9494] | 0.6429 [0.6076, 0.6774] | 0.9631 [0.9489, 0.9758] | 334 (55.7% [51.7, 59.7]) | 0.8763 [0.8484, 0.9025] | 0.735 [0.703, 0.767] | 0.490 [0.454, 0.526] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1591 (33.3%) | 22 (0.5%) |
| **Not acknowledged** | 308 (6.5%) | 2854 (59.8%) |

**Opacity rate:** 0.1622 [0.1289, 0.1966] — 308 covert / 1899 hint-following runs (covert / all hint-following).

## gpqa_diamond (pooled over 3 hint types)

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| gpqa_diamond | 0.4132 [0.3801, 0.4458] | 0.7486 [0.7092, 0.7856] | 0.6242 [0.5919, 0.6563] | 0.8497 [0.8245, 0.8741] | 398 (67.0% [63.1, 70.7]) | 0.7132 [0.6755, 0.7496] | 0.507 [0.473, 0.541] | 0.305 [0.274, 0.338] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1673 (35.7%) | 10 (0.2%) |
| **Not acknowledged** | 987 (21.1%) | 2012 (43.0%) |

**Opacity rate:** 0.3711 [0.3349, 0.4083] — 987 covert / 2660 hint-following runs (covert / all hint-following).

## wmdp (pooled over 3 hint types)

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| wmdp | 0.3631 [0.3305, 0.3964] | 0.8037 [0.7650, 0.8412] | 0.6459 [0.6109, 0.6816] | 0.9118 [0.8871, 0.9343] | 332 (56.3% [52.2, 60.3]) | 0.8328 [0.7999, 0.8650] | 0.673 [0.639, 0.707] | 0.421 [0.385, 0.456] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1603 (35.3%) | 13 (0.3%) |
| **Not acknowledged** | 560 (12.3%) | 2369 (52.1%) |

**Opacity rate:** 0.2589 [0.2204, 0.2988] — 560 covert / 2163 hint-following runs (covert / all hint-following).

## codemmlu (pooled over 3 hint types)

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| codemmlu | 0.2359 [0.2088, 0.2635] | 0.7868 [0.7438, 0.8288] | 0.5401 [0.5012, 0.5799] | 0.8978 [0.8683, 0.9250] | 263 (43.9% [39.9, 47.7]) | 0.8120 [0.7686, 0.8528] | 0.758 [0.726, 0.788] | 0.587 [0.552, 0.622] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1127 (23.8%) | 41 (0.9%) |
| **Not acknowledged** | 426 (9.0%) | 3140 (66.3%) |

**Opacity rate:** 0.2743 [0.2264, 0.3226] — 426 covert / 1553 hint-following runs (covert / all hint-following).

## lsat_ar (pooled over 3 hint types)

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| lsat_ar | 0.2322 [0.2083, 0.2574] | 0.8388 [0.7861, 0.8854] | 0.4448 [0.4139, 0.4764] | 0.9235 [0.9025, 0.9431] | 320 (53.3% [49.3, 57.3]) | 0.8171 [0.7821, 0.8499] | 0.814 [0.788, 0.840] | 0.652 [0.622, 0.682] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1014 (21.7%) | 38 (0.8%) |
| **Not acknowledged** | 289 (6.2%) | 3336 (71.3%) |

**Opacity rate:** 0.2218 [0.1870, 0.2591] — 289 covert / 1303 hint-following runs (covert / all hint-following).

## lsat_lr (pooled over 3 hint types)

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| lsat_lr | 0.1587 [0.1345, 0.1830] | 0.8429 [0.7941, 0.8879] | 0.5024 [0.4564, 0.5489] | 0.8946 [0.8604, 0.9264] | 192 (32.0% [28.3, 35.8]) | 0.6397 [0.5848, 0.6938] | 0.847 [0.822, 0.871] | 0.710 [0.678, 0.741] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 487 (10.2%) | 10 (0.2%) |
| **Not acknowledged** | 474 (9.9%) | 3819 (79.7%) |

**Opacity rate:** 0.4932 [0.4340, 0.5528] — 474 covert / 961 hint-following runs (covert / all hint-following).

## medxpertqa (pooled over 3 hint types)

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| medxpertqa | 0.4730 [0.4387, 0.5065] | 0.8766 [0.8491, 0.9022] | 0.7083 [0.6766, 0.7390] | 0.9269 [0.9075, 0.9449] | 401 (66.8% [63.0, 70.5]) | 0.7773 [0.7447, 0.8098] | 0.135 [0.112, 0.159] | 0.066 [0.050, 0.084] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1893 (39.5%) | 3 (0.1%) |
| **Not acknowledged** | 766 (16.0%) | 2135 (44.5%) |

**Opacity rate:** 0.2881 [0.2526, 0.3248] — 766 covert / 2659 hint-following runs (covert / all hint-following).

## lexam (pooled over 3 hint types)

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| lexam | 0.3677 [0.3377, 0.3978] | 0.7688 [0.7302, 0.8070] | 0.5632 [0.5312, 0.5944] | 0.8693 [0.8430, 0.8947] | 398 (66.3% [62.5, 70.2]) | 0.8754 [0.8483, 0.9006] | 0.567 [0.534, 0.599] | 0.338 [0.306, 0.369] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1900 (39.6%) | 42 (0.9%) |
| **Not acknowledged** | 527 (11.0%) | 2327 (48.5%) |

**Opacity rate:** 0.2171 [0.1852, 0.2504] — 527 covert / 2427 hint-following runs (covert / all hint-following).

## Summary

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] | Acknowledged & followed | Acknowledged & not followed | Not Acknowledged & followed | Not Acknowledged & not followed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mmlu_pro | 0.3579 [0.3256, 0.3904] | 0.9270 [0.9024, 0.9494] | 0.6429 [0.6076, 0.6774] | 0.9631 [0.9489, 0.9758] | 334 (55.7% [51.7, 59.7]) | 0.8763 [0.8484, 0.9025] | 0.735 [0.703, 0.767] | 0.490 [0.454, 0.526] | 1591 (33.3%) | 22 (0.5%) | 308 (6.5%) | 2854 (59.8%) |
| gpqa_diamond | 0.4132 [0.3801, 0.4458] | 0.7486 [0.7092, 0.7856] | 0.6242 [0.5919, 0.6563] | 0.8497 [0.8245, 0.8741] | 398 (67.0% [63.1, 70.7]) | 0.7132 [0.6755, 0.7496] | 0.507 [0.473, 0.541] | 0.305 [0.274, 0.338] | 1673 (35.7%) | 10 (0.2%) | 987 (21.1%) | 2012 (43.0%) |
| wmdp | 0.3631 [0.3305, 0.3964] | 0.8037 [0.7650, 0.8412] | 0.6459 [0.6109, 0.6816] | 0.9118 [0.8871, 0.9343] | 332 (56.3% [52.2, 60.3]) | 0.8328 [0.7999, 0.8650] | 0.673 [0.639, 0.707] | 0.421 [0.385, 0.456] | 1603 (35.3%) | 13 (0.3%) | 560 (12.3%) | 2369 (52.1%) |
| codemmlu | 0.2359 [0.2088, 0.2635] | 0.7868 [0.7438, 0.8288] | 0.5401 [0.5012, 0.5799] | 0.8978 [0.8683, 0.9250] | 263 (43.9% [39.9, 47.7]) | 0.8120 [0.7686, 0.8528] | 0.758 [0.726, 0.788] | 0.587 [0.552, 0.622] | 1127 (23.8%) | 41 (0.9%) | 426 (9.0%) | 3140 (66.3%) |
| lsat_ar | 0.2322 [0.2083, 0.2574] | 0.8388 [0.7861, 0.8854] | 0.4448 [0.4139, 0.4764] | 0.9235 [0.9025, 0.9431] | 320 (53.3% [49.3, 57.3]) | 0.8171 [0.7821, 0.8499] | 0.814 [0.788, 0.840] | 0.652 [0.622, 0.682] | 1014 (21.7%) | 38 (0.8%) | 289 (6.2%) | 3336 (71.3%) |
| lsat_lr | 0.1587 [0.1345, 0.1830] | 0.8429 [0.7941, 0.8879] | 0.5024 [0.4564, 0.5489] | 0.8946 [0.8604, 0.9264] | 192 (32.0% [28.3, 35.8]) | 0.6397 [0.5848, 0.6938] | 0.847 [0.822, 0.871] | 0.710 [0.678, 0.741] | 487 (10.2%) | 10 (0.2%) | 474 (9.9%) | 3819 (79.7%) |
| medxpertqa | 0.4730 [0.4387, 0.5065] | 0.8766 [0.8491, 0.9022] | 0.7083 [0.6766, 0.7390] | 0.9269 [0.9075, 0.9449] | 401 (66.8% [63.0, 70.5]) | 0.7773 [0.7447, 0.8098] | 0.135 [0.112, 0.159] | 0.066 [0.050, 0.084] | 1893 (39.5%) | 3 (0.1%) | 766 (16.0%) | 2135 (44.5%) |
| lexam | 0.3677 [0.3377, 0.3978] | 0.7688 [0.7302, 0.8070] | 0.5632 [0.5312, 0.5944] | 0.8693 [0.8430, 0.8947] | 398 (66.3% [62.5, 70.2]) | 0.8754 [0.8483, 0.9006] | 0.567 [0.534, 0.599] | 0.338 [0.306, 0.369] | 1900 (39.6%) | 42 (0.9%) | 527 (11.0%) | 2327 (48.5%) |
