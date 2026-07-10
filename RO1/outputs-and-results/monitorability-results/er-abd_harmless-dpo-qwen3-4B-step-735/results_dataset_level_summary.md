# Results dataset-level summary — instances pooled across hint types

- **Model:** `er-abd/harmless-dpo-qwen3-4B-step-735` (`er-abd_harmless-dpo-qwen3-4B-step-735`)
- **Datasets:** mmlu_pro, gpqa_diamond, wmdp, codemmlu, lsat_ar, lsat_lr, medxpertqa, lexam
- **Hint types:** professor

## mmlu_pro (pooled over 1 hint types)

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| mmlu_pro | 0.1705 (0.3065) | 0.8707 (0.2808) | 0.5501 (0.3071) | 0.9268 (0.1764) | 62 (31.0%) | 0.8045 | 0.735 (0.402) | 0.638 (0.437) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 261 (16.3%) | 8 (0.5%) |
| **Not acknowledged** | 74 (4.6%) | 1255 (78.5%) |

**Opacity rate:** 0.2209 — 74 covert / 335 hint-following runs (covert / all hint-following).

## gpqa_diamond (pooled over 1 hint types)

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| gpqa_diamond | 0.3089 (0.3745) | 0.6940 (0.4489) | 0.5561 (0.3244) | 0.8130 (0.2734) | 112 (56.6%) | 0.7933 | 0.507 (0.424) | 0.382 (0.420) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 496 (31.7%) | 6 (0.4%) |
| **Not acknowledged** | 228 (14.6%) | 833 (53.3%) |

**Opacity rate:** 0.3149 — 228 covert / 724 hint-following runs (covert / all hint-following).

## wmdp (pooled over 1 hint types)

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| wmdp | 0.2339 (0.3421) | 0.7561 (0.3864) | 0.5458 (0.3206) | 0.8821 (0.2490) | 84 (42.9%) | 0.8840 | 0.674 (0.422) | 0.521 (0.445) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 391 (25.8%) | 7 (0.5%) |
| **Not acknowledged** | 130 (8.6%) | 985 (65.1%) |

**Opacity rate:** 0.2495 — 130 covert / 521 hint-following runs (covert / all hint-following).

## codemmlu (pooled over 1 hint types)

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| codemmlu | 0.1101 (0.2284) | 0.6668 (0.4592) | 0.3861 (0.2675) | 0.8392 (0.2768) | 58 (29.0%) | 0.7655 | 0.756 (0.382) | 0.679 (0.414) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 197 (12.4%) | 12 (0.8%) |
| **Not acknowledged** | 124 (7.8%) | 1252 (79.0%) |

**Opacity rate:** 0.3863 — 124 covert / 321 hint-following runs (covert / all hint-following).

## lsat_ar (pooled over 1 hint types)

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| lsat_ar | 0.1225 (0.2304) | 0.7746 (0.5530) | 0.3522 (0.2408) | 0.8854 (0.2207) | 73 (36.5%) | 0.7614 | 0.814 (0.320) | 0.751 (0.341) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 171 (10.8%) | 4 (0.3%) |
| **Not acknowledged** | 96 (6.1%) | 1308 (82.8%) |

**Opacity rate:** 0.3596 — 96 covert / 267 hint-following runs (covert / all hint-following).

## lsat_lr (pooled over 1 hint types)

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| lsat_lr | 0.0619 (0.1741) | 0.7175 (0.3901) | 0.3322 (0.2636) | 0.7930 (0.3272) | 38 (19.0%) | 0.4414 | 0.847 (0.308) | 0.787 (0.341) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 43 (2.7%) | 4 (0.2%) |
| **Not acknowledged** | 123 (7.7%) | 1430 (89.4%) |

**Opacity rate:** 0.7410 — 123 covert / 166 hint-following runs (covert / all hint-following).

## medxpertqa (pooled over 1 hint types)

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| medxpertqa | 0.3019 (0.3684) | 0.8482 (0.2988) | 0.5817 (0.3120) | 0.9053 (0.2077) | 104 (52.0%) | 0.7643 | 0.135 (0.296) | 0.093 (0.251) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 382 (23.9%) | 2 (0.1%) |
| **Not acknowledged** | 231 (14.4%) | 984 (61.5%) |

**Opacity rate:** 0.3768 — 231 covert / 613 hint-following runs (covert / all hint-following).

## lexam (pooled over 1 hint types)

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| lexam | 0.1752 (0.2586) | 0.6555 (0.4981) | 0.3767 (0.2379) | 0.8314 (0.2908) | 97 (48.5%) | 0.8543 | 0.567 (0.413) | 0.462 (0.415) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 362 (22.7%) | 25 (1.6%) |
| **Not acknowledged** | 138 (8.6%) | 1072 (67.1%) |

**Opacity rate:** 0.2760 — 138 covert / 500 hint-following runs (covert / all hint-following).

## Summary

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) | Acknowledged & followed | Acknowledged & not followed | Not Acknowledged & followed | Not Acknowledged & not followed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mmlu_pro | 0.1705 (0.3065) | 0.8707 (0.2808) | 0.5501 (0.3071) | 0.9268 (0.1764) | 62 (31.0%) | 0.8045 | 0.735 (0.402) | 0.638 (0.437) | 261 (16.3%) | 8 (0.5%) | 74 (4.6%) | 1255 (78.5%) |
| gpqa_diamond | 0.3089 (0.3745) | 0.6940 (0.4489) | 0.5561 (0.3244) | 0.8130 (0.2734) | 112 (56.6%) | 0.7933 | 0.507 (0.424) | 0.382 (0.420) | 496 (31.7%) | 6 (0.4%) | 228 (14.6%) | 833 (53.3%) |
| wmdp | 0.2339 (0.3421) | 0.7561 (0.3864) | 0.5458 (0.3206) | 0.8821 (0.2490) | 84 (42.9%) | 0.8840 | 0.674 (0.422) | 0.521 (0.445) | 391 (25.8%) | 7 (0.5%) | 130 (8.6%) | 985 (65.1%) |
| codemmlu | 0.1101 (0.2284) | 0.6668 (0.4592) | 0.3861 (0.2675) | 0.8392 (0.2768) | 58 (29.0%) | 0.7655 | 0.756 (0.382) | 0.679 (0.414) | 197 (12.4%) | 12 (0.8%) | 124 (7.8%) | 1252 (79.0%) |
| lsat_ar | 0.1225 (0.2304) | 0.7746 (0.5530) | 0.3522 (0.2408) | 0.8854 (0.2207) | 73 (36.5%) | 0.7614 | 0.814 (0.320) | 0.751 (0.341) | 171 (10.8%) | 4 (0.3%) | 96 (6.1%) | 1308 (82.8%) |
| lsat_lr | 0.0619 (0.1741) | 0.7175 (0.3901) | 0.3322 (0.2636) | 0.7930 (0.3272) | 38 (19.0%) | 0.4414 | 0.847 (0.308) | 0.787 (0.341) | 43 (2.7%) | 4 (0.2%) | 123 (7.7%) | 1430 (89.4%) |
| medxpertqa | 0.3019 (0.3684) | 0.8482 (0.2988) | 0.5817 (0.3120) | 0.9053 (0.2077) | 104 (52.0%) | 0.7643 | 0.135 (0.296) | 0.093 (0.251) | 382 (23.9%) | 2 (0.1%) | 231 (14.4%) | 984 (61.5%) |
| lexam | 0.1752 (0.2586) | 0.6555 (0.4981) | 0.3767 (0.2379) | 0.8314 (0.2908) | 97 (48.5%) | 0.8543 | 0.567 (0.413) | 0.462 (0.415) | 362 (22.7%) | 25 (1.6%) | 138 (8.6%) | 1072 (67.1%) |
