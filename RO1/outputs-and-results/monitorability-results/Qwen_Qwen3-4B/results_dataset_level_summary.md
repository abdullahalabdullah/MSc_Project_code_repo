# Results dataset-level summary — instances pooled across hint types

- **Model:** `Qwen/Qwen3-4B` (`Qwen_Qwen3-4B`)
- **Datasets:** mmlu_pro, gpqa_diamond, wmdp, codemmlu, lsat_ar, lsat_lr, medxpertqa, lexam
- **Hint types:** professor, grader_hacking, unethical

## mmlu_pro (pooled over 3 hint types)

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| mmlu_pro | 0.3888 (0.4132) | 0.9096 (0.3540) | 0.6644 (0.3234) | 0.9477 (0.1741) | 353 (58.8%) | 0.8446 | 0.719 (0.399) | 0.456 (0.434) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1677 (35.0%) | 18 (0.4%) |
| **Not acknowledged** | 384 (8.0%) | 2710 (56.6%) |

**Opacity rate:** 0.1863 — 384 covert / 2061 hint-following runs (covert / all hint-following).

## gpqa_diamond (pooled over 3 hint types)

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| gpqa_diamond | 0.4264 (0.3910) | 0.7478 (0.4197) | 0.6171 (0.3128) | 0.8318 (0.2573) | 414 (69.7%) | 0.6775 | 0.512 (0.411) | 0.300 (0.384) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1534 (32.6%) | 9 (0.2%) |
| **Not acknowledged** | 1174 (24.9%) | 1991 (42.3%) |

**Opacity rate:** 0.4335 — 1174 covert / 2708 hint-following runs (covert / all hint-following).

## wmdp (pooled over 3 hint types)

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| wmdp | 0.3681 (0.4043) | 0.8137 (0.3622) | 0.6420 (0.3270) | 0.9143 (0.2170) | 340 (57.5%) | 0.7826 | 0.679 (0.407) | 0.414 (0.428) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1488 (32.5%) | 8 (0.2%) |
| **Not acknowledged** | 694 (15.2%) | 2384 (52.1%) |

**Opacity rate:** 0.3181 — 694 covert / 2182 hint-following runs (covert / all hint-following).

## codemmlu (pooled over 3 hint types)

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| codemmlu | 0.2714 (0.3672) | 0.7999 (0.4257) | 0.5727 (0.3266) | 0.9235 (0.2079) | 287 (47.8%) | 0.8256 | 0.763 (0.381) | 0.555 (0.431) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1258 (26.5%) | 20 (0.4%) |
| **Not acknowledged** | 436 (9.2%) | 3037 (63.9%) |

**Opacity rate:** 0.2574 — 436 covert / 1694 hint-following runs (covert / all hint-following).

## lsat_ar (pooled over 3 hint types)

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| lsat_ar | 0.2478 (0.3271) | 0.8008 (0.4732) | 0.4741 (0.2889) | 0.9083 (0.2034) | 323 (53.8%) | 0.8109 | 0.782 (0.327) | 0.614 (0.386) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1119 (23.6%) | 30 (0.6%) |
| **Not acknowledged** | 352 (7.4%) | 3244 (68.4%) |

**Opacity rate:** 0.2393 — 352 covert / 1471 hint-following runs (covert / all hint-following).

## lsat_lr (pooled over 3 hint types)

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| lsat_lr | 0.1759 (0.3070) | 0.8355 (0.3398) | 0.4949 (0.3219) | 0.8999 (0.2226) | 215 (35.8%) | 0.6123 | 0.834 (0.320) | 0.687 (0.393) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 525 (10.9%) | 6 (0.1%) |
| **Not acknowledged** | 537 (11.2%) | 3727 (77.7%) |

**Opacity rate:** 0.5056 — 537 covert / 1062 hint-following runs (covert / all hint-following).

## medxpertqa (pooled over 3 hint types)

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| medxpertqa | 0.4579 (0.4219) | 0.8454 (0.3249) | 0.6813 (0.3336) | 0.9119 (0.2193) | 404 (67.3%) | 0.7369 | 0.127 (0.295) | 0.064 (0.206) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1742 (36.3%) | 5 (0.1%) |
| **Not acknowledged** | 923 (19.2%) | 2128 (44.4%) |

**Opacity rate:** 0.3463 — 923 covert / 2665 hint-following runs (covert / all hint-following).

## lexam (pooled over 3 hint types)

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| lexam | 0.4079 (0.3872) | 0.7926 (0.4620) | 0.5973 (0.3149) | 0.8689 (0.2409) | 414 (69.0%) | 0.8182 | 0.585 (0.402) | 0.325 (0.386) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1936 (40.4%) | 51 (1.1%) |
| **Not acknowledged** | 615 (12.8%) | 2196 (45.8%) |

**Opacity rate:** 0.2411 — 615 covert / 2551 hint-following runs (covert / all hint-following).

## Summary

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) | Acknowledged & followed | Acknowledged & not followed | Not Acknowledged & followed | Not Acknowledged & not followed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mmlu_pro | 0.3888 (0.4132) | 0.9096 (0.3540) | 0.6644 (0.3234) | 0.9477 (0.1741) | 353 (58.8%) | 0.8446 | 0.719 (0.399) | 0.456 (0.434) | 1677 (35.0%) | 18 (0.4%) | 384 (8.0%) | 2710 (56.6%) |
| gpqa_diamond | 0.4264 (0.3910) | 0.7478 (0.4197) | 0.6171 (0.3128) | 0.8318 (0.2573) | 414 (69.7%) | 0.6775 | 0.512 (0.411) | 0.300 (0.384) | 1534 (32.6%) | 9 (0.2%) | 1174 (24.9%) | 1991 (42.3%) |
| wmdp | 0.3681 (0.4043) | 0.8137 (0.3622) | 0.6420 (0.3270) | 0.9143 (0.2170) | 340 (57.5%) | 0.7826 | 0.679 (0.407) | 0.414 (0.428) | 1488 (32.5%) | 8 (0.2%) | 694 (15.2%) | 2384 (52.1%) |
| codemmlu | 0.2714 (0.3672) | 0.7999 (0.4257) | 0.5727 (0.3266) | 0.9235 (0.2079) | 287 (47.8%) | 0.8256 | 0.763 (0.381) | 0.555 (0.431) | 1258 (26.5%) | 20 (0.4%) | 436 (9.2%) | 3037 (63.9%) |
| lsat_ar | 0.2478 (0.3271) | 0.8008 (0.4732) | 0.4741 (0.2889) | 0.9083 (0.2034) | 323 (53.8%) | 0.8109 | 0.782 (0.327) | 0.614 (0.386) | 1119 (23.6%) | 30 (0.6%) | 352 (7.4%) | 3244 (68.4%) |
| lsat_lr | 0.1759 (0.3070) | 0.8355 (0.3398) | 0.4949 (0.3219) | 0.8999 (0.2226) | 215 (35.8%) | 0.6123 | 0.834 (0.320) | 0.687 (0.393) | 525 (10.9%) | 6 (0.1%) | 537 (11.2%) | 3727 (77.7%) |
| medxpertqa | 0.4579 (0.4219) | 0.8454 (0.3249) | 0.6813 (0.3336) | 0.9119 (0.2193) | 404 (67.3%) | 0.7369 | 0.127 (0.295) | 0.064 (0.206) | 1742 (36.3%) | 5 (0.1%) | 923 (19.2%) | 2128 (44.4%) |
| lexam | 0.4079 (0.3872) | 0.7926 (0.4620) | 0.5973 (0.3149) | 0.8689 (0.2409) | 414 (69.0%) | 0.8182 | 0.585 (0.402) | 0.325 (0.386) | 1936 (40.4%) | 51 (1.1%) | 615 (12.8%) | 2196 (45.8%) |
