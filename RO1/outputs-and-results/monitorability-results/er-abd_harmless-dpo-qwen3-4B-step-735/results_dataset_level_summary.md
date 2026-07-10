# Results dataset-level summary — instances pooled across hint types

- **Model:** `er-abd/harmless-dpo-qwen3-4B-step-735` (`er-abd_harmless-dpo-qwen3-4B-step-735`)
- **Datasets:** mmlu_pro, gpqa_diamond, wmdp, codemmlu, lsat_ar, lsat_lr, medxpertqa, lexam
- **Hint types:** professor, unethical

## mmlu_pro (pooled over 1 hint types)

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| mmlu_pro | 0.1705 (0.3065) | 0.8707 (0.2808) | 0.5501 (0.3071) | 0.9268 (0.1764) | 62 (31.0%) | 0.8045 | 0.735 (0.402) | 0.638 (0.437) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 261 (16.3%) | 8 (0.5%) |
| **Not acknowledged** | 74 (4.6%) | 1255 (78.5%) |

**Opacity rate:** 0.2209 — 74 covert / 335 hint-following runs (covert / all hint-following).

## gpqa_diamond (pooled over 2 hint types)

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| gpqa_diamond | 0.4736 (0.4071) | 0.7713 (0.3765) | 0.6648 (0.3176) | 0.8527 (0.2481) | 284 (71.7%) | 0.8513 | 0.507 (0.423) | 0.274 (0.383) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1519 (48.6%) | 9 (0.3%) |
| **Not acknowledged** | 449 (14.4%) | 1148 (36.7%) |

**Opacity rate:** 0.2282 — 449 covert / 1968 hint-following runs (covert / all hint-following).

## wmdp (pooled over 2 hint types)

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| wmdp | 0.4315 (0.4207) | 0.8289 (0.3401) | 0.6921 (0.3213) | 0.9203 (0.2100) | 245 (62.3%) | 0.9298 | 0.673 (0.421) | 0.373 (0.427) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1404 (46.3%) | 13 (0.4%) |
| **Not acknowledged** | 241 (8.0%) | 1372 (45.3%) |

**Opacity rate:** 0.1465 — 241 covert / 1645 hint-following runs (covert / all hint-following).

## codemmlu (pooled over 2 hint types)

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| codemmlu | 0.3108 (0.3774) | 0.8204 (0.3598) | 0.5854 (0.3249) | 0.9153 (0.2142) | 213 (53.2%) | 0.8956 | 0.756 (0.382) | 0.525 (0.435) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1047 (33.1%) | 40 (1.3%) |
| **Not acknowledged** | 230 (7.3%) | 1846 (58.4%) |

**Opacity rate:** 0.1801 — 230 covert / 1277 hint-following runs (covert / all hint-following).

## lsat_ar (pooled over 2 hint types)

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| lsat_ar | 0.2796 (0.3290) | 0.8803 (0.3603) | 0.4775 (0.2915) | 0.9284 (0.1798) | 237 (59.2%) | 0.8406 | 0.814 (0.320) | 0.610 (0.381) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 838 (26.6%) | 36 (1.1%) |
| **Not acknowledged** | 191 (6.1%) | 2080 (66.1%) |

**Opacity rate:** 0.1856 — 191 covert / 1029 hint-following runs (covert / all hint-following).

## lsat_lr (pooled over 2 hint types)

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| lsat_lr | 0.1954 (0.3257) | 0.8686 (0.2841) | 0.5159 (0.3368) | 0.9029 (0.2297) | 152 (38.0%) | 0.6949 | 0.847 (0.308) | 0.679 (0.399) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 425 (13.3%) | 9 (0.3%) |
| **Not acknowledged** | 333 (10.4%) | 2427 (76.0%) |

**Opacity rate:** 0.4393 — 333 covert / 758 hint-following runs (covert / all hint-following).

## medxpertqa (pooled over 2 hint types)

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| medxpertqa | 0.5663 (0.4239) | 0.8922 (0.2598) | 0.7735 (0.2906) | 0.9348 (0.1753) | 293 (73.2%) | 0.8543 | 0.135 (0.295) | 0.052 (0.189) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1654 (51.7%) | 3 (0.1%) |
| **Not acknowledged** | 418 (13.1%) | 1124 (35.1%) |

**Opacity rate:** 0.2017 — 418 covert / 2072 hint-following runs (covert / all hint-following).

## lexam (pooled over 2 hint types)

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| lexam | 0.4269 (0.3952) | 0.7797 (0.3958) | 0.6117 (0.3233) | 0.8735 (0.2567) | 282 (70.5%) | 0.9331 | 0.567 (0.412) | 0.296 (0.380) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1568 (49.0%) | 38 (1.2%) |
| **Not acknowledged** | 239 (7.5%) | 1352 (42.3%) |

**Opacity rate:** 0.1323 — 239 covert / 1807 hint-following runs (covert / all hint-following).

## Summary

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) | Acknowledged & followed | Acknowledged & not followed | Not Acknowledged & followed | Not Acknowledged & not followed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mmlu_pro | 0.1705 (0.3065) | 0.8707 (0.2808) | 0.5501 (0.3071) | 0.9268 (0.1764) | 62 (31.0%) | 0.8045 | 0.735 (0.402) | 0.638 (0.437) | 261 (16.3%) | 8 (0.5%) | 74 (4.6%) | 1255 (78.5%) |
| gpqa_diamond | 0.4736 (0.4071) | 0.7713 (0.3765) | 0.6648 (0.3176) | 0.8527 (0.2481) | 284 (71.7%) | 0.8513 | 0.507 (0.423) | 0.274 (0.383) | 1519 (48.6%) | 9 (0.3%) | 449 (14.4%) | 1148 (36.7%) |
| wmdp | 0.4315 (0.4207) | 0.8289 (0.3401) | 0.6921 (0.3213) | 0.9203 (0.2100) | 245 (62.3%) | 0.9298 | 0.673 (0.421) | 0.373 (0.427) | 1404 (46.3%) | 13 (0.4%) | 241 (8.0%) | 1372 (45.3%) |
| codemmlu | 0.3108 (0.3774) | 0.8204 (0.3598) | 0.5854 (0.3249) | 0.9153 (0.2142) | 213 (53.2%) | 0.8956 | 0.756 (0.382) | 0.525 (0.435) | 1047 (33.1%) | 40 (1.3%) | 230 (7.3%) | 1846 (58.4%) |
| lsat_ar | 0.2796 (0.3290) | 0.8803 (0.3603) | 0.4775 (0.2915) | 0.9284 (0.1798) | 237 (59.2%) | 0.8406 | 0.814 (0.320) | 0.610 (0.381) | 838 (26.6%) | 36 (1.1%) | 191 (6.1%) | 2080 (66.1%) |
| lsat_lr | 0.1954 (0.3257) | 0.8686 (0.2841) | 0.5159 (0.3368) | 0.9029 (0.2297) | 152 (38.0%) | 0.6949 | 0.847 (0.308) | 0.679 (0.399) | 425 (13.3%) | 9 (0.3%) | 333 (10.4%) | 2427 (76.0%) |
| medxpertqa | 0.5663 (0.4239) | 0.8922 (0.2598) | 0.7735 (0.2906) | 0.9348 (0.1753) | 293 (73.2%) | 0.8543 | 0.135 (0.295) | 0.052 (0.189) | 1654 (51.7%) | 3 (0.1%) | 418 (13.1%) | 1124 (35.1%) |
| lexam | 0.4269 (0.3952) | 0.7797 (0.3958) | 0.6117 (0.3233) | 0.8735 (0.2567) | 282 (70.5%) | 0.9331 | 0.567 (0.412) | 0.296 (0.380) | 1568 (49.0%) | 38 (1.2%) | 239 (7.5%) | 1352 (42.3%) |
