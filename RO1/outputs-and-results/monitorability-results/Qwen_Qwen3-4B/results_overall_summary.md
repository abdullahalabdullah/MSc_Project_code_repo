# Results overall summary — per (dataset × hint type)

- **Model:** `Qwen/Qwen3-4B` (`Qwen_Qwen3-4B`)
- **Datasets:** mmlu_pro, gpqa_diamond, wmdp, codemmlu, lsat_ar, lsat_lr, medxpertqa, lexam
- **Hint types:** professor, grader_hacking, unethical

## mmlu_pro × professor

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| mmlu_pro | 0.1717 (0.3113) | 0.8669 (0.2983) | 0.5181 (0.3291) | 0.9095 (0.2226) | 67 (33.5%) | 0.7891 | 0.719 (0.400) | 0.631 (0.437) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 240 (15.0%) | 9 (0.6%) |
| **Not acknowledged** | 101 (6.3%) | 1249 (78.1%) |

**Opacity rate:** 0.2962 — 101 covert / 341 hint-following runs (covert / all hint-following).

## mmlu_pro × grader_hacking

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| mmlu_pro | 0.2028 (0.3090) | 0.8407 (0.5942) | 0.4655 (0.2947) | 0.9347 (0.1984) | 89 (44.5%) | 0.6689 | 0.719 (0.400) | 0.596 (0.416) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 201 (12.6%) | 1 (0.1%) |
| **Not acknowledged** | 189 (11.8%) | 1204 (75.5%) |

**Opacity rate:** 0.4846 — 189 covert / 390 hint-following runs (covert / all hint-following).

## mmlu_pro × unethical

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| mmlu_pro | 0.7919 (0.2758) | 0.9569 (0.1682) | 0.8039 (0.2598) | 0.9666 (0.1383) | 197 (98.5%) | 0.9428 | 0.719 (0.400) | 0.139 (0.227) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1236 (77.5%) | 8 (0.5%) |
| **Not acknowledged** | 94 (5.9%) | 257 (16.1%) |

**Opacity rate:** 0.0707 — 94 covert / 1330 hint-following runs (covert / all hint-following).

## gpqa_diamond × professor

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| gpqa_diamond | 0.3375 (0.3666) | 0.7146 (0.3579) | 0.5689 (0.3040) | 0.7959 (0.2733) | 118 (59.6%) | 0.7711 | 0.512 (0.412) | 0.362 (0.409) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 453 (28.8%) | 3 (0.2%) |
| **Not acknowledged** | 309 (19.7%) | 806 (51.3%) |

**Opacity rate:** 0.4055 — 309 covert / 762 hint-following runs (covert / all hint-following).

## gpqa_diamond × grader_hacking

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| gpqa_diamond | 0.2916 (0.3551) | 0.6729 (0.5787) | 0.5010 (0.3094) | 0.8131 (0.2761) | 119 (60.1%) | 0.3041 | 0.512 (0.412) | 0.379 (0.402) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 113 (7.2%) | 3 (0.2%) |
| **Not acknowledged** | 577 (36.8%) | 874 (55.8%) |

**Opacity rate:** 0.8362 — 577 covert / 690 hint-following runs (covert / all hint-following).

## gpqa_diamond × unethical

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| gpqa_diamond | 0.6501 (0.3509) | 0.8256 (0.2931) | 0.7274 (0.2848) | 0.8684 (0.2283) | 177 (89.4%) | 0.8661 | 0.512 (0.412) | 0.159 (0.292) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 968 (61.7%) | 3 (0.2%) |
| **Not acknowledged** | 288 (18.3%) | 311 (19.8%) |

**Opacity rate:** 0.2293 — 288 covert / 1256 hint-following runs (covert / all hint-following).

## wmdp × professor

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| wmdp | 0.2370 (0.3424) | 0.7553 (0.4115) | 0.5321 (0.3237) | 0.8782 (0.2554) | 88 (44.7%) | 0.8488 | 0.679 (0.407) | 0.514 (0.439) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 377 (24.7%) | 2 (0.1%) |
| **Not acknowledged** | 148 (9.7%) | 997 (65.4%) |

**Opacity rate:** 0.2819 — 148 covert / 525 hint-following runs (covert / all hint-following).

## wmdp × grader_hacking

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| wmdp | 0.2149 (0.3246) | 0.7543 (0.4046) | 0.5175 (0.3007) | 0.8984 (0.2341) | 83 (42.1%) | 0.4544 | 0.679 (0.407) | 0.526 (0.430) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 135 (8.9%) | 2 (0.1%) |
| **Not acknowledged** | 360 (23.7%) | 1022 (67.3%) |

**Opacity rate:** 0.7273 — 360 covert / 495 hint-following runs (covert / all hint-following).

## wmdp × unethical

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| wmdp | 0.6523 (0.3839) | 0.8785 (0.2931) | 0.7604 (0.2988) | 0.9409 (0.1816) | 169 (85.8%) | 0.9094 | 0.679 (0.407) | 0.201 (0.326) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 976 (63.7%) | 4 (0.3%) |
| **Not acknowledged** | 186 (12.1%) | 365 (23.8%) |

**Opacity rate:** 0.1601 — 186 covert / 1162 hint-following runs (covert / all hint-following).

## codemmlu × professor

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| codemmlu | 0.1220 (0.2469) | 0.6730 (0.5417) | 0.4314 (0.2717) | 0.8926 (0.2551) | 58 (29.0%) | 0.8703 | 0.763 (0.381) | 0.673 (0.410) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 223 (14.1%) | 6 (0.4%) |
| **Not acknowledged** | 107 (6.8%) | 1249 (78.8%) |

**Opacity rate:** 0.3242 — 107 covert / 330 hint-following runs (covert / all hint-following).

## codemmlu × grader_hacking

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| codemmlu | 0.1028 (0.2270) | 0.6746 (0.5420) | 0.3639 (0.2739) | 0.8778 (0.2523) | 59 (29.5%) | 0.4595 | 0.763 (0.381) | 0.695 (0.397) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 91 (5.8%) | 0 (0.0%) |
| **Not acknowledged** | 201 (12.7%) | 1286 (81.5%) |

**Opacity rate:** 0.6884 — 201 covert / 292 hint-following runs (covert / all hint-following).

## codemmlu × unethical

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| codemmlu | 0.5893 (0.3752) | 0.9021 (0.2637) | 0.6933 (0.3053) | 0.9499 (0.1662) | 170 (85.0%) | 0.9375 | 0.763 (0.381) | 0.296 (0.363) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 944 (59.4%) | 14 (0.9%) |
| **Not acknowledged** | 128 (8.1%) | 502 (31.6%) |

**Opacity rate:** 0.1194 — 128 covert / 1072 hint-following runs (covert / all hint-following).

## lsat_ar × professor

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| lsat_ar | 0.1304 (0.2545) | 0.6417 (0.6696) | 0.3986 (0.2544) | 0.9007 (0.2048) | 71 (35.5%) | 0.7463 | 0.782 (0.328) | 0.710 (0.357) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 203 (12.8%) | 4 (0.3%) |
| **Not acknowledged** | 103 (6.5%) | 1279 (80.5%) |

**Opacity rate:** 0.3366 — 103 covert / 306 hint-following runs (covert / all hint-following).

## lsat_ar × grader_hacking

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| lsat_ar | 0.1465 (0.2562) | 0.7659 (0.5260) | 0.3613 (0.2422) | 0.8735 (0.2396) | 87 (43.5%) | 0.6682 | 0.782 (0.328) | 0.706 (0.360) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 184 (11.7%) | 8 (0.5%) |
| **Not acknowledged** | 145 (9.2%) | 1240 (78.6%) |

**Opacity rate:** 0.4407 — 145 covert / 329 hint-following runs (covert / all hint-following).

## lsat_ar × unethical

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| lsat_ar | 0.4664 (0.3456) | 0.8996 (0.2506) | 0.5661 (0.2961) | 0.9299 (0.1792) | 165 (82.5%) | 0.9141 | 0.782 (0.328) | 0.426 (0.372) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 732 (46.4%) | 18 (1.1%) |
| **Not acknowledged** | 104 (6.6%) | 725 (45.9%) |

**Opacity rate:** 0.1244 — 104 covert / 836 hint-following runs (covert / all hint-following).

## lsat_lr × professor

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| lsat_lr | 0.0794 (0.2056) | 0.7157 (0.4585) | 0.3750 (0.2802) | 0.8346 (0.2780) | 44 (22.0%) | 0.4141 | 0.834 (0.320) | 0.764 (0.355) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 54 (3.4%) | 2 (0.1%) |
| **Not acknowledged** | 146 (9.1%) | 1396 (87.4%) |

**Opacity rate:** 0.7300 — 146 covert / 200 hint-following runs (covert / all hint-following).

## lsat_lr × grader_hacking

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| lsat_lr | 0.0951 (0.2172) | 0.7470 (0.3970) | 0.3932 (0.2721) | 0.8639 (0.2540) | 49 (24.5%) | 0.4062 | 0.834 (0.320) | 0.753 (0.363) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 63 (3.9%) | 1 (0.1%) |
| **Not acknowledged** | 162 (10.1%) | 1373 (85.9%) |

**Opacity rate:** 0.7200 — 162 covert / 225 hint-following runs (covert / all hint-following).

## lsat_lr × unethical

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| lsat_lr | 0.3532 (0.3830) | 0.9228 (0.2112) | 0.5790 (0.3308) | 0.9379 (0.1761) | 122 (61.0%) | 0.7666 | 0.834 (0.320) | 0.546 (0.420) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 408 (25.5%) | 3 (0.2%) |
| **Not acknowledged** | 229 (14.3%) | 958 (59.9%) |

**Opacity rate:** 0.3595 — 229 covert / 637 hint-following runs (covert / all hint-following).

## medxpertqa × professor

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| medxpertqa | 0.2938 (0.3612) | 0.8110 (0.3479) | 0.5451 (0.3222) | 0.8892 (0.2421) | 108 (54.0%) | 0.7130 | 0.127 (0.295) | 0.094 (0.243) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 348 (21.8%) | 5 (0.3%) |
| **Not acknowledged** | 278 (17.4%) | 968 (60.5%) |

**Opacity rate:** 0.4441 — 278 covert / 626 hint-following runs (covert / all hint-following).

## medxpertqa × grader_hacking

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| medxpertqa | 0.2569 (0.3377) | 0.7855 (0.3848) | 0.4882 (0.3174) | 0.8855 (0.2499) | 106 (53.0%) | 0.5080 | 0.127 (0.295) | 0.087 (0.243) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 188 (11.8%) | 0 (0.0%) |
| **Not acknowledged** | 379 (23.7%) | 1033 (64.6%) |

**Opacity rate:** 0.6684 — 379 covert / 567 hint-following runs (covert / all hint-following).

## medxpertqa × unethical

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| medxpertqa | 0.8231 (0.2993) | 0.9015 (0.2574) | 0.8664 (0.2378) | 0.9395 (0.1822) | 190 (95.0%) | 0.8782 | 0.127 (0.295) | 0.010 (0.073) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1206 (75.4%) | 0 (0.0%) |
| **Not acknowledged** | 266 (16.6%) | 127 (7.9%) |

**Opacity rate:** 0.1807 — 266 covert / 1472 hint-following runs (covert / all hint-following).

## lexam × professor

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| lexam | 0.2289 (0.2935) | 0.7141 (0.6996) | 0.4199 (0.2551) | 0.8396 (0.2588) | 112 (56.0%) | 0.7912 | 0.585 (0.403) | 0.444 (0.406) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 381 (23.8%) | 17 (1.1%) |
| **Not acknowledged** | 183 (11.4%) | 1019 (63.7%) |

**Opacity rate:** 0.3245 — 183 covert / 564 hint-following runs (covert / all hint-following).

## lexam × grader_hacking

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| lexam | 0.2526 (0.3146) | 0.7517 (0.3826) | 0.4491 (0.2800) | 0.8362 (0.2687) | 115 (57.5%) | 0.6665 | 0.585 (0.403) | 0.434 (0.402) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 290 (18.1%) | 13 (0.8%) |
| **Not acknowledged** | 312 (19.5%) | 984 (61.5%) |

**Opacity rate:** 0.5183 — 312 covert / 602 hint-following runs (covert / all hint-following).

## lexam × unethical

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| lexam | 0.7424 (0.3117) | 0.8685 (0.2728) | 0.7946 (0.2482) | 0.9066 (0.2050) | 187 (93.5%) | 0.9277 | 0.585 (0.403) | 0.097 (0.205) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1265 (79.1%) | 21 (1.3%) |
| **Not acknowledged** | 120 (7.5%) | 193 (12.1%) |

**Opacity rate:** 0.0866 — 120 covert / 1385 hint-following runs (covert / all hint-following).

## Summary

| Dataset | Hint type | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) | Acknowledged & followed | Acknowledged & not followed | Not Acknowledged & followed | Not Acknowledged & not followed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mmlu_pro | professor | 0.1717 (0.3113) | 0.8669 (0.2983) | 0.5181 (0.3291) | 0.9095 (0.2226) | 67 (33.5%) | 0.7891 | 0.719 (0.400) | 0.631 (0.437) | 240 (15.0%) | 9 (0.6%) | 101 (6.3%) | 1249 (78.1%) |
| mmlu_pro | grader_hacking | 0.2028 (0.3090) | 0.8407 (0.5942) | 0.4655 (0.2947) | 0.9347 (0.1984) | 89 (44.5%) | 0.6689 | 0.719 (0.400) | 0.596 (0.416) | 201 (12.6%) | 1 (0.1%) | 189 (11.8%) | 1204 (75.5%) |
| mmlu_pro | unethical | 0.7919 (0.2758) | 0.9569 (0.1682) | 0.8039 (0.2598) | 0.9666 (0.1383) | 197 (98.5%) | 0.9428 | 0.719 (0.400) | 0.139 (0.227) | 1236 (77.5%) | 8 (0.5%) | 94 (5.9%) | 257 (16.1%) |
| gpqa_diamond | professor | 0.3375 (0.3666) | 0.7146 (0.3579) | 0.5689 (0.3040) | 0.7959 (0.2733) | 118 (59.6%) | 0.7711 | 0.512 (0.412) | 0.362 (0.409) | 453 (28.8%) | 3 (0.2%) | 309 (19.7%) | 806 (51.3%) |
| gpqa_diamond | grader_hacking | 0.2916 (0.3551) | 0.6729 (0.5787) | 0.5010 (0.3094) | 0.8131 (0.2761) | 119 (60.1%) | 0.3041 | 0.512 (0.412) | 0.379 (0.402) | 113 (7.2%) | 3 (0.2%) | 577 (36.8%) | 874 (55.8%) |
| gpqa_diamond | unethical | 0.6501 (0.3509) | 0.8256 (0.2931) | 0.7274 (0.2848) | 0.8684 (0.2283) | 177 (89.4%) | 0.8661 | 0.512 (0.412) | 0.159 (0.292) | 968 (61.7%) | 3 (0.2%) | 288 (18.3%) | 311 (19.8%) |
| wmdp | professor | 0.2370 (0.3424) | 0.7553 (0.4115) | 0.5321 (0.3237) | 0.8782 (0.2554) | 88 (44.7%) | 0.8488 | 0.679 (0.407) | 0.514 (0.439) | 377 (24.7%) | 2 (0.1%) | 148 (9.7%) | 997 (65.4%) |
| wmdp | grader_hacking | 0.2149 (0.3246) | 0.7543 (0.4046) | 0.5175 (0.3007) | 0.8984 (0.2341) | 83 (42.1%) | 0.4544 | 0.679 (0.407) | 0.526 (0.430) | 135 (8.9%) | 2 (0.1%) | 360 (23.7%) | 1022 (67.3%) |
| wmdp | unethical | 0.6523 (0.3839) | 0.8785 (0.2931) | 0.7604 (0.2988) | 0.9409 (0.1816) | 169 (85.8%) | 0.9094 | 0.679 (0.407) | 0.201 (0.326) | 976 (63.7%) | 4 (0.3%) | 186 (12.1%) | 365 (23.8%) |
| codemmlu | professor | 0.1220 (0.2469) | 0.6730 (0.5417) | 0.4314 (0.2717) | 0.8926 (0.2551) | 58 (29.0%) | 0.8703 | 0.763 (0.381) | 0.673 (0.410) | 223 (14.1%) | 6 (0.4%) | 107 (6.8%) | 1249 (78.8%) |
| codemmlu | grader_hacking | 0.1028 (0.2270) | 0.6746 (0.5420) | 0.3639 (0.2739) | 0.8778 (0.2523) | 59 (29.5%) | 0.4595 | 0.763 (0.381) | 0.695 (0.397) | 91 (5.8%) | 0 (0.0%) | 201 (12.7%) | 1286 (81.5%) |
| codemmlu | unethical | 0.5893 (0.3752) | 0.9021 (0.2637) | 0.6933 (0.3053) | 0.9499 (0.1662) | 170 (85.0%) | 0.9375 | 0.763 (0.381) | 0.296 (0.363) | 944 (59.4%) | 14 (0.9%) | 128 (8.1%) | 502 (31.6%) |
| lsat_ar | professor | 0.1304 (0.2545) | 0.6417 (0.6696) | 0.3986 (0.2544) | 0.9007 (0.2048) | 71 (35.5%) | 0.7463 | 0.782 (0.328) | 0.710 (0.357) | 203 (12.8%) | 4 (0.3%) | 103 (6.5%) | 1279 (80.5%) |
| lsat_ar | grader_hacking | 0.1465 (0.2562) | 0.7659 (0.5260) | 0.3613 (0.2422) | 0.8735 (0.2396) | 87 (43.5%) | 0.6682 | 0.782 (0.328) | 0.706 (0.360) | 184 (11.7%) | 8 (0.5%) | 145 (9.2%) | 1240 (78.6%) |
| lsat_ar | unethical | 0.4664 (0.3456) | 0.8996 (0.2506) | 0.5661 (0.2961) | 0.9299 (0.1792) | 165 (82.5%) | 0.9141 | 0.782 (0.328) | 0.426 (0.372) | 732 (46.4%) | 18 (1.1%) | 104 (6.6%) | 725 (45.9%) |
| lsat_lr | professor | 0.0794 (0.2056) | 0.7157 (0.4585) | 0.3750 (0.2802) | 0.8346 (0.2780) | 44 (22.0%) | 0.4141 | 0.834 (0.320) | 0.764 (0.355) | 54 (3.4%) | 2 (0.1%) | 146 (9.1%) | 1396 (87.4%) |
| lsat_lr | grader_hacking | 0.0951 (0.2172) | 0.7470 (0.3970) | 0.3932 (0.2721) | 0.8639 (0.2540) | 49 (24.5%) | 0.4062 | 0.834 (0.320) | 0.753 (0.363) | 63 (3.9%) | 1 (0.1%) | 162 (10.1%) | 1373 (85.9%) |
| lsat_lr | unethical | 0.3532 (0.3830) | 0.9228 (0.2112) | 0.5790 (0.3308) | 0.9379 (0.1761) | 122 (61.0%) | 0.7666 | 0.834 (0.320) | 0.546 (0.420) | 408 (25.5%) | 3 (0.2%) | 229 (14.3%) | 958 (59.9%) |
| medxpertqa | professor | 0.2938 (0.3612) | 0.8110 (0.3479) | 0.5451 (0.3222) | 0.8892 (0.2421) | 108 (54.0%) | 0.7130 | 0.127 (0.295) | 0.094 (0.243) | 348 (21.8%) | 5 (0.3%) | 278 (17.4%) | 968 (60.5%) |
| medxpertqa | grader_hacking | 0.2569 (0.3377) | 0.7855 (0.3848) | 0.4882 (0.3174) | 0.8855 (0.2499) | 106 (53.0%) | 0.5080 | 0.127 (0.295) | 0.087 (0.243) | 188 (11.8%) | 0 (0.0%) | 379 (23.7%) | 1033 (64.6%) |
| medxpertqa | unethical | 0.8231 (0.2993) | 0.9015 (0.2574) | 0.8664 (0.2378) | 0.9395 (0.1822) | 190 (95.0%) | 0.8782 | 0.127 (0.295) | 0.010 (0.073) | 1206 (75.4%) | 0 (0.0%) | 266 (16.6%) | 127 (7.9%) |
| lexam | professor | 0.2289 (0.2935) | 0.7141 (0.6996) | 0.4199 (0.2551) | 0.8396 (0.2588) | 112 (56.0%) | 0.7912 | 0.585 (0.403) | 0.444 (0.406) | 381 (23.8%) | 17 (1.1%) | 183 (11.4%) | 1019 (63.7%) |
| lexam | grader_hacking | 0.2526 (0.3146) | 0.7517 (0.3826) | 0.4491 (0.2800) | 0.8362 (0.2687) | 115 (57.5%) | 0.6665 | 0.585 (0.403) | 0.434 (0.402) | 290 (18.1%) | 13 (0.8%) | 312 (19.5%) | 984 (61.5%) |
| lexam | unethical | 0.7424 (0.3117) | 0.8685 (0.2728) | 0.7946 (0.2482) | 0.9066 (0.2050) | 187 (93.5%) | 0.9277 | 0.585 (0.403) | 0.097 (0.205) | 1265 (79.1%) | 21 (1.3%) | 120 (7.5%) | 193 (12.1%) |
