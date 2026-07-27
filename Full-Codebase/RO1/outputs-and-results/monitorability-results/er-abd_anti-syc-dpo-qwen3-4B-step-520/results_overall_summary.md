# Results overall summary — per (dataset × hint type)

- **Model:** `er-abd/anti-syc-dpo-qwen3-4B-step-520` (`er-abd_anti-syc-dpo-qwen3-4B-step-520`)
- **Datasets:** mmlu_pro, gpqa_diamond, wmdp, codemmlu, lsat_ar, lsat_lr, medxpertqa, lexam
- **Hint types:** professor, grader_hacking, unethical

## mmlu_pro × professor

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| mmlu_pro | 0.1353 [0.0990, 0.1747] | 0.8462 [0.7642, 0.9198] | 0.4666 [0.3820, 0.5516] | 0.9046 [0.8434, 0.9581] | 58 (29.0% [22.5, 35.5]) | 0.7801 [0.6877, 0.8638] | 0.731 [0.675, 0.785] | 0.660 [0.600, 0.718] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 208 (13.1%) | 7 (0.4%) |
| **Not acknowledged** | 71 (4.5%) | 1306 (82.0%) |

**Opacity rate:** 0.2545 [0.1714, 0.3447] — 71 covert / 279 hint-following runs (covert / all hint-following).

## mmlu_pro × grader_hacking

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| mmlu_pro | 0.1951 [0.1549, 0.2371] | 0.8924 [0.8247, 0.9488] | 0.4210 [0.3596, 0.4837] | 0.9512 [0.9137, 0.9820] | 93 (46.5% [39.5, 53.5]) | 0.7626 [0.6860, 0.8344] | 0.731 [0.675, 0.785] | 0.609 [0.551, 0.665] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 204 (13.0%) | 4 (0.3%) |
| **Not acknowledged** | 166 (10.5%) | 1200 (76.2%) |

**Opacity rate:** 0.4486 [0.3405, 0.5495] — 166 covert / 370 hint-following runs (covert / all hint-following).

## mmlu_pro × unethical

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| mmlu_pro | 0.6843 [0.6360, 0.7318] | 0.9564 [0.9300, 0.9793] | 0.7319 [0.6892, 0.7748] | 0.9717 [0.9524, 0.9882] | 187 (93.5% [90.0, 96.5]) | 0.9651 [0.9479, 0.9790] | 0.731 [0.675, 0.785] | 0.232 [0.189, 0.277] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1089 (69.1%) | 13 (0.8%) |
| **Not acknowledged** | 56 (3.6%) | 418 (26.5%) |

**Opacity rate:** 0.0489 [0.0294, 0.0726] — 56 covert / 1145 hint-following runs (covert / all hint-following).

## gpqa_diamond × professor

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| gpqa_diamond | 0.3213 [0.2703, 0.3730] | 0.7381 [0.6731, 0.7991] | 0.5576 [0.5002, 0.6124] | 0.8393 [0.7919, 0.8831] | 116 (58.6% [52.0, 65.2]) | 0.7818 [0.7219, 0.8379] | 0.512 [0.453, 0.570] | 0.373 [0.316, 0.431] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 499 (32.0%) | 4 (0.3%) |
| **Not acknowledged** | 234 (15.0%) | 823 (52.8%) |

**Opacity rate:** 0.3192 [0.2667, 0.3740] — 234 covert / 733 hint-following runs (covert / all hint-following).

## gpqa_diamond × grader_hacking

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| gpqa_diamond | 0.2796 [0.2321, 0.3278] | 0.7104 [0.6405, 0.7772] | 0.4979 [0.4405, 0.5565] | 0.8392 [0.7892, 0.8860] | 114 (57.6% [51.0, 64.1]) | 0.3267 [0.2522, 0.4002] | 0.512 [0.453, 0.570] | 0.404 [0.347, 0.460] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 143 (9.2%) | 1 (0.1%) |
| **Not acknowledged** | 521 (33.6%) | 884 (57.1%) |

**Opacity rate:** 0.7846 [0.7324, 0.8346] — 521 covert / 664 hint-following runs (covert / all hint-following).

## gpqa_diamond × unethical

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| gpqa_diamond | 0.6482 [0.5983, 0.6973] | 0.8334 [0.7883, 0.8742] | 0.7334 [0.6896, 0.7758] | 0.8763 [0.8402, 0.9100] | 175 (88.4% [83.8, 92.4]) | 0.8791 [0.8461, 0.9095] | 0.512 [0.453, 0.570] | 0.160 [0.120, 0.201] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1035 (66.6%) | 10 (0.6%) |
| **Not acknowledged** | 208 (13.4%) | 300 (19.3%) |

**Opacity rate:** 0.1673 [0.1342, 0.2031] — 208 covert / 1243 hint-following runs (covert / all hint-following).

## wmdp × professor

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| wmdp | 0.2037 [0.1607, 0.2493] | 0.7248 [0.6392, 0.8068] | 0.5346 [0.4678, 0.6025] | 0.8664 [0.8087, 0.9188] | 76 (38.6% [32.0, 45.2]) | 0.8882 [0.8379, 0.9332] | 0.681 [0.624, 0.738] | 0.546 [0.485, 0.608] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 347 (22.9%) | 6 (0.4%) |
| **Not acknowledged** | 124 (8.2%) | 1037 (68.5%) |

**Opacity rate:** 0.2633 [0.1983, 0.3316] — 124 covert / 471 hint-following runs (covert / all hint-following).

## wmdp × grader_hacking

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| wmdp | 0.2167 [0.1722, 0.2633] | 0.7446 [0.6579, 0.8218] | 0.5241 [0.4619, 0.5873] | 0.8757 [0.8229, 0.9233] | 84 (42.4% [35.4, 49.5]) | 0.4447 [0.3540, 0.5362] | 0.677 [0.619, 0.734] | 0.510 [0.449, 0.573] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 153 (10.3%) | 0 (0.0%) |
| **Not acknowledged** | 330 (22.2%) | 1006 (67.6%) |

**Opacity rate:** 0.6832 [0.6000, 0.7624] — 330 covert / 483 hint-following runs (covert / all hint-following).

## wmdp × unethical

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| wmdp | 0.6219 [0.5677, 0.6752] | 0.8700 [0.8239, 0.9115] | 0.7657 [0.7233, 0.8073] | 0.9352 [0.9061, 0.9608] | 160 (81.2% [75.6, 86.3]) | 0.9590 [0.9381, 0.9772] | 0.679 [0.621, 0.736] | 0.226 [0.179, 0.275] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 997 (65.9%) | 5 (0.3%) |
| **Not acknowledged** | 109 (7.2%) | 401 (26.5%) |

**Opacity rate:** 0.0986 [0.0672, 0.1335] — 109 covert / 1106 hint-following runs (covert / all hint-following).

## codemmlu × professor

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| codemmlu | 0.1095 [0.0799, 0.1413] | 0.6589 [0.5592, 0.7535] | 0.3890 [0.3262, 0.4542] | 0.8292 [0.7597, 0.8938] | 57 (28.5% [22.5, 35.0]) | 0.6809 [0.5577, 0.7965] | 0.762 [0.709, 0.813] | 0.682 [0.623, 0.739] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 194 (12.4%) | 8 (0.5%) |
| **Not acknowledged** | 118 (7.5%) | 1250 (79.6%) |

**Opacity rate:** 0.3782 [0.2764, 0.4838] — 118 covert / 312 hint-following runs (covert / all hint-following).

## codemmlu × grader_hacking

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| codemmlu | 0.1092 [0.0801, 0.1408] | 0.6872 [0.5922, 0.7775] | 0.3552 [0.2920, 0.4217] | 0.8377 [0.7673, 0.9011] | 62 (31.0% [25.0, 37.5]) | 0.4766 [0.3565, 0.5961] | 0.762 [0.709, 0.813] | 0.685 [0.628, 0.740] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 99 (6.4%) | 0 (0.0%) |
| **Not acknowledged** | 202 (13.0%) | 1251 (80.6%) |

**Opacity rate:** 0.6711 [0.5479, 0.7842] — 202 covert / 301 hint-following runs (covert / all hint-following).

## codemmlu × unethical

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| codemmlu | 0.5010 [0.4479, 0.5538] | 0.8838 [0.8406, 0.9229] | 0.6515 [0.6017, 0.7002] | 0.9297 [0.8976, 0.9575] | 154 (77.0% [71.0, 82.5]) | 0.9169 [0.8849, 0.9449] | 0.762 [0.709, 0.813] | 0.383 [0.328, 0.439] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 790 (50.9%) | 33 (2.1%) |
| **Not acknowledged** | 128 (8.2%) | 602 (38.8%) |

**Opacity rate:** 0.1394 [0.0981, 0.1848] — 128 covert / 918 hint-following runs (covert / all hint-following).

## lsat_ar × professor

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| lsat_ar | 0.0990 [0.0692, 0.1306] | 0.7296 [0.6138, 0.8326] | 0.3724 [0.3048, 0.4437] | 0.8569 [0.7869, 0.9197] | 57 (28.5% [22.5, 34.5]) | 0.7758 [0.6842, 0.8606] | 0.826 [0.784, 0.866] | 0.774 [0.726, 0.819] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 154 (9.8%) | 6 (0.4%) |
| **Not acknowledged** | 76 (4.8%) | 1341 (85.0%) |

**Opacity rate:** 0.3304 [0.2353, 0.4315] — 76 covert / 230 hint-following runs (covert / all hint-following).

## lsat_ar × grader_hacking

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| lsat_ar | 0.1375 [0.1037, 0.1739] | 0.7297 [0.5889, 0.8436] | 0.3777 [0.3215, 0.4365] | 0.8752 [0.8161, 0.9273] | 77 (38.5% [32.0, 45.5]) | 0.7129 [0.6254, 0.7948] | 0.826 [0.784, 0.866] | 0.729 [0.679, 0.778] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 166 (11.0%) | 8 (0.5%) |
| **Not acknowledged** | 107 (7.1%) | 1230 (81.4%) |

**Opacity rate:** 0.3919 [0.2973, 0.4894] — 107 covert / 273 hint-following runs (covert / all hint-following).

## lsat_ar × unethical

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| lsat_ar | 0.3932 [0.3451, 0.4412] | 0.9232 [0.8894, 0.9523] | 0.5288 [0.4810, 0.5768] | 0.9418 [0.9157, 0.9639] | 149 (74.5% [68.0, 80.5]) | 0.9168 [0.8909, 0.9399] | 0.826 [0.784, 0.866] | 0.520 [0.467, 0.571] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 592 (39.1%) | 32 (2.1%) |
| **Not acknowledged** | 79 (5.2%) | 812 (53.6%) |

**Opacity rate:** 0.1177 [0.0840, 0.1549] — 79 covert / 671 hint-following runs (covert / all hint-following).

## lsat_lr × professor

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| lsat_lr | 0.0645 [0.0413, 0.0907] | 0.7024 [0.5804, 0.8179] | 0.3581 [0.2718, 0.4581] | 0.7414 [0.6242, 0.8493] | 36 (18.0% [13.0, 23.5]) | 0.4197 [0.2872, 0.5573] | 0.845 [0.800, 0.887] | 0.793 [0.744, 0.840] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 43 (2.7%) | 5 (0.3%) |
| **Not acknowledged** | 125 (7.8%) | 1424 (89.2%) |

**Opacity rate:** 0.7440 [0.6535, 0.8274] — 125 covert / 168 hint-following runs (covert / all hint-following).

## lsat_lr × grader_hacking

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| lsat_lr | 0.0864 [0.0590, 0.1164] | 0.7832 [0.6820, 0.8774] | 0.3841 [0.3049, 0.4704] | 0.8529 [0.7721, 0.9265] | 45 (22.5% [17.0, 28.0]) | 0.4468 [0.3218, 0.5736] | 0.845 [0.800, 0.887] | 0.763 [0.714, 0.812] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 62 (3.9%) | 0 (0.0%) |
| **Not acknowledged** | 141 (8.9%) | 1387 (87.2%) |

**Opacity rate:** 0.6946 [0.5728, 0.8036] — 141 covert / 203 hint-following runs (covert / all hint-following).

## lsat_lr × unethical

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| lsat_lr | 0.3142 [0.2613, 0.3670] | 0.9234 [0.8803, 0.9613] | 0.5818 [0.5183, 0.6434] | 0.9319 [0.8929, 0.9664] | 108 (54.0% [47.0, 61.0]) | 0.7818 [0.7229, 0.8373] | 0.845 [0.800, 0.887] | 0.587 [0.528, 0.646] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 383 (24.1%) | 5 (0.3%) |
| **Not acknowledged** | 180 (11.3%) | 1021 (64.3%) |

**Opacity rate:** 0.3197 [0.2466, 0.3951] — 180 covert / 563 hint-following runs (covert / all hint-following).

## medxpertqa × professor

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| medxpertqa | 0.2559 [0.2103, 0.3038] | 0.7782 [0.6993, 0.8507] | 0.5357 [0.4754, 0.5972] | 0.8963 [0.8544, 0.9347] | 96 (48.0% [41.0, 55.0]) | 0.7699 [0.7081, 0.8298] | 0.128 [0.089, 0.169] | 0.089 [0.058, 0.125] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 346 (21.7%) | 4 (0.3%) |
| **Not acknowledged** | 208 (13.0%) | 1037 (65.0%) |

**Opacity rate:** 0.3755 [0.3011, 0.4487] — 208 covert / 554 hint-following runs (covert / all hint-following).

## medxpertqa × grader_hacking

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| medxpertqa | 0.2397 [0.1950, 0.2864] | 0.7829 [0.7048, 0.8540] | 0.4968 [0.4383, 0.5543] | 0.9086 [0.8703, 0.9436] | 98 (49.0% [42.0, 56.0]) | 0.5941 [0.5109, 0.6788] | 0.128 [0.089, 0.169] | 0.093 [0.061, 0.129] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 213 (13.4%) | 0 (0.0%) |
| **Not acknowledged** | 315 (19.8%) | 1062 (66.8%) |

**Opacity rate:** 0.5966 [0.5118, 0.6724] — 315 covert / 528 hint-following runs (covert / all hint-following).

## medxpertqa × unethical

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| medxpertqa | 0.8138 [0.7705, 0.8542] | 0.9062 [0.8693, 0.9384] | 0.8751 [0.8433, 0.9047] | 0.9550 [0.9353, 0.9724] | 186 (93.0% [89.0, 96.5]) | 0.9183 [0.8918, 0.9422] | 0.128 [0.089, 0.169] | 0.015 [0.004, 0.030] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1273 (79.7%) | 4 (0.3%) |
| **Not acknowledged** | 174 (10.9%) | 146 (9.1%) |

**Opacity rate:** 0.1202 [0.0899, 0.1527] — 174 covert / 1447 hint-following runs (covert / all hint-following).

## lexam × professor

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| lexam | 0.1548 [0.1215, 0.1907] | 0.6282 [0.5340, 0.7143] | 0.3670 [0.3215, 0.4143] | 0.8097 [0.7535, 0.8646] | 90 (45.0% [38.0, 52.0]) | 0.8549 [0.7949, 0.9106] | 0.551 [0.493, 0.608] | 0.463 [0.405, 0.521] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 351 (22.2%) | 17 (1.1%) |
| **Not acknowledged** | 128 (8.1%) | 1083 (68.6%) |

**Opacity rate:** 0.2672 [0.2073, 0.3295] — 128 covert / 479 hint-following runs (covert / all hint-following).

## lexam × grader_hacking

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| lexam | 0.2177 [0.1787, 0.2598] | 0.6692 [0.5687, 0.7566] | 0.3930 [0.3441, 0.4446] | 0.8373 [0.7879, 0.8837] | 114 (57.0% [50.0, 64.0]) | 0.7625 [0.6939, 0.8267] | 0.551 [0.493, 0.608] | 0.439 [0.381, 0.496] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 316 (20.1%) | 9 (0.6%) |
| **Not acknowledged** | 262 (16.7%) | 985 (62.7%) |

**Opacity rate:** 0.4533 [0.3755, 0.5289] — 262 covert / 578 hint-following runs (covert / all hint-following).

## lexam × unethical

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| lexam | 0.6284 [0.5798, 0.6763] | 0.8320 [0.7856, 0.8744] | 0.7140 [0.6721, 0.7545] | 0.9029 [0.8709, 0.9321] | 176 (88.0% [83.5, 92.5]) | 0.9534 [0.9334, 0.9706] | 0.551 [0.493, 0.608] | 0.148 [0.110, 0.187] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1141 (71.6%) | 28 (1.8%) |
| **Not acknowledged** | 97 (6.1%) | 327 (20.5%) |

**Opacity rate:** 0.0784 [0.0533, 0.1055] — 97 covert / 1238 hint-following runs (covert / all hint-following).

## Summary

| Dataset | Hint type | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] | Acknowledged & followed | Acknowledged & not followed | Not Acknowledged & followed | Not Acknowledged & not followed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mmlu_pro | professor | 0.1353 [0.0990, 0.1747] | 0.8462 [0.7642, 0.9198] | 0.4666 [0.3820, 0.5516] | 0.9046 [0.8434, 0.9581] | 58 (29.0% [22.5, 35.5]) | 0.7801 [0.6877, 0.8638] | 0.731 [0.675, 0.785] | 0.660 [0.600, 0.718] | 208 (13.1%) | 7 (0.4%) | 71 (4.5%) | 1306 (82.0%) |
| mmlu_pro | grader_hacking | 0.1951 [0.1549, 0.2371] | 0.8924 [0.8247, 0.9488] | 0.4210 [0.3596, 0.4837] | 0.9512 [0.9137, 0.9820] | 93 (46.5% [39.5, 53.5]) | 0.7626 [0.6860, 0.8344] | 0.731 [0.675, 0.785] | 0.609 [0.551, 0.665] | 204 (13.0%) | 4 (0.3%) | 166 (10.5%) | 1200 (76.2%) |
| mmlu_pro | unethical | 0.6843 [0.6360, 0.7318] | 0.9564 [0.9300, 0.9793] | 0.7319 [0.6892, 0.7748] | 0.9717 [0.9524, 0.9882] | 187 (93.5% [90.0, 96.5]) | 0.9651 [0.9479, 0.9790] | 0.731 [0.675, 0.785] | 0.232 [0.189, 0.277] | 1089 (69.1%) | 13 (0.8%) | 56 (3.6%) | 418 (26.5%) |
| gpqa_diamond | professor | 0.3213 [0.2703, 0.3730] | 0.7381 [0.6731, 0.7991] | 0.5576 [0.5002, 0.6124] | 0.8393 [0.7919, 0.8831] | 116 (58.6% [52.0, 65.2]) | 0.7818 [0.7219, 0.8379] | 0.512 [0.453, 0.570] | 0.373 [0.316, 0.431] | 499 (32.0%) | 4 (0.3%) | 234 (15.0%) | 823 (52.8%) |
| gpqa_diamond | grader_hacking | 0.2796 [0.2321, 0.3278] | 0.7104 [0.6405, 0.7772] | 0.4979 [0.4405, 0.5565] | 0.8392 [0.7892, 0.8860] | 114 (57.6% [51.0, 64.1]) | 0.3267 [0.2522, 0.4002] | 0.512 [0.453, 0.570] | 0.404 [0.347, 0.460] | 143 (9.2%) | 1 (0.1%) | 521 (33.6%) | 884 (57.1%) |
| gpqa_diamond | unethical | 0.6482 [0.5983, 0.6973] | 0.8334 [0.7883, 0.8742] | 0.7334 [0.6896, 0.7758] | 0.8763 [0.8402, 0.9100] | 175 (88.4% [83.8, 92.4]) | 0.8791 [0.8461, 0.9095] | 0.512 [0.453, 0.570] | 0.160 [0.120, 0.201] | 1035 (66.6%) | 10 (0.6%) | 208 (13.4%) | 300 (19.3%) |
| wmdp | professor | 0.2037 [0.1607, 0.2493] | 0.7248 [0.6392, 0.8068] | 0.5346 [0.4678, 0.6025] | 0.8664 [0.8087, 0.9188] | 76 (38.6% [32.0, 45.2]) | 0.8882 [0.8379, 0.9332] | 0.681 [0.624, 0.738] | 0.546 [0.485, 0.608] | 347 (22.9%) | 6 (0.4%) | 124 (8.2%) | 1037 (68.5%) |
| wmdp | grader_hacking | 0.2167 [0.1722, 0.2633] | 0.7446 [0.6579, 0.8218] | 0.5241 [0.4619, 0.5873] | 0.8757 [0.8229, 0.9233] | 84 (42.4% [35.4, 49.5]) | 0.4447 [0.3540, 0.5362] | 0.677 [0.619, 0.734] | 0.510 [0.449, 0.573] | 153 (10.3%) | 0 (0.0%) | 330 (22.2%) | 1006 (67.6%) |
| wmdp | unethical | 0.6219 [0.5677, 0.6752] | 0.8700 [0.8239, 0.9115] | 0.7657 [0.7233, 0.8073] | 0.9352 [0.9061, 0.9608] | 160 (81.2% [75.6, 86.3]) | 0.9590 [0.9381, 0.9772] | 0.679 [0.621, 0.736] | 0.226 [0.179, 0.275] | 997 (65.9%) | 5 (0.3%) | 109 (7.2%) | 401 (26.5%) |
| codemmlu | professor | 0.1095 [0.0799, 0.1413] | 0.6589 [0.5592, 0.7535] | 0.3890 [0.3262, 0.4542] | 0.8292 [0.7597, 0.8938] | 57 (28.5% [22.5, 35.0]) | 0.6809 [0.5577, 0.7965] | 0.762 [0.709, 0.813] | 0.682 [0.623, 0.739] | 194 (12.4%) | 8 (0.5%) | 118 (7.5%) | 1250 (79.6%) |
| codemmlu | grader_hacking | 0.1092 [0.0801, 0.1408] | 0.6872 [0.5922, 0.7775] | 0.3552 [0.2920, 0.4217] | 0.8377 [0.7673, 0.9011] | 62 (31.0% [25.0, 37.5]) | 0.4766 [0.3565, 0.5961] | 0.762 [0.709, 0.813] | 0.685 [0.628, 0.740] | 99 (6.4%) | 0 (0.0%) | 202 (13.0%) | 1251 (80.6%) |
| codemmlu | unethical | 0.5010 [0.4479, 0.5538] | 0.8838 [0.8406, 0.9229] | 0.6515 [0.6017, 0.7002] | 0.9297 [0.8976, 0.9575] | 154 (77.0% [71.0, 82.5]) | 0.9169 [0.8849, 0.9449] | 0.762 [0.709, 0.813] | 0.383 [0.328, 0.439] | 790 (50.9%) | 33 (2.1%) | 128 (8.2%) | 602 (38.8%) |
| lsat_ar | professor | 0.0990 [0.0692, 0.1306] | 0.7296 [0.6138, 0.8326] | 0.3724 [0.3048, 0.4437] | 0.8569 [0.7869, 0.9197] | 57 (28.5% [22.5, 34.5]) | 0.7758 [0.6842, 0.8606] | 0.826 [0.784, 0.866] | 0.774 [0.726, 0.819] | 154 (9.8%) | 6 (0.4%) | 76 (4.8%) | 1341 (85.0%) |
| lsat_ar | grader_hacking | 0.1375 [0.1037, 0.1739] | 0.7297 [0.5889, 0.8436] | 0.3777 [0.3215, 0.4365] | 0.8752 [0.8161, 0.9273] | 77 (38.5% [32.0, 45.5]) | 0.7129 [0.6254, 0.7948] | 0.826 [0.784, 0.866] | 0.729 [0.679, 0.778] | 166 (11.0%) | 8 (0.5%) | 107 (7.1%) | 1230 (81.4%) |
| lsat_ar | unethical | 0.3932 [0.3451, 0.4412] | 0.9232 [0.8894, 0.9523] | 0.5288 [0.4810, 0.5768] | 0.9418 [0.9157, 0.9639] | 149 (74.5% [68.0, 80.5]) | 0.9168 [0.8909, 0.9399] | 0.826 [0.784, 0.866] | 0.520 [0.467, 0.571] | 592 (39.1%) | 32 (2.1%) | 79 (5.2%) | 812 (53.6%) |
| lsat_lr | professor | 0.0645 [0.0413, 0.0907] | 0.7024 [0.5804, 0.8179] | 0.3581 [0.2718, 0.4581] | 0.7414 [0.6242, 0.8493] | 36 (18.0% [13.0, 23.5]) | 0.4197 [0.2872, 0.5573] | 0.845 [0.800, 0.887] | 0.793 [0.744, 0.840] | 43 (2.7%) | 5 (0.3%) | 125 (7.8%) | 1424 (89.2%) |
| lsat_lr | grader_hacking | 0.0864 [0.0590, 0.1164] | 0.7832 [0.6820, 0.8774] | 0.3841 [0.3049, 0.4704] | 0.8529 [0.7721, 0.9265] | 45 (22.5% [17.0, 28.0]) | 0.4468 [0.3218, 0.5736] | 0.845 [0.800, 0.887] | 0.763 [0.714, 0.812] | 62 (3.9%) | 0 (0.0%) | 141 (8.9%) | 1387 (87.2%) |
| lsat_lr | unethical | 0.3142 [0.2613, 0.3670] | 0.9234 [0.8803, 0.9613] | 0.5818 [0.5183, 0.6434] | 0.9319 [0.8929, 0.9664] | 108 (54.0% [47.0, 61.0]) | 0.7818 [0.7229, 0.8373] | 0.845 [0.800, 0.887] | 0.587 [0.528, 0.646] | 383 (24.1%) | 5 (0.3%) | 180 (11.3%) | 1021 (64.3%) |
| medxpertqa | professor | 0.2559 [0.2103, 0.3038] | 0.7782 [0.6993, 0.8507] | 0.5357 [0.4754, 0.5972] | 0.8963 [0.8544, 0.9347] | 96 (48.0% [41.0, 55.0]) | 0.7699 [0.7081, 0.8298] | 0.128 [0.089, 0.169] | 0.089 [0.058, 0.125] | 346 (21.7%) | 4 (0.3%) | 208 (13.0%) | 1037 (65.0%) |
| medxpertqa | grader_hacking | 0.2397 [0.1950, 0.2864] | 0.7829 [0.7048, 0.8540] | 0.4968 [0.4383, 0.5543] | 0.9086 [0.8703, 0.9436] | 98 (49.0% [42.0, 56.0]) | 0.5941 [0.5109, 0.6788] | 0.128 [0.089, 0.169] | 0.093 [0.061, 0.129] | 213 (13.4%) | 0 (0.0%) | 315 (19.8%) | 1062 (66.8%) |
| medxpertqa | unethical | 0.8138 [0.7705, 0.8542] | 0.9062 [0.8693, 0.9384] | 0.8751 [0.8433, 0.9047] | 0.9550 [0.9353, 0.9724] | 186 (93.0% [89.0, 96.5]) | 0.9183 [0.8918, 0.9422] | 0.128 [0.089, 0.169] | 0.015 [0.004, 0.030] | 1273 (79.7%) | 4 (0.3%) | 174 (10.9%) | 146 (9.1%) |
| lexam | professor | 0.1548 [0.1215, 0.1907] | 0.6282 [0.5340, 0.7143] | 0.3670 [0.3215, 0.4143] | 0.8097 [0.7535, 0.8646] | 90 (45.0% [38.0, 52.0]) | 0.8549 [0.7949, 0.9106] | 0.551 [0.493, 0.608] | 0.463 [0.405, 0.521] | 351 (22.2%) | 17 (1.1%) | 128 (8.1%) | 1083 (68.6%) |
| lexam | grader_hacking | 0.2177 [0.1787, 0.2598] | 0.6692 [0.5687, 0.7566] | 0.3930 [0.3441, 0.4446] | 0.8373 [0.7879, 0.8837] | 114 (57.0% [50.0, 64.0]) | 0.7625 [0.6939, 0.8267] | 0.551 [0.493, 0.608] | 0.439 [0.381, 0.496] | 316 (20.1%) | 9 (0.6%) | 262 (16.7%) | 985 (62.7%) |
| lexam | unethical | 0.6284 [0.5798, 0.6763] | 0.8320 [0.7856, 0.8744] | 0.7140 [0.6721, 0.7545] | 0.9029 [0.8709, 0.9321] | 176 (88.0% [83.5, 92.5]) | 0.9534 [0.9334, 0.9706] | 0.551 [0.493, 0.608] | 0.148 [0.110, 0.187] | 1141 (71.6%) | 28 (1.8%) | 97 (6.1%) | 327 (20.5%) |
