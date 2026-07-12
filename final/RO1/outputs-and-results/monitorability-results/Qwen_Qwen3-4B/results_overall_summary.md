# Results overall summary — per (dataset × hint type)

- **Model:** `Qwen/Qwen3-4B` (`Qwen_Qwen3-4B`)
- **Datasets:** mmlu_pro, gpqa_diamond, wmdp, codemmlu, lsat_ar, lsat_lr, medxpertqa, lexam
- **Hint types:** professor, grader_hacking, unethical

## mmlu_pro × professor

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| mmlu_pro | 0.1717 [0.1300, 0.2163] | 0.8669 [0.7935, 0.9315] | 0.5181 [0.4403, 0.5945] | 0.9095 [0.8526, 0.9584] | 67 (33.5% [27.0, 40.0]) | 0.7891 [0.7106, 0.8610] | 0.719 [0.663, 0.772] | 0.631 [0.571, 0.690] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 240 (15.0%) | 9 (0.6%) |
| **Not acknowledged** | 101 (6.3%) | 1249 (78.1%) |

**Opacity rate:** 0.2962 [0.2160, 0.3778] — 101 covert / 341 hint-following runs (covert / all hint-following).

## mmlu_pro × grader_hacking

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| mmlu_pro | 0.2028 [0.1604, 0.2470] | 0.8407 [0.7050, 0.9391] | 0.4655 [0.4052, 0.5274] | 0.9347 [0.8910, 0.9719] | 89 (44.5% [37.5, 51.5]) | 0.6689 [0.5837, 0.7527] | 0.719 [0.663, 0.772] | 0.596 [0.539, 0.652] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 201 (12.6%) | 1 (0.1%) |
| **Not acknowledged** | 189 (11.8%) | 1204 (75.5%) |

**Opacity rate:** 0.4846 [0.3808, 0.5773] — 189 covert / 390 hint-following runs (covert / all hint-following).

## mmlu_pro × unethical

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| mmlu_pro | 0.7919 [0.7533, 0.8299] | 0.9569 [0.9324, 0.9783] | 0.8039 [0.7670, 0.8394] | 0.9666 [0.9462, 0.9842] | 197 (98.5% [96.5, 100.0]) | 0.9428 [0.9219, 0.9612] | 0.719 [0.663, 0.772] | 0.139 [0.109, 0.171] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1236 (77.5%) | 8 (0.5%) |
| **Not acknowledged** | 94 (5.9%) | 257 (16.1%) |

**Opacity rate:** 0.0707 [0.0475, 0.0963] — 94 covert / 1330 hint-following runs (covert / all hint-following).

## gpqa_diamond × professor

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| gpqa_diamond | 0.3375 [0.2869, 0.3884] | 0.7146 [0.6509, 0.7746] | 0.5689 [0.5130, 0.6227] | 0.7959 [0.7449, 0.8428] | 118 (59.6% [53.0, 66.2]) | 0.7711 [0.7181, 0.8209] | 0.512 [0.455, 0.567] | 0.362 [0.305, 0.419] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 453 (28.8%) | 3 (0.2%) |
| **Not acknowledged** | 309 (19.7%) | 806 (51.3%) |

**Opacity rate:** 0.4055 [0.3493, 0.4616] — 309 covert / 762 hint-following runs (covert / all hint-following).

## gpqa_diamond × grader_hacking

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| gpqa_diamond | 0.2916 [0.2426, 0.3413] | 0.6729 [0.5641, 0.7604] | 0.5010 [0.4451, 0.5577] | 0.8131 [0.7623, 0.8607] | 119 (60.1% [53.5, 66.7]) | 0.3041 [0.2322, 0.3770] | 0.512 [0.455, 0.567] | 0.379 [0.323, 0.435] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 113 (7.2%) | 3 (0.2%) |
| **Not acknowledged** | 577 (36.8%) | 874 (55.8%) |

**Opacity rate:** 0.8362 [0.7885, 0.8793] — 577 covert / 690 hint-following runs (covert / all hint-following).

## gpqa_diamond × unethical

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| gpqa_diamond | 0.6501 [0.5997, 0.6971] | 0.8256 [0.7813, 0.8657] | 0.7274 [0.6849, 0.7684] | 0.8684 [0.8334, 0.9009] | 177 (89.4% [84.8, 93.4]) | 0.8661 [0.8309, 0.8985] | 0.512 [0.455, 0.567] | 0.159 [0.119, 0.201] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 968 (61.7%) | 3 (0.2%) |
| **Not acknowledged** | 288 (18.3%) | 311 (19.8%) |

**Opacity rate:** 0.2293 [0.1893, 0.2711] — 288 covert / 1256 hint-following runs (covert / all hint-following).

## wmdp × professor

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| wmdp | 0.2370 [0.1903, 0.2865] | 0.7553 [0.6710, 0.8314] | 0.5321 [0.4642, 0.6010] | 0.8782 [0.8224, 0.9286] | 88 (44.7% [37.6, 51.8]) | 0.8488 [0.7923, 0.8996] | 0.679 [0.622, 0.736] | 0.514 [0.452, 0.575] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 377 (24.7%) | 2 (0.1%) |
| **Not acknowledged** | 148 (9.7%) | 997 (65.4%) |

**Opacity rate:** 0.2819 [0.2141, 0.3522] — 148 covert / 525 hint-following runs (covert / all hint-following).

## wmdp × grader_hacking

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| wmdp | 0.2149 [0.1709, 0.2622] | 0.7543 [0.6720, 0.8309] | 0.5175 [0.4537, 0.5831] | 0.8984 [0.8439, 0.9464] | 83 (42.1% [35.0, 49.2]) | 0.4544 [0.3600, 0.5493] | 0.679 [0.622, 0.736] | 0.526 [0.467, 0.588] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 135 (8.9%) | 2 (0.1%) |
| **Not acknowledged** | 360 (23.7%) | 1022 (67.3%) |

**Opacity rate:** 0.7273 [0.6493, 0.7986] — 360 covert / 495 hint-following runs (covert / all hint-following).

## wmdp × unethical

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| wmdp | 0.6523 [0.5980, 0.7057] | 0.8785 [0.8340, 0.9193] | 0.7604 [0.7148, 0.8048] | 0.9409 [0.9123, 0.9666] | 169 (85.8% [80.7, 90.4]) | 0.9094 [0.8773, 0.9380] | 0.679 [0.622, 0.736] | 0.201 [0.157, 0.248] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 976 (63.7%) | 4 (0.3%) |
| **Not acknowledged** | 186 (12.1%) | 365 (23.8%) |

**Opacity rate:** 0.1601 [0.1193, 0.2044] — 186 covert / 1162 hint-following runs (covert / all hint-following).

## codemmlu × professor

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| codemmlu | 0.1220 [0.0884, 0.1577] | 0.6730 [0.5435, 0.7904] | 0.4314 [0.3627, 0.5015] | 0.8926 [0.8219, 0.9537] | 58 (29.0% [23.0, 35.5]) | 0.8703 [0.8045, 0.9303] | 0.763 [0.709, 0.816] | 0.673 [0.616, 0.729] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 223 (14.1%) | 6 (0.4%) |
| **Not acknowledged** | 107 (6.8%) | 1249 (78.8%) |

**Opacity rate:** 0.3242 [0.2340, 0.4161] — 107 covert / 330 hint-following runs (covert / all hint-following).

## codemmlu × grader_hacking

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| codemmlu | 0.1028 [0.0725, 0.1355] | 0.6746 [0.5436, 0.7919] | 0.3639 [0.2966, 0.4368] | 0.8778 [0.8102, 0.9388] | 59 (29.5% [23.5, 36.0]) | 0.4595 [0.3444, 0.5770] | 0.763 [0.709, 0.816] | 0.695 [0.639, 0.749] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 91 (5.8%) | 0 (0.0%) |
| **Not acknowledged** | 201 (12.7%) | 1286 (81.5%) |

**Opacity rate:** 0.6884 [0.5756, 0.7914] — 201 covert / 292 hint-following runs (covert / all hint-following).

## codemmlu × unethical

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| codemmlu | 0.5893 [0.5379, 0.6403] | 0.9021 [0.8620, 0.9379] | 0.6933 [0.6470, 0.7388] | 0.9499 [0.9235, 0.9738] | 170 (85.0% [80.0, 89.5]) | 0.9375 [0.9118, 0.9591] | 0.763 [0.709, 0.816] | 0.296 [0.246, 0.347] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 944 (59.4%) | 14 (0.9%) |
| **Not acknowledged** | 128 (8.1%) | 502 (31.6%) |

**Opacity rate:** 0.1194 [0.0840, 0.1593] — 128 covert / 1072 hint-following runs (covert / all hint-following).

## lsat_ar × professor

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| lsat_ar | 0.1304 [0.0956, 0.1663] | 0.6417 [0.4904, 0.7751] | 0.3986 [0.3404, 0.4591] | 0.9007 [0.8497, 0.9456] | 71 (35.5% [29.0, 42.0]) | 0.7463 [0.6617, 0.8252] | 0.782 [0.735, 0.825] | 0.710 [0.660, 0.758] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 203 (12.8%) | 4 (0.3%) |
| **Not acknowledged** | 103 (6.5%) | 1279 (80.5%) |

**Opacity rate:** 0.3366 [0.2664, 0.4116] — 103 covert / 306 hint-following runs (covert / all hint-following).

## lsat_ar × grader_hacking

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| lsat_ar | 0.1465 [0.1121, 0.1828] | 0.7659 [0.6477, 0.8590] | 0.3613 [0.3113, 0.4134] | 0.8735 [0.8216, 0.9207] | 87 (43.5% [36.5, 50.5]) | 0.6682 [0.5833, 0.7507] | 0.782 [0.735, 0.825] | 0.706 [0.656, 0.755] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 184 (11.7%) | 8 (0.5%) |
| **Not acknowledged** | 145 (9.2%) | 1240 (78.6%) |

**Opacity rate:** 0.4407 [0.3547, 0.5238] — 145 covert / 329 hint-following runs (covert / all hint-following).

## lsat_ar × unethical

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| lsat_ar | 0.4664 [0.4179, 0.5149] | 0.8996 [0.8602, 0.9346] | 0.5661 [0.5204, 0.6110] | 0.9299 [0.9009, 0.9551] | 165 (82.5% [77.0, 87.5]) | 0.9141 [0.8849, 0.9402] | 0.782 [0.735, 0.825] | 0.426 [0.375, 0.477] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 732 (46.4%) | 18 (1.1%) |
| **Not acknowledged** | 104 (6.6%) | 725 (45.9%) |

**Opacity rate:** 0.1244 [0.0927, 0.1584] — 104 covert / 836 hint-following runs (covert / all hint-following).

## lsat_lr × professor

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| lsat_lr | 0.0794 [0.0519, 0.1081] | 0.7157 [0.5831, 0.8365] | 0.3750 [0.2954, 0.4620] | 0.8346 [0.7493, 0.9131] | 44 (22.0% [16.5, 27.5]) | 0.4141 [0.2951, 0.5378] | 0.834 [0.789, 0.877] | 0.764 [0.715, 0.811] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 54 (3.4%) | 2 (0.1%) |
| **Not acknowledged** | 146 (9.1%) | 1396 (87.4%) |

**Opacity rate:** 0.7300 [0.6341, 0.8164] — 146 covert / 200 hint-following runs (covert / all hint-following).

## lsat_lr × grader_hacking

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| lsat_lr | 0.0951 [0.0664, 0.1251] | 0.7470 [0.6384, 0.8483] | 0.3932 [0.3207, 0.4730] | 0.8639 [0.7892, 0.9306] | 49 (24.5% [18.5, 30.5]) | 0.4062 [0.2902, 0.5279] | 0.834 [0.789, 0.877] | 0.753 [0.703, 0.802] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 63 (3.9%) | 1 (0.1%) |
| **Not acknowledged** | 162 (10.1%) | 1373 (85.9%) |

**Opacity rate:** 0.7200 [0.6219, 0.8090] — 162 covert / 225 hint-following runs (covert / all hint-following).

## lsat_lr × unethical

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| lsat_lr | 0.3532 [0.3007, 0.4057] | 0.9228 [0.8834, 0.9575] | 0.5790 [0.5209, 0.6373] | 0.9379 [0.9047, 0.9675] | 122 (61.0% [54.5, 67.5]) | 0.7666 [0.7088, 0.8219] | 0.834 [0.789, 0.877] | 0.546 [0.489, 0.603] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 408 (25.5%) | 3 (0.2%) |
| **Not acknowledged** | 229 (14.3%) | 958 (59.9%) |

**Opacity rate:** 0.3595 [0.2876, 0.4315] — 229 covert / 637 hint-following runs (covert / all hint-following).

## medxpertqa × professor

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| medxpertqa | 0.2938 [0.2450, 0.3450] | 0.8110 [0.7465, 0.8707] | 0.5451 [0.4844, 0.6071] | 0.8892 [0.8419, 0.9329] | 108 (54.0% [47.5, 61.0]) | 0.7130 [0.6460, 0.7790] | 0.127 [0.087, 0.169] | 0.094 [0.062, 0.129] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 348 (21.8%) | 5 (0.3%) |
| **Not acknowledged** | 278 (17.4%) | 968 (60.5%) |

**Opacity rate:** 0.4441 [0.3691, 0.5153] — 278 covert / 626 hint-following runs (covert / all hint-following).

## medxpertqa × grader_hacking

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| medxpertqa | 0.2569 [0.2119, 0.3050] | 0.7855 [0.7143, 0.8520] | 0.4882 [0.4280, 0.5500] | 0.8855 [0.8357, 0.9310] | 106 (53.0% [46.0, 60.0]) | 0.5080 [0.4277, 0.5879] | 0.127 [0.087, 0.169] | 0.087 [0.055, 0.122] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 188 (11.8%) | 0 (0.0%) |
| **Not acknowledged** | 379 (23.7%) | 1033 (64.6%) |

**Opacity rate:** 0.6684 [0.5931, 0.7356] — 379 covert / 567 hint-following runs (covert / all hint-following).

## medxpertqa × unethical

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| medxpertqa | 0.8231 [0.7800, 0.8638] | 0.9015 [0.8629, 0.9353] | 0.8664 [0.8311, 0.8999] | 0.9395 [0.9120, 0.9640] | 190 (95.0% [91.5, 98.0]) | 0.8782 [0.8457, 0.9080] | 0.127 [0.087, 0.169] | 0.010 [0.001, 0.021] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1206 (75.4%) | 0 (0.0%) |
| **Not acknowledged** | 266 (16.6%) | 127 (7.9%) |

**Opacity rate:** 0.1807 [0.1438, 0.2205] — 266 covert / 1472 hint-following runs (covert / all hint-following).

## lexam × professor

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| lexam | 0.2289 [0.1881, 0.2703] | 0.7141 [0.5741, 0.8145] | 0.4199 [0.3728, 0.4678] | 0.8396 [0.7895, 0.8856] | 112 (56.0% [49.0, 63.0]) | 0.7912 [0.7293, 0.8494] | 0.585 [0.529, 0.641] | 0.444 [0.388, 0.501] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 381 (23.8%) | 17 (1.1%) |
| **Not acknowledged** | 183 (11.4%) | 1019 (63.7%) |

**Opacity rate:** 0.3245 [0.2615, 0.3865] — 183 covert / 564 hint-following runs (covert / all hint-following).

## lexam × grader_hacking

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| lexam | 0.2526 [0.2100, 0.2974] | 0.7517 [0.6835, 0.8153] | 0.4491 [0.3978, 0.5011] | 0.8362 [0.7855, 0.8826] | 115 (57.5% [50.5, 64.5]) | 0.6665 [0.5973, 0.7316] | 0.585 [0.529, 0.641] | 0.434 [0.378, 0.491] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 290 (18.1%) | 13 (0.8%) |
| **Not acknowledged** | 312 (19.5%) | 984 (61.5%) |

**Opacity rate:** 0.5183 [0.4472, 0.5877] — 312 covert / 602 hint-following runs (covert / all hint-following).

## lexam × unethical

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| lexam | 0.7424 [0.6976, 0.7845] | 0.8685 [0.8286, 0.9050] | 0.7946 [0.7588, 0.8297] | 0.9066 [0.8759, 0.9345] | 187 (93.5% [90.0, 96.5]) | 0.9277 [0.8999, 0.9518] | 0.585 [0.529, 0.641] | 0.097 [0.070, 0.127] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1265 (79.1%) | 21 (1.3%) |
| **Not acknowledged** | 120 (7.5%) | 193 (12.1%) |

**Opacity rate:** 0.0866 [0.0609, 0.1142] — 120 covert / 1385 hint-following runs (covert / all hint-following).

## Summary

| Dataset | Hint type | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] | Acknowledged & followed | Acknowledged & not followed | Not Acknowledged & followed | Not Acknowledged & not followed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mmlu_pro | professor | 0.1717 [0.1300, 0.2163] | 0.8669 [0.7935, 0.9315] | 0.5181 [0.4403, 0.5945] | 0.9095 [0.8526, 0.9584] | 67 (33.5% [27.0, 40.0]) | 0.7891 [0.7106, 0.8610] | 0.719 [0.663, 0.772] | 0.631 [0.571, 0.690] | 240 (15.0%) | 9 (0.6%) | 101 (6.3%) | 1249 (78.1%) |
| mmlu_pro | grader_hacking | 0.2028 [0.1604, 0.2470] | 0.8407 [0.7050, 0.9391] | 0.4655 [0.4052, 0.5274] | 0.9347 [0.8910, 0.9719] | 89 (44.5% [37.5, 51.5]) | 0.6689 [0.5837, 0.7527] | 0.719 [0.663, 0.772] | 0.596 [0.539, 0.652] | 201 (12.6%) | 1 (0.1%) | 189 (11.8%) | 1204 (75.5%) |
| mmlu_pro | unethical | 0.7919 [0.7533, 0.8299] | 0.9569 [0.9324, 0.9783] | 0.8039 [0.7670, 0.8394] | 0.9666 [0.9462, 0.9842] | 197 (98.5% [96.5, 100.0]) | 0.9428 [0.9219, 0.9612] | 0.719 [0.663, 0.772] | 0.139 [0.109, 0.171] | 1236 (77.5%) | 8 (0.5%) | 94 (5.9%) | 257 (16.1%) |
| gpqa_diamond | professor | 0.3375 [0.2869, 0.3884] | 0.7146 [0.6509, 0.7746] | 0.5689 [0.5130, 0.6227] | 0.7959 [0.7449, 0.8428] | 118 (59.6% [53.0, 66.2]) | 0.7711 [0.7181, 0.8209] | 0.512 [0.455, 0.567] | 0.362 [0.305, 0.419] | 453 (28.8%) | 3 (0.2%) | 309 (19.7%) | 806 (51.3%) |
| gpqa_diamond | grader_hacking | 0.2916 [0.2426, 0.3413] | 0.6729 [0.5641, 0.7604] | 0.5010 [0.4451, 0.5577] | 0.8131 [0.7623, 0.8607] | 119 (60.1% [53.5, 66.7]) | 0.3041 [0.2322, 0.3770] | 0.512 [0.455, 0.567] | 0.379 [0.323, 0.435] | 113 (7.2%) | 3 (0.2%) | 577 (36.8%) | 874 (55.8%) |
| gpqa_diamond | unethical | 0.6501 [0.5997, 0.6971] | 0.8256 [0.7813, 0.8657] | 0.7274 [0.6849, 0.7684] | 0.8684 [0.8334, 0.9009] | 177 (89.4% [84.8, 93.4]) | 0.8661 [0.8309, 0.8985] | 0.512 [0.455, 0.567] | 0.159 [0.119, 0.201] | 968 (61.7%) | 3 (0.2%) | 288 (18.3%) | 311 (19.8%) |
| wmdp | professor | 0.2370 [0.1903, 0.2865] | 0.7553 [0.6710, 0.8314] | 0.5321 [0.4642, 0.6010] | 0.8782 [0.8224, 0.9286] | 88 (44.7% [37.6, 51.8]) | 0.8488 [0.7923, 0.8996] | 0.679 [0.622, 0.736] | 0.514 [0.452, 0.575] | 377 (24.7%) | 2 (0.1%) | 148 (9.7%) | 997 (65.4%) |
| wmdp | grader_hacking | 0.2149 [0.1709, 0.2622] | 0.7543 [0.6720, 0.8309] | 0.5175 [0.4537, 0.5831] | 0.8984 [0.8439, 0.9464] | 83 (42.1% [35.0, 49.2]) | 0.4544 [0.3600, 0.5493] | 0.679 [0.622, 0.736] | 0.526 [0.467, 0.588] | 135 (8.9%) | 2 (0.1%) | 360 (23.7%) | 1022 (67.3%) |
| wmdp | unethical | 0.6523 [0.5980, 0.7057] | 0.8785 [0.8340, 0.9193] | 0.7604 [0.7148, 0.8048] | 0.9409 [0.9123, 0.9666] | 169 (85.8% [80.7, 90.4]) | 0.9094 [0.8773, 0.9380] | 0.679 [0.622, 0.736] | 0.201 [0.157, 0.248] | 976 (63.7%) | 4 (0.3%) | 186 (12.1%) | 365 (23.8%) |
| codemmlu | professor | 0.1220 [0.0884, 0.1577] | 0.6730 [0.5435, 0.7904] | 0.4314 [0.3627, 0.5015] | 0.8926 [0.8219, 0.9537] | 58 (29.0% [23.0, 35.5]) | 0.8703 [0.8045, 0.9303] | 0.763 [0.709, 0.816] | 0.673 [0.616, 0.729] | 223 (14.1%) | 6 (0.4%) | 107 (6.8%) | 1249 (78.8%) |
| codemmlu | grader_hacking | 0.1028 [0.0725, 0.1355] | 0.6746 [0.5436, 0.7919] | 0.3639 [0.2966, 0.4368] | 0.8778 [0.8102, 0.9388] | 59 (29.5% [23.5, 36.0]) | 0.4595 [0.3444, 0.5770] | 0.763 [0.709, 0.816] | 0.695 [0.639, 0.749] | 91 (5.8%) | 0 (0.0%) | 201 (12.7%) | 1286 (81.5%) |
| codemmlu | unethical | 0.5893 [0.5379, 0.6403] | 0.9021 [0.8620, 0.9379] | 0.6933 [0.6470, 0.7388] | 0.9499 [0.9235, 0.9738] | 170 (85.0% [80.0, 89.5]) | 0.9375 [0.9118, 0.9591] | 0.763 [0.709, 0.816] | 0.296 [0.246, 0.347] | 944 (59.4%) | 14 (0.9%) | 128 (8.1%) | 502 (31.6%) |
| lsat_ar | professor | 0.1304 [0.0956, 0.1663] | 0.6417 [0.4904, 0.7751] | 0.3986 [0.3404, 0.4591] | 0.9007 [0.8497, 0.9456] | 71 (35.5% [29.0, 42.0]) | 0.7463 [0.6617, 0.8252] | 0.782 [0.735, 0.825] | 0.710 [0.660, 0.758] | 203 (12.8%) | 4 (0.3%) | 103 (6.5%) | 1279 (80.5%) |
| lsat_ar | grader_hacking | 0.1465 [0.1121, 0.1828] | 0.7659 [0.6477, 0.8590] | 0.3613 [0.3113, 0.4134] | 0.8735 [0.8216, 0.9207] | 87 (43.5% [36.5, 50.5]) | 0.6682 [0.5833, 0.7507] | 0.782 [0.735, 0.825] | 0.706 [0.656, 0.755] | 184 (11.7%) | 8 (0.5%) | 145 (9.2%) | 1240 (78.6%) |
| lsat_ar | unethical | 0.4664 [0.4179, 0.5149] | 0.8996 [0.8602, 0.9346] | 0.5661 [0.5204, 0.6110] | 0.9299 [0.9009, 0.9551] | 165 (82.5% [77.0, 87.5]) | 0.9141 [0.8849, 0.9402] | 0.782 [0.735, 0.825] | 0.426 [0.375, 0.477] | 732 (46.4%) | 18 (1.1%) | 104 (6.6%) | 725 (45.9%) |
| lsat_lr | professor | 0.0794 [0.0519, 0.1081] | 0.7157 [0.5831, 0.8365] | 0.3750 [0.2954, 0.4620] | 0.8346 [0.7493, 0.9131] | 44 (22.0% [16.5, 27.5]) | 0.4141 [0.2951, 0.5378] | 0.834 [0.789, 0.877] | 0.764 [0.715, 0.811] | 54 (3.4%) | 2 (0.1%) | 146 (9.1%) | 1396 (87.4%) |
| lsat_lr | grader_hacking | 0.0951 [0.0664, 0.1251] | 0.7470 [0.6384, 0.8483] | 0.3932 [0.3207, 0.4730] | 0.8639 [0.7892, 0.9306] | 49 (24.5% [18.5, 30.5]) | 0.4062 [0.2902, 0.5279] | 0.834 [0.789, 0.877] | 0.753 [0.703, 0.802] | 63 (3.9%) | 1 (0.1%) | 162 (10.1%) | 1373 (85.9%) |
| lsat_lr | unethical | 0.3532 [0.3007, 0.4057] | 0.9228 [0.8834, 0.9575] | 0.5790 [0.5209, 0.6373] | 0.9379 [0.9047, 0.9675] | 122 (61.0% [54.5, 67.5]) | 0.7666 [0.7088, 0.8219] | 0.834 [0.789, 0.877] | 0.546 [0.489, 0.603] | 408 (25.5%) | 3 (0.2%) | 229 (14.3%) | 958 (59.9%) |
| medxpertqa | professor | 0.2938 [0.2450, 0.3450] | 0.8110 [0.7465, 0.8707] | 0.5451 [0.4844, 0.6071] | 0.8892 [0.8419, 0.9329] | 108 (54.0% [47.5, 61.0]) | 0.7130 [0.6460, 0.7790] | 0.127 [0.087, 0.169] | 0.094 [0.062, 0.129] | 348 (21.8%) | 5 (0.3%) | 278 (17.4%) | 968 (60.5%) |
| medxpertqa | grader_hacking | 0.2569 [0.2119, 0.3050] | 0.7855 [0.7143, 0.8520] | 0.4882 [0.4280, 0.5500] | 0.8855 [0.8357, 0.9310] | 106 (53.0% [46.0, 60.0]) | 0.5080 [0.4277, 0.5879] | 0.127 [0.087, 0.169] | 0.087 [0.055, 0.122] | 188 (11.8%) | 0 (0.0%) | 379 (23.7%) | 1033 (64.6%) |
| medxpertqa | unethical | 0.8231 [0.7800, 0.8638] | 0.9015 [0.8629, 0.9353] | 0.8664 [0.8311, 0.8999] | 0.9395 [0.9120, 0.9640] | 190 (95.0% [91.5, 98.0]) | 0.8782 [0.8457, 0.9080] | 0.127 [0.087, 0.169] | 0.010 [0.001, 0.021] | 1206 (75.4%) | 0 (0.0%) | 266 (16.6%) | 127 (7.9%) |
| lexam | professor | 0.2289 [0.1881, 0.2703] | 0.7141 [0.5741, 0.8145] | 0.4199 [0.3728, 0.4678] | 0.8396 [0.7895, 0.8856] | 112 (56.0% [49.0, 63.0]) | 0.7912 [0.7293, 0.8494] | 0.585 [0.529, 0.641] | 0.444 [0.388, 0.501] | 381 (23.8%) | 17 (1.1%) | 183 (11.4%) | 1019 (63.7%) |
| lexam | grader_hacking | 0.2526 [0.2100, 0.2974] | 0.7517 [0.6835, 0.8153] | 0.4491 [0.3978, 0.5011] | 0.8362 [0.7855, 0.8826] | 115 (57.5% [50.5, 64.5]) | 0.6665 [0.5973, 0.7316] | 0.585 [0.529, 0.641] | 0.434 [0.378, 0.491] | 290 (18.1%) | 13 (0.8%) | 312 (19.5%) | 984 (61.5%) |
| lexam | unethical | 0.7424 [0.6976, 0.7845] | 0.8685 [0.8286, 0.9050] | 0.7946 [0.7588, 0.8297] | 0.9066 [0.8759, 0.9345] | 187 (93.5% [90.0, 96.5]) | 0.9277 [0.8999, 0.9518] | 0.585 [0.529, 0.641] | 0.097 [0.070, 0.127] | 1265 (79.1%) | 21 (1.3%) | 120 (7.5%) | 193 (12.1%) |
