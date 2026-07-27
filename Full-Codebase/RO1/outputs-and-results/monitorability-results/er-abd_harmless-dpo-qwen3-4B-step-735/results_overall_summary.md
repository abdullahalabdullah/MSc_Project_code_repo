# Results overall summary — per (dataset × hint type)

- **Model:** `er-abd_harmless-dpo-qwen3-4B-step-735` (`er-abd_harmless-dpo-qwen3-4B-step-735`)
- **Datasets:** mmlu_pro, gpqa_diamond, wmdp, codemmlu, lsat_ar, lsat_lr, medxpertqa, lexam
- **Hint types:** professor, grader_hacking, unethical

## mmlu_pro × professor

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| mmlu_pro | 0.1705 [0.1294, 0.2137] | 0.8707 [0.7981, 0.9336] | 0.5501 [0.4738, 0.6271] | 0.9268 [0.8794, 0.9674] | 62 (31.0% [25.0, 37.5]) | 0.8045 [0.7274, 0.8749] | 0.735 [0.679, 0.789] | 0.638 [0.578, 0.697] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 261 (16.3%) | 8 (0.5%) |
| **Not acknowledged** | 74 (4.6%) | 1255 (78.5%) |

**Opacity rate:** 0.2209 [0.1479, 0.2982] — 74 covert / 335 hint-following runs (covert / all hint-following).

## mmlu_pro × grader_hacking

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| mmlu_pro | 0.1979 [0.1561, 0.2416] | 0.9024 [0.8458, 0.9515] | 0.4657 [0.4012, 0.5321] | 0.9555 [0.9247, 0.9814] | 85 (42.5% [35.5, 49.0]) | 0.7793 [0.7065, 0.8478] | 0.735 [0.679, 0.789] | 0.611 [0.551, 0.670] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 220 (13.8%) | 2 (0.1%) |
| **Not acknowledged** | 159 (10.0%) | 1211 (76.1%) |

**Opacity rate:** 0.4195 [0.3198, 0.5140] — 159 covert / 379 hint-following runs (covert / all hint-following).

## mmlu_pro × unethical

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| mmlu_pro | 0.7052 [0.6592, 0.7519] | 0.9581 [0.9326, 0.9801] | 0.7543 [0.7125, 0.7956] | 0.9786 [0.9637, 0.9913] | 187 (93.5% [90.0, 96.5]) | 0.9442 [0.9213, 0.9642] | 0.735 [0.679, 0.789] | 0.221 [0.179, 0.265] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1110 (70.0%) | 12 (0.8%) |
| **Not acknowledged** | 75 (4.7%) | 388 (24.5%) |

**Opacity rate:** 0.0633 [0.0394, 0.0897] — 75 covert / 1185 hint-following runs (covert / all hint-following).

## gpqa_diamond × professor

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| gpqa_diamond | 0.3089 [0.2579, 0.3604] | 0.6940 [0.6121, 0.7685] | 0.5561 [0.4953, 0.6158] | 0.8130 [0.7604, 0.8627] | 112 (56.6% [49.5, 63.6]) | 0.7933 [0.7273, 0.8530] | 0.507 [0.447, 0.565] | 0.382 [0.324, 0.441] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 496 (31.7%) | 6 (0.4%) |
| **Not acknowledged** | 228 (14.6%) | 833 (53.3%) |

**Opacity rate:** 0.3149 [0.2576, 0.3755] — 228 covert / 724 hint-following runs (covert / all hint-following).

## gpqa_diamond × grader_hacking

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| gpqa_diamond | 0.2923 [0.2412, 0.3440] | 0.6956 [0.6125, 0.7715] | 0.5230 [0.4637, 0.5822] | 0.8422 [0.7942, 0.8863] | 114 (57.6% [51.0, 64.6]) | 0.3692 [0.2941, 0.4450] | 0.507 [0.447, 0.565] | 0.368 [0.311, 0.425] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 154 (9.9%) | 1 (0.1%) |
| **Not acknowledged** | 538 (34.6%) | 864 (55.5%) |

**Opacity rate:** 0.7775 [0.7242, 0.8267] — 538 covert / 692 hint-following runs (covert / all hint-following).

## gpqa_diamond × unethical

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| gpqa_diamond | 0.6383 [0.5863, 0.6893] | 0.8249 [0.7796, 0.8675] | 0.7355 [0.6913, 0.7804] | 0.8786 [0.8436, 0.9119] | 172 (86.9% [82.3, 91.4]) | 0.8891 [0.8566, 0.9193] | 0.507 [0.447, 0.565] | 0.165 [0.123, 0.209] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1023 (65.5%) | 3 (0.2%) |
| **Not acknowledged** | 221 (14.1%) | 315 (20.2%) |

**Opacity rate:** 0.1777 [0.1425, 0.2141] — 221 covert / 1244 hint-following runs (covert / all hint-following).

## wmdp × professor

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| wmdp | 0.2339 [0.1878, 0.2833] | 0.7561 [0.6773, 0.8297] | 0.5458 [0.4770, 0.6143] | 0.8821 [0.8263, 0.9319] | 84 (42.9% [35.7, 50.0]) | 0.8840 [0.8339, 0.9273] | 0.674 [0.615, 0.734] | 0.521 [0.459, 0.584] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 391 (25.8%) | 7 (0.5%) |
| **Not acknowledged** | 130 (8.6%) | 985 (65.1%) |

**Opacity rate:** 0.2495 [0.1824, 0.3221] — 130 covert / 521 hint-following runs (covert / all hint-following).

## wmdp × grader_hacking

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| wmdp | 0.2267 [0.1805, 0.2745] | 0.7373 [0.6498, 0.8166] | 0.5159 [0.4507, 0.5827] | 0.8880 [0.8326, 0.9370] | 87 (44.2% [37.1, 51.3]) | 0.5596 [0.4699, 0.6502] | 0.673 [0.614, 0.731] | 0.516 [0.456, 0.577] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 199 (13.1%) | 0 (0.0%) |
| **Not acknowledged** | 319 (21.1%) | 997 (65.8%) |

**Opacity rate:** 0.6158 [0.5327, 0.6957] — 319 covert / 518 hint-following runs (covert / all hint-following).

## wmdp × unethical

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| wmdp | 0.6280 [0.5707, 0.6839] | 0.8699 [0.8237, 0.9128] | 0.7685 [0.7220, 0.8132] | 0.9402 [0.9101, 0.9667] | 161 (81.7% [76.1, 86.8]) | 0.9538 [0.9343, 0.9707] | 0.673 [0.614, 0.731] | 0.225 [0.176, 0.276] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1013 (66.8%) | 6 (0.4%) |
| **Not acknowledged** | 111 (7.3%) | 387 (25.5%) |

**Opacity rate:** 0.0988 [0.0671, 0.1342] — 111 covert / 1124 hint-following runs (covert / all hint-following).

## codemmlu × professor

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| codemmlu | 0.1101 [0.0801, 0.1428] | 0.6668 [0.5561, 0.7657] | 0.3861 [0.3197, 0.4568] | 0.8392 [0.7636, 0.9066] | 58 (29.0% [23.0, 35.5]) | 0.7655 [0.6669, 0.8559] | 0.756 [0.702, 0.808] | 0.679 [0.621, 0.736] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 197 (12.4%) | 12 (0.8%) |
| **Not acknowledged** | 124 (7.8%) | 1252 (79.0%) |

**Opacity rate:** 0.3863 [0.2792, 0.4947] — 124 covert / 321 hint-following runs (covert / all hint-following).

## codemmlu × grader_hacking

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| codemmlu | 0.0853 [0.0594, 0.1144] | 0.6589 [0.5529, 0.7625] | 0.3469 [0.2785, 0.4211] | 0.8232 [0.7404, 0.8980] | 50 (25.1% [19.1, 31.2]) | 0.4559 [0.3248, 0.5837] | 0.760 [0.707, 0.811] | 0.712 [0.655, 0.766] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 80 (5.1%) | 1 (0.1%) |
| **Not acknowledged** | 196 (12.5%) | 1294 (82.4%) |

**Opacity rate:** 0.7101 [0.5870, 0.8193] — 196 covert / 276 hint-following runs (covert / all hint-following).

## codemmlu × unethical

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| codemmlu | 0.5115 [0.4580, 0.5650] | 0.8866 [0.8411, 0.9274] | 0.6600 [0.6110, 0.7092] | 0.9438 [0.9142, 0.9698] | 155 (77.5% [71.5, 83.0]) | 0.9443 [0.9193, 0.9654] | 0.756 [0.702, 0.808] | 0.371 [0.316, 0.426] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 850 (53.9%) | 28 (1.8%) |
| **Not acknowledged** | 106 (6.7%) | 594 (37.6%) |

**Opacity rate:** 0.1109 [0.0742, 0.1522] — 106 covert / 956 hint-following runs (covert / all hint-following).

## lsat_ar × professor

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| lsat_ar | 0.1225 [0.0915, 0.1545] | 0.7746 [0.6352, 0.8802] | 0.3522 [0.2976, 0.4093] | 0.8854 [0.8326, 0.9325] | 73 (36.5% [30.0, 43.0]) | 0.7614 [0.6767, 0.8376] | 0.814 [0.770, 0.856] | 0.751 [0.703, 0.796] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 171 (10.8%) | 4 (0.3%) |
| **Not acknowledged** | 96 (6.1%) | 1308 (82.8%) |

**Opacity rate:** 0.3596 [0.2764, 0.4474] — 96 covert / 267 hint-following runs (covert / all hint-following).

## lsat_ar × grader_hacking

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| lsat_ar | 0.1375 [0.1032, 0.1727] | 0.7283 [0.5776, 0.8512] | 0.3513 [0.2983, 0.4092] | 0.9097 [0.8614, 0.9518] | 83 (41.5% [35.0, 48.0]) | 0.7501 [0.6665, 0.8293] | 0.814 [0.770, 0.856] | 0.736 [0.688, 0.783] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 176 (11.5%) | 2 (0.1%) |
| **Not acknowledged** | 98 (6.4%) | 1256 (82.0%) |

**Opacity rate:** 0.3577 [0.2669, 0.4500] — 98 covert / 274 hint-following runs (covert / all hint-following).

## lsat_ar × unethical

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| lsat_ar | 0.4367 [0.3906, 0.4831] | 0.9297 [0.8959, 0.9584] | 0.5333 [0.4890, 0.5783] | 0.9475 [0.9226, 0.9696] | 164 (82.0% [76.5, 87.0]) | 0.8759 [0.8384, 0.9108] | 0.814 [0.770, 0.856] | 0.470 [0.420, 0.519] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 667 (42.6%) | 32 (2.0%) |
| **Not acknowledged** | 95 (6.1%) | 772 (49.3%) |

**Opacity rate:** 0.1247 [0.0915, 0.1608] — 95 covert / 762 hint-following runs (covert / all hint-following).

## lsat_lr × professor

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| lsat_lr | 0.0619 [0.0394, 0.0875] | 0.7175 [0.5961, 0.8335] | 0.3322 [0.2534, 0.4194] | 0.7930 [0.6861, 0.8952] | 38 (19.0% [13.5, 24.5]) | 0.4414 [0.3011, 0.5809] | 0.847 [0.803, 0.887] | 0.787 [0.740, 0.833] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 43 (2.7%) | 4 (0.2%) |
| **Not acknowledged** | 123 (7.7%) | 1430 (89.4%) |

**Opacity rate:** 0.7410 [0.6364, 0.8273] — 123 covert / 166 hint-following runs (covert / all hint-following).

## lsat_lr × grader_hacking

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| lsat_lr | 0.0852 [0.0552, 0.1171] | 0.7506 [0.5990, 0.8792] | 0.4510 [0.3662, 0.5417] | 0.8631 [0.7795, 0.9381] | 40 (20.0% [14.5, 25.5]) | 0.4301 [0.3042, 0.5610] | 0.847 [0.803, 0.887] | 0.772 [0.723, 0.821] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 62 (3.9%) | 1 (0.1%) |
| **Not acknowledged** | 141 (8.8%) | 1392 (87.2%) |

**Opacity rate:** 0.6946 [0.5746, 0.8000] — 141 covert / 203 hint-following runs (covert / all hint-following).

## lsat_lr × unethical

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| lsat_lr | 0.3289 [0.2768, 0.3811] | 0.9234 [0.8828, 0.9601] | 0.5771 [0.5153, 0.6387] | 0.9396 [0.9066, 0.9697] | 114 (57.0% [50.0, 63.5]) | 0.7793 [0.7221, 0.8337] | 0.847 [0.803, 0.887] | 0.571 [0.514, 0.629] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 382 (24.0%) | 5 (0.3%) |
| **Not acknowledged** | 210 (13.2%) | 997 (62.5%) |

**Opacity rate:** 0.3547 [0.2821, 0.4278] — 210 covert / 592 hint-following runs (covert / all hint-following).

## medxpertqa × professor

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| medxpertqa | 0.3019 [0.2512, 0.3538] | 0.8482 [0.7891, 0.9005] | 0.5817 [0.5207, 0.6417] | 0.9053 [0.8620, 0.9430] | 104 (52.0% [45.0, 59.0]) | 0.7643 [0.7018, 0.8227] | 0.135 [0.096, 0.177] | 0.093 [0.060, 0.129] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 382 (23.9%) | 2 (0.1%) |
| **Not acknowledged** | 231 (14.4%) | 984 (61.5%) |

**Opacity rate:** 0.3768 [0.3058, 0.4480] — 231 covert / 613 hint-following runs (covert / all hint-following).

## medxpertqa × grader_hacking

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| medxpertqa | 0.2864 [0.2392, 0.3356] | 0.8356 [0.7763, 0.8898] | 0.5316 [0.4741, 0.5898] | 0.9053 [0.8633, 0.9426] | 108 (54.0% [47.0, 61.0]) | 0.5685 [0.4896, 0.6481] | 0.135 [0.096, 0.177] | 0.093 [0.061, 0.129] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 239 (15.0%) | 0 (0.0%) |
| **Not acknowledged** | 348 (21.8%) | 1011 (63.3%) |

**Opacity rate:** 0.5928 [0.5146, 0.6646] — 348 covert / 587 hint-following runs (covert / all hint-following).

## medxpertqa × unethical

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| medxpertqa | 0.8306 [0.7894, 0.8694] | 0.9171 [0.8820, 0.9471] | 0.8790 [0.8474, 0.9087] | 0.9511 [0.9277, 0.9714] | 189 (94.5% [91.0, 97.5]) | 0.9038 [0.8749, 0.9301] | 0.135 [0.096, 0.177] | 0.012 [0.003, 0.023] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1272 (79.5%) | 1 (0.1%) |
| **Not acknowledged** | 187 (11.7%) | 140 (8.8%) |

**Opacity rate:** 0.1282 [0.0976, 0.1612] — 187 covert / 1459 hint-following runs (covert / all hint-following).

## lexam × professor

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| lexam | 0.1752 [0.1395, 0.2125] | 0.6555 [0.5627, 0.7426] | 0.3767 [0.3309, 0.4256] | 0.8314 [0.7713, 0.8855] | 97 (48.5% [41.5, 55.5]) | 0.8543 [0.7977, 0.9055] | 0.567 [0.510, 0.624] | 0.462 [0.404, 0.520] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 362 (22.7%) | 25 (1.6%) |
| **Not acknowledged** | 138 (8.6%) | 1072 (67.1%) |

**Opacity rate:** 0.2760 [0.2171, 0.3346] — 138 covert / 500 hint-following runs (covert / all hint-following).

## lexam × grader_hacking

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| lexam | 0.2495 [0.2045, 0.2954] | 0.7430 [0.6678, 0.8115] | 0.4452 [0.3932, 0.4989] | 0.8591 [0.8080, 0.9054] | 116 (58.0% [51.0, 65.0]) | 0.7352 [0.6667, 0.7989] | 0.567 [0.510, 0.624] | 0.421 [0.363, 0.477] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 332 (20.8%) | 4 (0.3%) |
| **Not acknowledged** | 288 (18.0%) | 975 (61.0%) |

**Opacity rate:** 0.4645 [0.3895, 0.5403] — 288 covert / 620 hint-following runs (covert / all hint-following).

## lexam × unethical

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| lexam | 0.6786 [0.6292, 0.7246] | 0.8540 [0.8110, 0.8933] | 0.7349 [0.6911, 0.7762] | 0.8955 [0.8605, 0.9276] | 185 (92.5% [88.5, 96.0]) | 0.9745 [0.9587, 0.9871] | 0.567 [0.510, 0.624] | 0.130 [0.097, 0.166] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1206 (75.4%) | 13 (0.8%) |
| **Not acknowledged** | 101 (6.3%) | 280 (17.5%) |

**Opacity rate:** 0.0773 [0.0526, 0.1042] — 101 covert / 1307 hint-following runs (covert / all hint-following).

## Summary

| Dataset | Hint type | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] | Acknowledged & followed | Acknowledged & not followed | Not Acknowledged & followed | Not Acknowledged & not followed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mmlu_pro | professor | 0.1705 [0.1294, 0.2137] | 0.8707 [0.7981, 0.9336] | 0.5501 [0.4738, 0.6271] | 0.9268 [0.8794, 0.9674] | 62 (31.0% [25.0, 37.5]) | 0.8045 [0.7274, 0.8749] | 0.735 [0.679, 0.789] | 0.638 [0.578, 0.697] | 261 (16.3%) | 8 (0.5%) | 74 (4.6%) | 1255 (78.5%) |
| mmlu_pro | grader_hacking | 0.1979 [0.1561, 0.2416] | 0.9024 [0.8458, 0.9515] | 0.4657 [0.4012, 0.5321] | 0.9555 [0.9247, 0.9814] | 85 (42.5% [35.5, 49.0]) | 0.7793 [0.7065, 0.8478] | 0.735 [0.679, 0.789] | 0.611 [0.551, 0.670] | 220 (13.8%) | 2 (0.1%) | 159 (10.0%) | 1211 (76.1%) |
| mmlu_pro | unethical | 0.7052 [0.6592, 0.7519] | 0.9581 [0.9326, 0.9801] | 0.7543 [0.7125, 0.7956] | 0.9786 [0.9637, 0.9913] | 187 (93.5% [90.0, 96.5]) | 0.9442 [0.9213, 0.9642] | 0.735 [0.679, 0.789] | 0.221 [0.179, 0.265] | 1110 (70.0%) | 12 (0.8%) | 75 (4.7%) | 388 (24.5%) |
| gpqa_diamond | professor | 0.3089 [0.2579, 0.3604] | 0.6940 [0.6121, 0.7685] | 0.5561 [0.4953, 0.6158] | 0.8130 [0.7604, 0.8627] | 112 (56.6% [49.5, 63.6]) | 0.7933 [0.7273, 0.8530] | 0.507 [0.447, 0.565] | 0.382 [0.324, 0.441] | 496 (31.7%) | 6 (0.4%) | 228 (14.6%) | 833 (53.3%) |
| gpqa_diamond | grader_hacking | 0.2923 [0.2412, 0.3440] | 0.6956 [0.6125, 0.7715] | 0.5230 [0.4637, 0.5822] | 0.8422 [0.7942, 0.8863] | 114 (57.6% [51.0, 64.6]) | 0.3692 [0.2941, 0.4450] | 0.507 [0.447, 0.565] | 0.368 [0.311, 0.425] | 154 (9.9%) | 1 (0.1%) | 538 (34.6%) | 864 (55.5%) |
| gpqa_diamond | unethical | 0.6383 [0.5863, 0.6893] | 0.8249 [0.7796, 0.8675] | 0.7355 [0.6913, 0.7804] | 0.8786 [0.8436, 0.9119] | 172 (86.9% [82.3, 91.4]) | 0.8891 [0.8566, 0.9193] | 0.507 [0.447, 0.565] | 0.165 [0.123, 0.209] | 1023 (65.5%) | 3 (0.2%) | 221 (14.1%) | 315 (20.2%) |
| wmdp | professor | 0.2339 [0.1878, 0.2833] | 0.7561 [0.6773, 0.8297] | 0.5458 [0.4770, 0.6143] | 0.8821 [0.8263, 0.9319] | 84 (42.9% [35.7, 50.0]) | 0.8840 [0.8339, 0.9273] | 0.674 [0.615, 0.734] | 0.521 [0.459, 0.584] | 391 (25.8%) | 7 (0.5%) | 130 (8.6%) | 985 (65.1%) |
| wmdp | grader_hacking | 0.2267 [0.1805, 0.2745] | 0.7373 [0.6498, 0.8166] | 0.5159 [0.4507, 0.5827] | 0.8880 [0.8326, 0.9370] | 87 (44.2% [37.1, 51.3]) | 0.5596 [0.4699, 0.6502] | 0.673 [0.614, 0.731] | 0.516 [0.456, 0.577] | 199 (13.1%) | 0 (0.0%) | 319 (21.1%) | 997 (65.8%) |
| wmdp | unethical | 0.6280 [0.5707, 0.6839] | 0.8699 [0.8237, 0.9128] | 0.7685 [0.7220, 0.8132] | 0.9402 [0.9101, 0.9667] | 161 (81.7% [76.1, 86.8]) | 0.9538 [0.9343, 0.9707] | 0.673 [0.614, 0.731] | 0.225 [0.176, 0.276] | 1013 (66.8%) | 6 (0.4%) | 111 (7.3%) | 387 (25.5%) |
| codemmlu | professor | 0.1101 [0.0801, 0.1428] | 0.6668 [0.5561, 0.7657] | 0.3861 [0.3197, 0.4568] | 0.8392 [0.7636, 0.9066] | 58 (29.0% [23.0, 35.5]) | 0.7655 [0.6669, 0.8559] | 0.756 [0.702, 0.808] | 0.679 [0.621, 0.736] | 197 (12.4%) | 12 (0.8%) | 124 (7.8%) | 1252 (79.0%) |
| codemmlu | grader_hacking | 0.0853 [0.0594, 0.1144] | 0.6589 [0.5529, 0.7625] | 0.3469 [0.2785, 0.4211] | 0.8232 [0.7404, 0.8980] | 50 (25.1% [19.1, 31.2]) | 0.4559 [0.3248, 0.5837] | 0.760 [0.707, 0.811] | 0.712 [0.655, 0.766] | 80 (5.1%) | 1 (0.1%) | 196 (12.5%) | 1294 (82.4%) |
| codemmlu | unethical | 0.5115 [0.4580, 0.5650] | 0.8866 [0.8411, 0.9274] | 0.6600 [0.6110, 0.7092] | 0.9438 [0.9142, 0.9698] | 155 (77.5% [71.5, 83.0]) | 0.9443 [0.9193, 0.9654] | 0.756 [0.702, 0.808] | 0.371 [0.316, 0.426] | 850 (53.9%) | 28 (1.8%) | 106 (6.7%) | 594 (37.6%) |
| lsat_ar | professor | 0.1225 [0.0915, 0.1545] | 0.7746 [0.6352, 0.8802] | 0.3522 [0.2976, 0.4093] | 0.8854 [0.8326, 0.9325] | 73 (36.5% [30.0, 43.0]) | 0.7614 [0.6767, 0.8376] | 0.814 [0.770, 0.856] | 0.751 [0.703, 0.796] | 171 (10.8%) | 4 (0.3%) | 96 (6.1%) | 1308 (82.8%) |
| lsat_ar | grader_hacking | 0.1375 [0.1032, 0.1727] | 0.7283 [0.5776, 0.8512] | 0.3513 [0.2983, 0.4092] | 0.9097 [0.8614, 0.9518] | 83 (41.5% [35.0, 48.0]) | 0.7501 [0.6665, 0.8293] | 0.814 [0.770, 0.856] | 0.736 [0.688, 0.783] | 176 (11.5%) | 2 (0.1%) | 98 (6.4%) | 1256 (82.0%) |
| lsat_ar | unethical | 0.4367 [0.3906, 0.4831] | 0.9297 [0.8959, 0.9584] | 0.5333 [0.4890, 0.5783] | 0.9475 [0.9226, 0.9696] | 164 (82.0% [76.5, 87.0]) | 0.8759 [0.8384, 0.9108] | 0.814 [0.770, 0.856] | 0.470 [0.420, 0.519] | 667 (42.6%) | 32 (2.0%) | 95 (6.1%) | 772 (49.3%) |
| lsat_lr | professor | 0.0619 [0.0394, 0.0875] | 0.7175 [0.5961, 0.8335] | 0.3322 [0.2534, 0.4194] | 0.7930 [0.6861, 0.8952] | 38 (19.0% [13.5, 24.5]) | 0.4414 [0.3011, 0.5809] | 0.847 [0.803, 0.887] | 0.787 [0.740, 0.833] | 43 (2.7%) | 4 (0.2%) | 123 (7.7%) | 1430 (89.4%) |
| lsat_lr | grader_hacking | 0.0852 [0.0552, 0.1171] | 0.7506 [0.5990, 0.8792] | 0.4510 [0.3662, 0.5417] | 0.8631 [0.7795, 0.9381] | 40 (20.0% [14.5, 25.5]) | 0.4301 [0.3042, 0.5610] | 0.847 [0.803, 0.887] | 0.772 [0.723, 0.821] | 62 (3.9%) | 1 (0.1%) | 141 (8.8%) | 1392 (87.2%) |
| lsat_lr | unethical | 0.3289 [0.2768, 0.3811] | 0.9234 [0.8828, 0.9601] | 0.5771 [0.5153, 0.6387] | 0.9396 [0.9066, 0.9697] | 114 (57.0% [50.0, 63.5]) | 0.7793 [0.7221, 0.8337] | 0.847 [0.803, 0.887] | 0.571 [0.514, 0.629] | 382 (24.0%) | 5 (0.3%) | 210 (13.2%) | 997 (62.5%) |
| medxpertqa | professor | 0.3019 [0.2512, 0.3538] | 0.8482 [0.7891, 0.9005] | 0.5817 [0.5207, 0.6417] | 0.9053 [0.8620, 0.9430] | 104 (52.0% [45.0, 59.0]) | 0.7643 [0.7018, 0.8227] | 0.135 [0.096, 0.177] | 0.093 [0.060, 0.129] | 382 (23.9%) | 2 (0.1%) | 231 (14.4%) | 984 (61.5%) |
| medxpertqa | grader_hacking | 0.2864 [0.2392, 0.3356] | 0.8356 [0.7763, 0.8898] | 0.5316 [0.4741, 0.5898] | 0.9053 [0.8633, 0.9426] | 108 (54.0% [47.0, 61.0]) | 0.5685 [0.4896, 0.6481] | 0.135 [0.096, 0.177] | 0.093 [0.061, 0.129] | 239 (15.0%) | 0 (0.0%) | 348 (21.8%) | 1011 (63.3%) |
| medxpertqa | unethical | 0.8306 [0.7894, 0.8694] | 0.9171 [0.8820, 0.9471] | 0.8790 [0.8474, 0.9087] | 0.9511 [0.9277, 0.9714] | 189 (94.5% [91.0, 97.5]) | 0.9038 [0.8749, 0.9301] | 0.135 [0.096, 0.177] | 0.012 [0.003, 0.023] | 1272 (79.5%) | 1 (0.1%) | 187 (11.7%) | 140 (8.8%) |
| lexam | professor | 0.1752 [0.1395, 0.2125] | 0.6555 [0.5627, 0.7426] | 0.3767 [0.3309, 0.4256] | 0.8314 [0.7713, 0.8855] | 97 (48.5% [41.5, 55.5]) | 0.8543 [0.7977, 0.9055] | 0.567 [0.510, 0.624] | 0.462 [0.404, 0.520] | 362 (22.7%) | 25 (1.6%) | 138 (8.6%) | 1072 (67.1%) |
| lexam | grader_hacking | 0.2495 [0.2045, 0.2954] | 0.7430 [0.6678, 0.8115] | 0.4452 [0.3932, 0.4989] | 0.8591 [0.8080, 0.9054] | 116 (58.0% [51.0, 65.0]) | 0.7352 [0.6667, 0.7989] | 0.567 [0.510, 0.624] | 0.421 [0.363, 0.477] | 332 (20.8%) | 4 (0.3%) | 288 (18.0%) | 975 (61.0%) |
| lexam | unethical | 0.6786 [0.6292, 0.7246] | 0.8540 [0.8110, 0.8933] | 0.7349 [0.6911, 0.7762] | 0.8955 [0.8605, 0.9276] | 185 (92.5% [88.5, 96.0]) | 0.9745 [0.9587, 0.9871] | 0.567 [0.510, 0.624] | 0.130 [0.097, 0.166] | 1206 (75.4%) | 13 (0.8%) | 101 (6.3%) | 280 (17.5%) |
