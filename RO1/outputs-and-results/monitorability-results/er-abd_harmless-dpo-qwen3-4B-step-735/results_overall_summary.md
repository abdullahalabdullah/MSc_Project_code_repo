# Results overall summary — per (dataset × hint type)

- **Model:** `er-abd/harmless-dpo-qwen3-4B-step-735` (`er-abd_harmless-dpo-qwen3-4B-step-735`)
- **Datasets:** mmlu_pro, gpqa_diamond, wmdp, codemmlu, lsat_ar, lsat_lr, medxpertqa, lexam
- **Hint types:** professor, unethical

## mmlu_pro × professor

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| mmlu_pro | 0.1705 (0.3065) | 0.8707 (0.2808) | 0.5501 (0.3071) | 0.9268 (0.1764) | 62 (31.0%) | 0.8045 | 0.735 (0.402) | 0.638 (0.437) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 261 (16.3%) | 8 (0.5%) |
| **Not acknowledged** | 74 (4.6%) | 1255 (78.5%) |

**Opacity rate:** 0.2209 — 74 covert / 335 hint-following runs (covert / all hint-following).

## gpqa_diamond × professor

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| gpqa_diamond | 0.3089 (0.3745) | 0.6940 (0.4489) | 0.5561 (0.3244) | 0.8130 (0.2734) | 112 (56.6%) | 0.7933 | 0.507 (0.424) | 0.382 (0.420) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 496 (31.7%) | 6 (0.4%) |
| **Not acknowledged** | 228 (14.6%) | 833 (53.3%) |

**Opacity rate:** 0.3149 — 228 covert / 724 hint-following runs (covert / all hint-following).

## gpqa_diamond × unethical

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| gpqa_diamond | 0.6383 (0.3707) | 0.8249 (0.3067) | 0.7355 (0.2930) | 0.8786 (0.2273) | 172 (86.9%) | 0.8891 | 0.507 (0.424) | 0.165 (0.307) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1023 (65.5%) | 3 (0.2%) |
| **Not acknowledged** | 221 (14.1%) | 315 (20.2%) |

**Opacity rate:** 0.1777 — 221 covert / 1244 hint-following runs (covert / all hint-following).

## wmdp × professor

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| wmdp | 0.2339 (0.3421) | 0.7561 (0.3864) | 0.5458 (0.3206) | 0.8821 (0.2490) | 84 (42.9%) | 0.8840 | 0.674 (0.422) | 0.521 (0.445) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 391 (25.8%) | 7 (0.5%) |
| **Not acknowledged** | 130 (8.6%) | 985 (65.1%) |

**Opacity rate:** 0.2495 — 130 covert / 521 hint-following runs (covert / all hint-following).

## wmdp × unethical

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| wmdp | 0.6280 (0.3995) | 0.8699 (0.3047) | 0.7685 (0.2949) | 0.9402 (0.1841) | 161 (81.7%) | 0.9538 | 0.673 (0.422) | 0.225 (0.353) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1013 (66.8%) | 6 (0.4%) |
| **Not acknowledged** | 111 (7.3%) | 387 (25.5%) |

**Opacity rate:** 0.0988 — 111 covert / 1124 hint-following runs (covert / all hint-following).

## codemmlu × professor

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| codemmlu | 0.1101 (0.2284) | 0.6668 (0.4592) | 0.3861 (0.2675) | 0.8392 (0.2768) | 58 (29.0%) | 0.7655 | 0.756 (0.382) | 0.679 (0.414) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 197 (12.4%) | 12 (0.8%) |
| **Not acknowledged** | 124 (7.8%) | 1252 (79.0%) |

**Opacity rate:** 0.3863 — 124 covert / 321 hint-following runs (covert / all hint-following).

## codemmlu × unethical

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| codemmlu | 0.5115 (0.3904) | 0.8866 (0.2844) | 0.6600 (0.3135) | 0.9438 (0.1783) | 155 (77.5%) | 0.9443 | 0.756 (0.382) | 0.371 (0.399) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 850 (53.9%) | 28 (1.8%) |
| **Not acknowledged** | 106 (6.7%) | 594 (37.6%) |

**Opacity rate:** 0.1109 — 106 covert / 956 hint-following runs (covert / all hint-following).

## lsat_ar × professor

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| lsat_ar | 0.1225 (0.2304) | 0.7746 (0.5530) | 0.3522 (0.2408) | 0.8854 (0.2207) | 73 (36.5%) | 0.7614 | 0.814 (0.320) | 0.751 (0.341) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 171 (10.8%) | 4 (0.3%) |
| **Not acknowledged** | 96 (6.1%) | 1308 (82.8%) |

**Opacity rate:** 0.3596 — 96 covert / 267 hint-following runs (covert / all hint-following).

## lsat_ar × unethical

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| lsat_ar | 0.4367 (0.3381) | 0.9297 (0.2032) | 0.5333 (0.2954) | 0.9475 (0.1551) | 164 (82.0%) | 0.8759 | 0.814 (0.320) | 0.470 (0.368) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 667 (42.6%) | 32 (2.0%) |
| **Not acknowledged** | 95 (6.1%) | 772 (49.3%) |

**Opacity rate:** 0.1247 — 95 covert / 762 hint-following runs (covert / all hint-following).

## lsat_lr × professor

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| lsat_lr | 0.0619 (0.1741) | 0.7175 (0.3901) | 0.3322 (0.2636) | 0.7930 (0.3272) | 38 (19.0%) | 0.4414 | 0.847 (0.308) | 0.787 (0.341) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 43 (2.7%) | 4 (0.2%) |
| **Not acknowledged** | 123 (7.7%) | 1430 (89.4%) |

**Opacity rate:** 0.7410 — 123 covert / 166 hint-following runs (covert / all hint-following).

## lsat_lr × unethical

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| lsat_lr | 0.3289 (0.3829) | 0.9234 (0.2111) | 0.5771 (0.3372) | 0.9396 (0.1732) | 114 (57.0%) | 0.7793 | 0.847 (0.308) | 0.571 (0.424) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 382 (24.0%) | 5 (0.3%) |
| **Not acknowledged** | 210 (13.2%) | 997 (62.5%) |

**Opacity rate:** 0.3547 — 210 covert / 592 hint-following runs (covert / all hint-following).

## medxpertqa × professor

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| medxpertqa | 0.3019 (0.3684) | 0.8482 (0.2988) | 0.5817 (0.3120) | 0.9053 (0.2077) | 104 (52.0%) | 0.7643 | 0.135 (0.296) | 0.093 (0.251) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 382 (23.9%) | 2 (0.1%) |
| **Not acknowledged** | 231 (14.4%) | 984 (61.5%) |

**Opacity rate:** 0.3768 — 231 covert / 613 hint-following runs (covert / all hint-following).

## medxpertqa × unethical

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| medxpertqa | 0.8306 (0.2901) | 0.9171 (0.2320) | 0.8790 (0.2153) | 0.9511 (0.1528) | 189 (94.5%) | 0.9038 | 0.135 (0.296) | 0.012 (0.072) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1272 (79.5%) | 1 (0.1%) |
| **Not acknowledged** | 187 (11.7%) | 140 (8.8%) |

**Opacity rate:** 0.1282 — 187 covert / 1459 hint-following runs (covert / all hint-following).

## lexam × professor

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| lexam | 0.1752 (0.2586) | 0.6555 (0.4981) | 0.3767 (0.2379) | 0.8314 (0.2908) | 97 (48.5%) | 0.8543 | 0.567 (0.413) | 0.462 (0.415) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 362 (22.7%) | 25 (1.6%) |
| **Not acknowledged** | 138 (8.6%) | 1072 (67.1%) |

**Opacity rate:** 0.2760 — 138 covert / 500 hint-following runs (covert / all hint-following).

## lexam × unethical

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| lexam | 0.6786 (0.3449) | 0.8540 (0.2970) | 0.7349 (0.2928) | 0.8955 (0.2348) | 185 (92.5%) | 0.9745 | 0.567 (0.413) | 0.130 (0.248) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1206 (75.4%) | 13 (0.8%) |
| **Not acknowledged** | 101 (6.3%) | 280 (17.5%) |

**Opacity rate:** 0.0773 — 101 covert / 1307 hint-following runs (covert / all hint-following).

## Summary

| Dataset      | Hint type | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) | Acknowledged & followed | Acknowledged & not followed | Not Acknowledged & followed | Not Acknowledged & not followed |
| ------------ | --------- | -------------------- | -------------------------- | ------------------- | ------------------------- | ---------------- | ------- | ------------------ | ---------------- | ----------------------- | --------------------------- | --------------------------- | ------------------------------- |
| mmlu_pro     | professor | 0.1705 (0.3065)      | 0.8707 (0.2808)            | 0.5501 (0.3071)     | 0.9268 (0.1764)           | 62 (31.0%)       | 0.8045  | 0.735 (0.402)      | 0.638 (0.437)    | 261 (16.3%)             | 8 (0.5%)                    | 74 (4.6%)                   | 1255 (78.5%)                    |
| gpqa_diamond | professor | 0.3089 (0.3745)      | 0.6940 (0.4489)            | 0.5561 (0.3244)     | 0.8130 (0.2734)           | 112 (56.6%)      | 0.7933  | 0.507 (0.424)      | 0.382 (0.420)    | 496 (31.7%)             | 6 (0.4%)                    | 228 (14.6%)                 | 833 (53.3%)                     |
| gpqa_diamond | unethical | 0.6383 (0.3707)      | 0.8249 (0.3067)            | 0.7355 (0.2930)     | 0.8786 (0.2273)           | 172 (86.9%)      | 0.8891  | 0.507 (0.424)      | 0.165 (0.307)    | 1023 (65.5%)            | 3 (0.2%)                    | 221 (14.1%)                 | 315 (20.2%)                     |
| wmdp         | professor | 0.2339 (0.3421)      | 0.7561 (0.3864)            | 0.5458 (0.3206)     | 0.8821 (0.2490)           | 84 (42.9%)       | 0.8840  | 0.674 (0.422)      | 0.521 (0.445)    | 391 (25.8%)             | 7 (0.5%)                    | 130 (8.6%)                  | 985 (65.1%)                     |
| wmdp         | unethical | 0.6280 (0.3995)      | 0.8699 (0.3047)            | 0.7685 (0.2949)     | 0.9402 (0.1841)           | 161 (81.7%)      | 0.9538  | 0.673 (0.422)      | 0.225 (0.353)    | 1013 (66.8%)            | 6 (0.4%)                    | 111 (7.3%)                  | 387 (25.5%)                     |
| codemmlu     | professor | 0.1101 (0.2284)      | 0.6668 (0.4592)            | 0.3861 (0.2675)     | 0.8392 (0.2768)           | 58 (29.0%)       | 0.7655  | 0.756 (0.382)      | 0.679 (0.414)    | 197 (12.4%)             | 12 (0.8%)                   | 124 (7.8%)                  | 1252 (79.0%)                    |
| codemmlu     | unethical | 0.5115 (0.3904)      | 0.8866 (0.2844)            | 0.6600 (0.3135)     | 0.9438 (0.1783)           | 155 (77.5%)      | 0.9443  | 0.756 (0.382)      | 0.371 (0.399)    | 850 (53.9%)             | 28 (1.8%)                   | 106 (6.7%)                  | 594 (37.6%)                     |
| lsat_ar      | professor | 0.1225 (0.2304)      | 0.7746 (0.5530)            | 0.3522 (0.2408)     | 0.8854 (0.2207)           | 73 (36.5%)       | 0.7614  | 0.814 (0.320)      | 0.751 (0.341)    | 171 (10.8%)             | 4 (0.3%)                    | 96 (6.1%)                   | 1308 (82.8%)                    |
| lsat_ar      | unethical | 0.4367 (0.3381)      | 0.9297 (0.2032)            | 0.5333 (0.2954)     | 0.9475 (0.1551)           | 164 (82.0%)      | 0.8759  | 0.814 (0.320)      | 0.470 (0.368)    | 667 (42.6%)             | 32 (2.0%)                   | 95 (6.1%)                   | 772 (49.3%)                     |
| lsat_lr      | professor | 0.0619 (0.1741)      | 0.7175 (0.3901)            | 0.3322 (0.2636)     | 0.7930 (0.3272)           | 38 (19.0%)       | 0.4414  | 0.847 (0.308)      | 0.787 (0.341)    | 43 (2.7%)               | 4 (0.2%)                    | 123 (7.7%)                  | 1430 (89.4%)                    |
| lsat_lr      | unethical | 0.3289 (0.3829)      | 0.9234 (0.2111)            | 0.5771 (0.3372)     | 0.9396 (0.1732)           | 114 (57.0%)      | 0.7793  | 0.847 (0.308)      | 0.571 (0.424)    | 382 (24.0%)             | 5 (0.3%)                    | 210 (13.2%)                 | 997 (62.5%)                     |
| medxpertqa   | professor | 0.3019 (0.3684)      | 0.8482 (0.2988)            | 0.5817 (0.3120)     | 0.9053 (0.2077)           | 104 (52.0%)      | 0.7643  | 0.135 (0.296)      | 0.093 (0.251)    | 382 (23.9%)             | 2 (0.1%)                    | 231 (14.4%)                 | 984 (61.5%)                     |
| medxpertqa   | unethical | 0.8306 (0.2901)      | 0.9171 (0.2320)            | 0.8790 (0.2153)     | 0.9511 (0.1528)           | 189 (94.5%)      | 0.9038  | 0.135 (0.296)      | 0.012 (0.072)    | 1272 (79.5%)            | 1 (0.1%)                    | 187 (11.7%)                 | 140 (8.8%)                      |
| lexam        | professor | 0.1752 (0.2586)      | 0.6555 (0.4981)            | 0.3767 (0.2379)     | 0.8314 (0.2908)           | 97 (48.5%)       | 0.8543  | 0.567 (0.413)      | 0.462 (0.415)    | 362 (22.7%)             | 25 (1.6%)                   | 138 (8.6%)                  | 1072 (67.1%)                    |
| lexam        | unethical | 0.6786 (0.3449)      | 0.8540 (0.2970)            | 0.7349 (0.2928)     | 0.8955 (0.2348)           | 185 (92.5%)      | 0.9745  | 0.567 (0.413)      | 0.130 (0.248)    | 1206 (75.4%)            | 13 (0.8%)                   | 101 (6.3%)                  | 280 (17.5%)                     |
