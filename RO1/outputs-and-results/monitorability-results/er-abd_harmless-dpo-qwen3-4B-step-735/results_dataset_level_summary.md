# Results dataset-level summary — instances pooled across hint types

- **Model:** `er-abd/harmless-dpo-qwen3-4B-step-735` (`er-abd_harmless-dpo-qwen3-4B-step-735`)
- **Datasets:** mmlu_pro, gpqa_diamond, wmdp, codemmlu, lsat_ar, lsat_lr, medxpertqa, lexam
- **Hint types:** professor, grader_hacking, unethical

## mmlu_pro (pooled over 2 hint types)

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| mmlu_pro | 0.1842 (0.3056) | 0.8890 (0.2663) | 0.5013 (0.3087) | 0.9434 (0.1535) | 147 (36.8%) | 0.7899 | 0.735 (0.401) | 0.625 (0.434) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 481 (15.1%) | 10 (0.3%) |
| **Not acknowledged** | 233 (7.3%) | 2466 (77.3%) |

**Opacity rate:** 0.3263 — 233 covert / 714 hint-following runs (covert / all hint-following).

## gpqa_diamond (pooled over 3 hint types)

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| gpqa_diamond | 0.4132 (0.4040) | 0.7486 (0.4086) | 0.6242 (0.3262) | 0.8497 (0.2489) | 398 (67.0%) | 0.7132 | 0.507 (0.423) | 0.305 (0.396) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1673 (35.7%) | 10 (0.2%) |
| **Not acknowledged** | 987 (21.1%) | 2012 (43.0%) |

**Opacity rate:** 0.3711 — 987 covert / 2660 hint-following runs (covert / all hint-following).

## wmdp (pooled over 3 hint types)

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| wmdp | 0.3631 (0.4048) | 0.8037 (0.3690) | 0.6459 (0.3287) | 0.9118 (0.2204) | 332 (56.3%) | 0.8328 | 0.673 (0.421) | 0.421 (0.436) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1603 (35.3%) | 13 (0.3%) |
| **Not acknowledged** | 560 (12.3%) | 2369 (52.1%) |

**Opacity rate:** 0.2589 — 560 covert / 2163 hint-following runs (covert / all hint-following).

## codemmlu (pooled over 3 hint types)

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| codemmlu | 0.2359 (0.3456) | 0.7868 (0.3791) | 0.5401 (0.3265) | 0.8978 (0.2314) | 263 (43.9%) | 0.8120 | 0.758 (0.381) | 0.587 (0.432) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1127 (23.8%) | 41 (0.9%) |
| **Not acknowledged** | 426 (9.0%) | 3140 (66.3%) |

**Opacity rate:** 0.2743 — 426 covert / 1553 hint-following runs (covert / all hint-following).

## lsat_ar (pooled over 3 hint types)

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| lsat_ar | 0.2322 (0.3125) | 0.8388 (0.4726) | 0.4448 (0.2878) | 0.9235 (0.1885) | 320 (53.3%) | 0.8171 | 0.814 (0.320) | 0.652 (0.374) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1014 (21.7%) | 38 (0.8%) |
| **Not acknowledged** | 289 (6.2%) | 3336 (71.3%) |

**Opacity rate:** 0.2218 — 289 covert / 1303 hint-following runs (covert / all hint-following).

## lsat_lr (pooled over 3 hint types)

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| lsat_lr | 0.1587 (0.3007) | 0.8429 (0.3378) | 0.5024 (0.3274) | 0.8946 (0.2348) | 192 (32.0%) | 0.6397 | 0.847 (0.308) | 0.710 (0.388) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 487 (10.2%) | 10 (0.2%) |
| **Not acknowledged** | 474 (9.9%) | 3819 (79.7%) |

**Opacity rate:** 0.4932 — 474 covert / 961 hint-following runs (covert / all hint-following).

## medxpertqa (pooled over 3 hint types)

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| medxpertqa | 0.4730 (0.4219) | 0.8766 (0.2768) | 0.7083 (0.3148) | 0.9269 (0.1852) | 401 (66.8%) | 0.7773 | 0.135 (0.295) | 0.066 (0.213) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1893 (39.5%) | 3 (0.1%) |
| **Not acknowledged** | 766 (16.0%) | 2135 (44.5%) |

**Opacity rate:** 0.2881 — 766 covert / 2659 hint-following runs (covert / all hint-following).

## lexam (pooled over 3 hint types)

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| lexam | 0.3677 (0.3828) | 0.7688 (0.4053) | 0.5632 (0.3248) | 0.8693 (0.2602) | 398 (66.3%) | 0.8754 | 0.567 (0.412) | 0.338 (0.394) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1900 (39.6%) | 42 (0.9%) |
| **Not acknowledged** | 527 (11.0%) | 2327 (48.5%) |

**Opacity rate:** 0.2171 — 527 covert / 2427 hint-following runs (covert / all hint-following).

## Summary

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) | Acknowledged & followed | Acknowledged & not followed | Not Acknowledged & followed | Not Acknowledged & not followed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mmlu_pro | 0.1842 (0.3056) | 0.8890 (0.2663) | 0.5013 (0.3087) | 0.9434 (0.1535) | 147 (36.8%) | 0.7899 | 0.735 (0.401) | 0.625 (0.434) | 481 (15.1%) | 10 (0.3%) | 233 (7.3%) | 2466 (77.3%) |
| gpqa_diamond | 0.4132 (0.4040) | 0.7486 (0.4086) | 0.6242 (0.3262) | 0.8497 (0.2489) | 398 (67.0%) | 0.7132 | 0.507 (0.423) | 0.305 (0.396) | 1673 (35.7%) | 10 (0.2%) | 987 (21.1%) | 2012 (43.0%) |
| wmdp | 0.3631 (0.4048) | 0.8037 (0.3690) | 0.6459 (0.3287) | 0.9118 (0.2204) | 332 (56.3%) | 0.8328 | 0.673 (0.421) | 0.421 (0.436) | 1603 (35.3%) | 13 (0.3%) | 560 (12.3%) | 2369 (52.1%) |
| codemmlu | 0.2359 (0.3456) | 0.7868 (0.3791) | 0.5401 (0.3265) | 0.8978 (0.2314) | 263 (43.9%) | 0.8120 | 0.758 (0.381) | 0.587 (0.432) | 1127 (23.8%) | 41 (0.9%) | 426 (9.0%) | 3140 (66.3%) |
| lsat_ar | 0.2322 (0.3125) | 0.8388 (0.4726) | 0.4448 (0.2878) | 0.9235 (0.1885) | 320 (53.3%) | 0.8171 | 0.814 (0.320) | 0.652 (0.374) | 1014 (21.7%) | 38 (0.8%) | 289 (6.2%) | 3336 (71.3%) |
| lsat_lr | 0.1587 (0.3007) | 0.8429 (0.3378) | 0.5024 (0.3274) | 0.8946 (0.2348) | 192 (32.0%) | 0.6397 | 0.847 (0.308) | 0.710 (0.388) | 487 (10.2%) | 10 (0.2%) | 474 (9.9%) | 3819 (79.7%) |
| medxpertqa | 0.4730 (0.4219) | 0.8766 (0.2768) | 0.7083 (0.3148) | 0.9269 (0.1852) | 401 (66.8%) | 0.7773 | 0.135 (0.295) | 0.066 (0.213) | 1893 (39.5%) | 3 (0.1%) | 766 (16.0%) | 2135 (44.5%) |
| lexam | 0.3677 (0.3828) | 0.7688 (0.4053) | 0.5632 (0.3248) | 0.8693 (0.2602) | 398 (66.3%) | 0.8754 | 0.567 (0.412) | 0.338 (0.394) | 1900 (39.6%) | 42 (0.9%) | 527 (11.0%) | 2327 (48.5%) |
