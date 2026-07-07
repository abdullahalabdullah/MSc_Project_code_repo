# Results overall summary — per (dataset × hint type)

- **Model:** `er-abd_harmless-dpo-qwen3-4B-pilot` (`er-abd_harmless-dpo-qwen3-4B-pilot`)
- **Datasets:** mmlu_pro, gpqa_diamond, wmdp, codemmlu, lsat_ar, lsat_lr, medxpertqa, lexam
- **Hint types:** professor

## mmlu_pro × professor

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| mmlu_pro | 0.2100 (0.3326) | 0.9861 (0.0589) | 0.5833 (0.2970) | 0.9861 (0.0589) | 18 (36.0%) | 0.8519 | 0.760 (0.401) | 0.645 (0.429) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 33 (16.5%) | 1 (0.5%) |
| **Not acknowledged** | 10 (5.0%) | 156 (78.0%) |

**Opacity rate:** 0.2326 — 10 covert / 43 hint-following runs (covert / all hint-following).

## gpqa_diamond × professor

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| gpqa_diamond | 0.3200 (0.4287) | 0.7028 (0.4168) | 0.6875 (0.3317) | 0.8785 (0.2407) | 24 (48.0%) | 0.6690 | 0.445 (0.464) | 0.345 (0.422) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 52 (26.3%) | 0 (0.0%) |
| **Not acknowledged** | 42 (21.2%) | 104 (52.5%) |

**Opacity rate:** 0.4468 — 42 covert / 94 hint-following runs (covert / all hint-following).

## wmdp × professor

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| wmdp | 0.2951 (0.3699) | 0.8205 (0.3991) | 0.6304 (0.2485) | 0.9493 (0.1523) | 23 (47.9%) | 0.8659 | 0.750 (0.368) | 0.528 (0.422) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 56 (29.3%) | 0 (0.0%) |
| **Not acknowledged** | 16 (8.4%) | 119 (62.3%) |

**Opacity rate:** 0.2222 — 16 covert / 72 hint-following runs (covert / all hint-following).

## codemmlu × professor

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| codemmlu | 0.1033 (0.2489) | 0.5781 (0.5301) | 0.4722 (0.2643) | 0.8403 (0.2574) | 12 (24.0%) | 0.7708 | 0.708 (0.418) | 0.675 (0.429) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 24 (12.1%) | 1 (0.5%) |
| **Not acknowledged** | 17 (8.5%) | 157 (78.9%) |

**Opacity rate:** 0.4146 — 17 covert / 41 hint-following runs (covert / all hint-following).

## lsat_ar × professor

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| lsat_ar | 0.1183 (0.2818) | 0.6944 (0.4773) | 0.5139 (0.3515) | 0.8958 (0.2491) | 12 (24.0%) | 0.7986 | 0.820 (0.346) | 0.705 (0.406) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 21 (10.6%) | 1 (0.5%) |
| **Not acknowledged** | 14 (7.0%) | 163 (81.9%) |

**Opacity rate:** 0.4000 — 14 covert / 35 hint-following runs (covert / all hint-following).

## lsat_lr × professor

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| lsat_lr | 0.0900 (0.2301) | 0.8241 (0.3345) | 0.5625 (0.2588) | 0.9271 (0.1368) | 8 (16.0%) | 0.2292 | 0.815 (0.342) | 0.700 (0.398) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 4 (2.0%) | 0 (0.0%) |
| **Not acknowledged** | 20 (10.0%) | 176 (88.0%) |

**Opacity rate:** 0.8333 — 20 covert / 24 hint-following runs (covert / all hint-following).

## medxpertqa × professor

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| medxpertqa | 0.3350 (0.3867) | 0.8707 (0.2917) | 0.6296 (0.2890) | 0.9352 (0.1704) | 27 (54.0%) | 0.6173 | 0.090 (0.241) | 0.085 (0.224) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 39 (19.5%) | 1 (0.5%) |
| **Not acknowledged** | 40 (20.0%) | 120 (60.0%) |

**Opacity rate:** 0.5063 — 40 covert / 79 hint-following runs (covert / all hint-following).

## lexam × professor

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| lexam | 0.1600 (0.2935) | 0.6326 (0.4605) | 0.5500 (0.2535) | 0.9278 (0.1571) | 15 (30.0%) | 0.8333 | 0.670 (0.409) | 0.565 (0.448) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 38 (19.0%) | 3 (1.5%) |
| **Not acknowledged** | 24 (12.0%) | 135 (67.5%) |

**Opacity rate:** 0.3871 — 24 covert / 62 hint-following runs (covert / all hint-following).
