# Results overall summary — per (dataset × hint type)

- **Model:** `er-abd_harmless-dpo-qwen3-4B-pilot` (`er-abd_harmless-dpo-qwen3-4B-pilot`)
- **Datasets:** mmlu_pro, gpqa_diamond, wmdp, codemmlu
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
