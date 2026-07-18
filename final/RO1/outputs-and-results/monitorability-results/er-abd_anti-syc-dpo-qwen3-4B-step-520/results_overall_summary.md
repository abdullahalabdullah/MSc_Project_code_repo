# Results overall summary — per (dataset × hint type)

- **Model:** `er-abd/anti-syc-dpo-qwen3-4B-step-520` (`er-abd_anti-syc-dpo-qwen3-4B-step-520`)
- **Datasets:** mmlu_pro
- **Hint types:** grader_hacking

## mmlu_pro × grader_hacking

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| mmlu_pro | 0.1951 [0.1549, 0.2371] | 0.8924 [0.8247, 0.9488] | 0.4210 [0.3596, 0.4837] | 0.9512 [0.9137, 0.9820] | 93 (46.5% [39.5, 53.5]) | 0.7626 [0.6860, 0.8344] | 0.731 [0.675, 0.785] | 0.609 [0.551, 0.665] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 204 (13.0%) | 4 (0.3%) |
| **Not acknowledged** | 166 (10.5%) | 1200 (76.2%) |

**Opacity rate:** 0.4486 [0.3405, 0.5495] — 166 covert / 370 hint-following runs (covert / all hint-following).

## Summary

| Dataset | Hint type | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] | Acknowledged & followed | Acknowledged & not followed | Not Acknowledged & followed | Not Acknowledged & not followed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mmlu_pro | grader_hacking | 0.1951 [0.1549, 0.2371] | 0.8924 [0.8247, 0.9488] | 0.4210 [0.3596, 0.4837] | 0.9512 [0.9137, 0.9820] | 93 (46.5% [39.5, 53.5]) | 0.7626 [0.6860, 0.8344] | 0.731 [0.675, 0.785] | 0.609 [0.551, 0.665] | 204 (13.0%) | 4 (0.3%) | 166 (10.5%) | 1200 (76.2%) |
