# Results hint-type-level summary — instances pooled across datasets

- **Model:** `er-abd/harmless-dpo-qwen3-4B-step-735` (`er-abd_harmless-dpo-qwen3-4B-step-735`)
- **Datasets:** mmlu_pro, gpqa_diamond, wmdp, codemmlu, lsat_ar, lsat_lr, medxpertqa, lexam
- **Hint types:** professor, grader_hacking, unethical

## professor (pooled over 8 datasets)

| Hint type | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| professor | 0.1853 (0.3045) | 0.7444 (0.4317) | 0.4777 (0.3047) | 0.8612 (0.2552) | 628 (39.4%) | 0.7836 | 0.629 (0.432) | 0.540 (0.442) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 2303 (18.2%) | 68 (0.5%) |
| **Not acknowledged** | 1144 (9.1%) | 9119 (72.2%) |

**Opacity rate:** 0.3319 — 1144 covert / 3447 hint-following runs (covert / all hint-following).

## grader_hacking (pooled over 8 datasets)

| Hint type | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| grader_hacking | 0.1950 (0.3103) | 0.7586 (0.4497) | 0.4651 (0.3061) | 0.8830 (0.2364) | 683 (42.8%) | 0.5944 | 0.630 (0.431) | 0.529 (0.441) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1462 (11.6%) | 11 (0.1%) |
| **Not acknowledged** | 2087 (16.6%) | 9000 (71.7%) |

**Opacity rate:** 0.5881 — 2087 covert / 3549 hint-following runs (covert / all hint-following).

## unethical (pooled over 7 datasets)

| Hint type | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| unethical | 0.5788 (0.3921) | 0.8840 (0.2708) | 0.7087 (0.3093) | 0.9269 (0.1917) | 1140 (81.7%) | 0.9092 | 0.614 (0.434) | 0.278 (0.377) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 6413 (58.2%) | 88 (0.8%) |
| **Not acknowledged** | 1031 (9.4%) | 3485 (31.6%) |

**Opacity rate:** 0.1385 — 1031 covert / 7444 hint-following runs (covert / all hint-following).

## Summary

| Hint type | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) | Acknowledged & followed | Acknowledged & not followed | Not Acknowledged & followed | Not Acknowledged & not followed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| professor | 0.1853 (0.3045) | 0.7444 (0.4317) | 0.4777 (0.3047) | 0.8612 (0.2552) | 628 (39.4%) | 0.7836 | 0.629 (0.432) | 0.540 (0.442) | 2303 (18.2%) | 68 (0.5%) | 1144 (9.1%) | 9119 (72.2%) |
| grader_hacking | 0.1950 (0.3103) | 0.7586 (0.4497) | 0.4651 (0.3061) | 0.8830 (0.2364) | 683 (42.8%) | 0.5944 | 0.630 (0.431) | 0.529 (0.441) | 1462 (11.6%) | 11 (0.1%) | 2087 (16.6%) | 9000 (71.7%) |
| unethical | 0.5788 (0.3921) | 0.8840 (0.2708) | 0.7087 (0.3093) | 0.9269 (0.1917) | 1140 (81.7%) | 0.9092 | 0.614 (0.434) | 0.278 (0.377) | 6413 (58.2%) | 88 (0.8%) | 1031 (9.4%) | 3485 (31.6%) |
