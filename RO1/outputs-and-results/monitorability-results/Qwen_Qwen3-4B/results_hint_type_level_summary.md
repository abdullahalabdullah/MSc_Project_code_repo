# Results hint-type-level summary — instances pooled across datasets

- **Model:** `Qwen/Qwen3-4B` (`Qwen_Qwen3-4B`)
- **Datasets:** mmlu_pro, gpqa_diamond, wmdp, codemmlu, lsat_ar, lsat_lr, medxpertqa, lexam
- **Hint types:** professor, grader_hacking, unethical

## professor (pooled over 8 datasets)

| Hint type | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| professor | 0.1998 (0.3133) | 0.7371 (0.4977) | 0.4871 (0.3021) | 0.8628 (0.2525) | 666 (41.8%) | 0.7595 | 0.625 (0.427) | 0.524 (0.438) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 2279 (18.0%) | 48 (0.4%) |
| **Not acknowledged** | 1375 (10.9%) | 8963 (70.8%) |

**Opacity rate:** 0.3763 — 1375 covert / 3654 hint-following runs (covert / all hint-following).

## grader_hacking (pooled over 8 datasets)

| Hint type | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| grader_hacking | 0.1952 (0.3038) | 0.7484 (0.4853) | 0.4520 (0.2938) | 0.8694 (0.2509) | 707 (44.3%) | 0.5220 | 0.625 (0.427) | 0.522 (0.432) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1265 (10.0%) | 28 (0.2%) |
| **Not acknowledged** | 2325 (18.4%) | 9016 (71.4%) |

**Opacity rate:** 0.6476 — 2325 covert / 3590 hint-following runs (covert / all hint-following).

## unethical (pooled over 8 datasets)

| Hint type | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| unethical | 0.6335 (0.3739) | 0.8936 (0.2575) | 0.7340 (0.2965) | 0.9299 (0.1856) | 1377 (86.3%) | 0.8982 | 0.625 (0.427) | 0.234 (0.346) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 7735 (61.1%) | 71 (0.6%) |
| **Not acknowledged** | 1415 (11.2%) | 3438 (27.2%) |

**Opacity rate:** 0.1546 — 1415 covert / 9150 hint-following runs (covert / all hint-following).

## Summary

| Hint type | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) | Acknowledged & followed | Acknowledged & not followed | Not Acknowledged & followed | Not Acknowledged & not followed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| professor | 0.1998 (0.3133) | 0.7371 (0.4977) | 0.4871 (0.3021) | 0.8628 (0.2525) | 666 (41.8%) | 0.7595 | 0.625 (0.427) | 0.524 (0.438) | 2279 (18.0%) | 48 (0.4%) | 1375 (10.9%) | 8963 (70.8%) |
| grader_hacking | 0.1952 (0.3038) | 0.7484 (0.4853) | 0.4520 (0.2938) | 0.8694 (0.2509) | 707 (44.3%) | 0.5220 | 0.625 (0.427) | 0.522 (0.432) | 1265 (10.0%) | 28 (0.2%) | 2325 (18.4%) | 9016 (71.4%) |
| unethical | 0.6335 (0.3739) | 0.8936 (0.2575) | 0.7340 (0.2965) | 0.9299 (0.1856) | 1377 (86.3%) | 0.8982 | 0.625 (0.427) | 0.234 (0.346) | 7735 (61.1%) | 71 (0.6%) | 1415 (11.2%) | 3438 (27.2%) |
