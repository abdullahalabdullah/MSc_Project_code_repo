# CoT token-length hint-type-level summary — runs pooled across datasets

- **Model:** `er-abd_harmless-dpo-qwen3-4B-step-735` (`er-abd_harmless-dpo-qwen3-4B-step-735`)
- **Datasets:** mmlu_pro, gpqa_diamond, wmdp, codemmlu, lsat_ar, lsat_lr, medxpertqa, lexam
- **Hint types:** professor, grader_hacking, unethical

`token_count` is the full completion (thinking trace + final answer). Condition tables use valid runs only (not truncated, answer parsed), so followed + not-followed partition hinted exactly; bucket tables use every response. Use the truncation rate to set `max_new_tokens`.

## professor (pooled over 8 datasets)

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 3498.9 | 3121.2 | 2398 | 262 | 16330 | 12634 | 1594 |
| Unhinted (X=0, all) | 3123.0 | 2957.0 | 2020 | 261 | 16356 | 12582 | 1594 |
| Hinted, followed (X=1, Y=1) | 4160.0 | 3505.0 | 2939 | 351 | 16330 | 3447 | 709 |
| Hinted, not followed (X=1, Y=0) | 3250.8 | 2926.3 | 2257 | 262 | 16209 | 9187 | 1387 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 865 | 4900 | 6126 | 6269 | 4872 | 2197 | 0 | 339 | 25568 | 339 (1.3%) | 16384 |

## grader_hacking (pooled over 8 datasets)

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 3726.2 | 3254.3 | 2651 | 227 | 16331 | 12560 | 1594 |
| Unhinted (X=0, all) | 3124.1 | 2957.7 | 2020 | 261 | 16356 | 12577 | 1594 |
| Hinted, followed (X=1, Y=1) | 4418.3 | 3445.0 | 3443 | 276 | 16331 | 3549 | 772 |
| Hinted, not followed (X=1, Y=0) | 3453.6 | 3134.6 | 2391 | 227 | 16273 | 9011 | 1377 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 963 | 4754 | 5729 | 6249 | 5073 | 2382 | 0 | 418 | 25568 | 418 (1.6%) | 16384 |

## unethical (pooled over 8 datasets)

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 3882.2 | 3319.3 | 2674 | 242 | 16377 | 12602 | 1595 |
| Unhinted (X=0, all) | 3123.5 | 2956.9 | 2021 | 261 | 16356 | 12585 | 1595 |
| Hinted, followed (X=1, Y=1) | 3879.7 | 3277.4 | 2626 | 311 | 16377 | 8629 | 1386 |
| Hinted, not followed (X=1, Y=0) | 3887.7 | 3409.0 | 2812 | 242 | 16332 | 3973 | 847 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 767 | 4435 | 6125 | 6273 | 5012 | 2589 | 0 | 367 | 25568 | 367 (1.4%) | 16384 |

## Summary

| Hint type | Hinted (mean (std) · median) | Unhinted (mean (std) · median) | Hinted & followed (mean (std) · median) | Hinted & not followed (mean (std) · median) | Gap (Hinted mean − Unhinted mean) |
|---|---|---|---|---|---|
| professor | 3498.9 (3121.2) · 2398 | 3123.0 (2957.0) · 2020 | 4160.0 (3505.0) · 2939 | 3250.8 (2926.3) · 2257 | 375.9 |
| grader_hacking | 3726.2 (3254.3) · 2651 | 3124.1 (2957.7) · 2020 | 4418.3 (3445.0) · 3443 | 3453.6 (3134.6) · 2391 | 602.1 |
| unethical | 3882.2 (3319.3) · 2674 | 3123.5 (2956.9) · 2021 | 3879.7 (3277.4) · 2626 | 3887.7 (3409.0) · 2812 | 758.7 |
