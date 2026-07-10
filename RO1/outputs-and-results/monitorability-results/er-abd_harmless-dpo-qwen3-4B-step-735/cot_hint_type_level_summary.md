# CoT token-length hint-type-level summary — runs pooled across datasets

- **Model:** `er-abd/harmless-dpo-qwen3-4B-step-735` (`er-abd_harmless-dpo-qwen3-4B-step-735`)
- **Datasets:** mmlu_pro, gpqa_diamond, wmdp, codemmlu, lsat_ar, lsat_lr, medxpertqa, lexam
- **Hint types:** professor, grader_hacking, unethical

`token_count` is the full completion (thinking trace + final answer). Condition tables use valid runs only (not truncated, answer parsed), so followed + not-followed partition hinted exactly; bucket tables use every response. Use the truncation rate to set `max_new_tokens`.

## professor (pooled over 8 datasets)

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 3498.9 | 3121.2 | 262 | 16330 | 12634 | 1594 |
| Unhinted (X=0, all) | 3123.0 | 2957.0 | 261 | 16356 | 12582 | 1594 |
| Hinted, followed (X=1, Y=1) | 4160.0 | 3505.0 | 351 | 16330 | 3447 | 709 |
| Hinted, not followed (X=1, Y=0) | 3250.8 | 2926.3 | 262 | 16209 | 9187 | 1387 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 865 | 4900 | 6126 | 6269 | 4872 | 2197 | 0 | 339 | 25568 | 339 (1.3%) | 16384 |

## grader_hacking (pooled over 8 datasets)

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 3726.2 | 3254.3 | 227 | 16331 | 12560 | 1594 |
| Unhinted (X=0, all) | 3124.1 | 2957.7 | 261 | 16356 | 12577 | 1594 |
| Hinted, followed (X=1, Y=1) | 4418.3 | 3445.0 | 276 | 16331 | 3549 | 772 |
| Hinted, not followed (X=1, Y=0) | 3453.6 | 3134.6 | 227 | 16273 | 9011 | 1377 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 963 | 4754 | 5729 | 6249 | 5073 | 2382 | 0 | 418 | 25568 | 418 (1.6%) | 16384 |

## unethical (pooled over 7 datasets)

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 3901.3 | 3320.4 | 242 | 16377 | 11017 | 1395 |
| Unhinted (X=0, all) | 3226.5 | 2980.7 | 261 | 16356 | 10995 | 1395 |
| Hinted, followed (X=1, Y=1) | 3936.7 | 3284.2 | 311 | 16377 | 7444 | 1195 |
| Hinted, not followed (X=1, Y=0) | 3827.5 | 3393.8 | 242 | 16221 | 3573 | 743 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 680 | 3656 | 5277 | 5590 | 4522 | 2301 | 0 | 342 | 22368 | 342 (1.5%) | 16384 |

## Summary

| Hint type | Hinted (mean and std) | Unhinted (mean and std) | Hinted & followed (mean and std) | Hinted & not followed (mean and std) |
|---|---|---|---|---|
| professor | 3498.9 (3121.2) | 3123.0 (2957.0) | 4160.0 (3505.0) | 3250.8 (2926.3) |
| grader_hacking | 3726.2 (3254.3) | 3124.1 (2957.7) | 4418.3 (3445.0) | 3453.6 (3134.6) |
| unethical | 3901.3 (3320.4) | 3226.5 (2980.7) | 3936.7 (3284.2) | 3827.5 (3393.8) |
