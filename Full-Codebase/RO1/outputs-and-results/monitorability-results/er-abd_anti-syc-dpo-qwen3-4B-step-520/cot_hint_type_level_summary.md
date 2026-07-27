# CoT token-length hint-type-level summary — runs pooled across datasets

- **Model:** `er-abd/anti-syc-dpo-qwen3-4B-step-520` (`er-abd_anti-syc-dpo-qwen3-4B-step-520`)
- **Datasets:** mmlu_pro, gpqa_diamond, wmdp, codemmlu, lsat_ar, lsat_lr, medxpertqa, lexam
- **Hint types:** professor, grader_hacking, unethical

`token_count` is the full completion (thinking trace + final answer). Condition tables use valid runs only (not truncated, answer parsed), so followed + not-followed partition hinted exactly; bucket tables use every response. Use the truncation rate to set `max_new_tokens`.

## professor (pooled over 8 datasets)

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 3701.5 | 3223.5 | 2626 | 238 | 16374 | 12584 | 1595 |
| Unhinted (X=0, all) | 3265.5 | 3055.7 | 2186 | 249 | 16365 | 12518 | 1595 |
| Hinted, followed (X=1, Y=1) | 4393.3 | 3588.2 | 3239 | 351 | 16374 | 3226 | 674 |
| Hinted, not followed (X=1, Y=0) | 3463.1 | 3052.0 | 2462 | 238 | 16297 | 9358 | 1399 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 1016 | 4469 | 5579 | 6439 | 5168 | 2436 | 0 | 461 | 25568 | 461 (1.8%) | 16384 |

## grader_hacking (pooled over 8 datasets)

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 4002.2 | 3366.1 | 2973 | 211 | 16379 | 12427 | 1596 |
| Unhinted (X=0, all) | 3269.0 | 3061.6 | 2188 | 249 | 16365 | 12522 | 1596 |
| Hinted, followed (X=1, Y=1) | 4724.5 | 3545.1 | 3746 | 280 | 16330 | 3400 | 782 |
| Hinted, not followed (X=1, Y=0) | 3730.2 | 3255.1 | 2700 | 211 | 16379 | 9027 | 1403 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 1064 | 4236 | 5129 | 6445 | 5410 | 2666 | 0 | 618 | 25568 | 618 (2.4%) | 16384 |

## unethical (pooled over 8 datasets)

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 4003.6 | 3333.2 | 2846 | 231 | 16375 | 12488 | 1595 |
| Unhinted (X=0, all) | 3267.0 | 3058.9 | 2187 | 249 | 16365 | 12519 | 1595 |
| Hinted, followed (X=1, Y=1) | 3986.6 | 3311.7 | 2763 | 363 | 16375 | 8331 | 1356 |
| Hinted, not followed (X=1, Y=0) | 4037.6 | 3376.0 | 3068 | 231 | 16155 | 4157 | 875 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 844 | 4073 | 5575 | 6600 | 5208 | 2712 | 0 | 556 | 25568 | 556 (2.2%) | 16384 |

## Summary

| Hint type | Hinted (mean (std) · median) | Unhinted (mean (std) · median) | Hinted & followed (mean (std) · median) | Hinted & not followed (mean (std) · median) | Gap (Hinted mean − Unhinted mean) |
|---|---|---|---|---|---|
| professor | 3701.5 (3223.5) · 2626 | 3265.5 (3055.7) · 2186 | 4393.3 (3588.2) · 3239 | 3463.1 (3052.0) · 2462 | 436.0 |
| grader_hacking | 4002.2 (3366.1) · 2973 | 3269.0 (3061.6) · 2188 | 4724.5 (3545.1) · 3746 | 3730.2 (3255.1) · 2700 | 733.2 |
| unethical | 4003.6 (3333.2) · 2846 | 3267.0 (3058.9) · 2187 | 3986.6 (3311.7) · 2763 | 4037.6 (3376.0) · 3068 | 736.6 |
