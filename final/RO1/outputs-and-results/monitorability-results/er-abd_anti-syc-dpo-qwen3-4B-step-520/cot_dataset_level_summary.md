# CoT token-length dataset-level summary — runs pooled across hint types

- **Model:** `er-abd/anti-syc-dpo-qwen3-4B-step-520` (`er-abd_anti-syc-dpo-qwen3-4B-step-520`)
- **Datasets:** mmlu_pro, gpqa_diamond, wmdp, codemmlu, lsat_ar, lsat_lr, medxpertqa, lexam
- **Hint types:** professor, grader_hacking, unethical

`token_count` is the full completion (thinking trace + final answer). Condition tables use valid runs only (not truncated, answer parsed), so followed + not-followed partition hinted exactly; bucket tables use every response. Use the truncation rate to set `max_new_tokens`.

## mmlu_pro (pooled over 3 hint types)

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 3508.0 | 3199.2 | 2288 | 325 | 16155 | 4742 | 600 |
| Unhinted (X=0, all) | 2562.7 | 2815.8 | 1434 | 278 | 16347 | 4761 | 600 |
| Hinted, followed (X=1, Y=1) | 3803.8 | 3350.5 | 2425 | 334 | 16081 | 1794 | 350 |
| Hinted, not followed (X=1, Y=0) | 3328.0 | 3090.3 | 2220 | 325 | 16155 | 2948 | 473 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 364 | 2436 | 2343 | 2123 | 1410 | 827 | 0 | 97 | 9600 | 97 (1.0%) | 16384 |

## gpqa_diamond (pooled over 3 hint types)

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 6264.4 | 3335.4 | 5971 | 441 | 16375 | 4662 | 594 |
| Unhinted (X=0, all) | 5816.5 | 3351.2 | 5504 | 503 | 16365 | 4638 | 594 |
| Hinted, followed (X=1, Y=1) | 6466.8 | 3249.8 | 6136 | 441 | 16375 | 2640 | 448 |
| Hinted, not followed (X=1, Y=0) | 6000.3 | 3426.8 | 5722 | 472 | 16159 | 2022 | 381 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 257 | 747 | 2007 | 4035 | 2249 | 0 | 204 | 9504 | 204 (2.1%) | 16384 |

## wmdp (pooled over 3 hint types)

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 2247.6 | 2759.6 | 1369 | 323 | 16351 | 4515 | 592 |
| Unhinted (X=0, all) | 1997.8 | 2730.1 | 1093 | 349 | 16330 | 4514 | 592 |
| Hinted, followed (X=1, Y=1) | 2456.1 | 2897.9 | 1469 | 397 | 16351 | 2060 | 359 |
| Hinted, not followed (X=1, Y=0) | 2072.8 | 2625.9 | 1264 | 323 | 16297 | 2455 | 422 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 520 | 3131 | 3104 | 1389 | 333 | 563 | 0 | 560 | 9600 | 560 (5.8%) | 16384 |

## codemmlu (pooled over 3 hint types)

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 3482.4 | 2986.7 | 2484 | 211 | 15518 | 4675 | 600 |
| Unhinted (X=0, all) | 2768.6 | 2859.9 | 1598 | 249 | 14313 | 4689 | 600 |
| Hinted, followed (X=1, Y=1) | 4504.9 | 3078.7 | 3971 | 280 | 14967 | 1531 | 308 |
| Hinted, not followed (X=1, Y=0) | 2984.4 | 2809.6 | 1929 | 211 | 15518 | 3144 | 488 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 1380 | 1575 | 1789 | 1855 | 2023 | 742 | 0 | 236 | 9600 | 236 (2.5%) | 16384 |

## lsat_ar (pooled over 3 hint types)

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 7985.5 | 3483.1 | 7664 | 1317 | 16379 | 4603 | 600 |
| Unhinted (X=0, all) | 6241.3 | 3578.4 | 5382 | 1123 | 16331 | 4680 | 600 |
| Hinted, followed (X=1, Y=1) | 10050.6 | 3012.3 | 10030 | 1703 | 16374 | 1174 | 301 |
| Hinted, not followed (X=1, Y=0) | 7278.4 | 3349.8 | 6699 | 1317 | 16379 | 3429 | 546 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 350 | 1878 | 3765 | 3290 | 0 | 317 | 9600 | 317 (3.3%) | 16384 |

## lsat_lr (pooled over 3 hint types)

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 2654.2 | 2083.0 | 2074 | 323 | 12752 | 4776 | 600 |
| Unhinted (X=0, all) | 1913.4 | 1663.5 | 1179 | 274 | 10129 | 4794 | 600 |
| Hinted, followed (X=1, Y=1) | 3663.9 | 2101.9 | 3470 | 499 | 11811 | 934 | 196 |
| Hinted, not followed (X=1, Y=0) | 2408.8 | 2003.2 | 1635 | 323 | 12752 | 3842 | 543 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 424 | 3208 | 2051 | 2132 | 1648 | 107 | 0 | 30 | 9600 | 30 (0.3%) | 16384 |

## medxpertqa (pooled over 3 hint types)

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 2359.7 | 1328.2 | 2099 | 339 | 12191 | 4782 | 600 |
| Unhinted (X=0, all) | 2306.3 | 1252.1 | 2067 | 370 | 11311 | 4794 | 600 |
| Hinted, followed (X=1, Y=1) | 2231.8 | 1318.4 | 1920 | 429 | 12191 | 2529 | 415 |
| Hinted, not followed (X=1, Y=0) | 2503.3 | 1324.8 | 2279 | 339 | 10348 | 2253 | 378 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 34 | 1005 | 3644 | 4127 | 730 | 36 | 0 | 24 | 9600 | 24 (0.2%) | 16384 |

## lexam (pooled over 3 hint types)

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 2809.2 | 1334.4 | 2704 | 371 | 7464 | 4744 | 600 |
| Unhinted (X=0, all) | 2579.8 | 1540.0 | 2308 | 318 | 7207 | 4689 | 600 |
| Hinted, followed (X=1, Y=1) | 2932.5 | 1302.5 | 2854 | 428 | 7464 | 2295 | 435 |
| Hinted, not followed (X=1, Y=0) | 2693.6 | 1353.7 | 2552 | 371 | 7463 | 2449 | 446 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 197 | 1166 | 2255 | 3973 | 1842 | 0 | 0 | 167 | 9600 | 167 (1.7%) | 16384 |

## Summary

| Dataset | Hinted (mean (std) · median) | Unhinted (mean (std) · median) | Hinted & followed (mean (std) · median) | Hinted & not followed (mean (std) · median) | Gap (Hinted mean − Unhinted mean) |
|---|---|---|---|---|---|
| mmlu_pro | 3508.0 (3199.2) · 2288 | 2562.7 (2815.8) · 1434 | 3803.8 (3350.5) · 2425 | 3328.0 (3090.3) · 2220 | 945.3 |
| gpqa_diamond | 6264.4 (3335.4) · 5971 | 5816.5 (3351.2) · 5504 | 6466.8 (3249.8) · 6136 | 6000.3 (3426.8) · 5722 | 448.0 |
| wmdp | 2247.6 (2759.6) · 1369 | 1997.8 (2730.1) · 1093 | 2456.1 (2897.9) · 1469 | 2072.8 (2625.9) · 1264 | 249.8 |
| codemmlu | 3482.4 (2986.7) · 2484 | 2768.6 (2859.9) · 1598 | 4504.9 (3078.7) · 3971 | 2984.4 (2809.6) · 1929 | 713.8 |
| lsat_ar | 7985.5 (3483.1) · 7664 | 6241.3 (3578.4) · 5382 | 10050.6 (3012.3) · 10030 | 7278.4 (3349.8) · 6699 | 1744.2 |
| lsat_lr | 2654.2 (2083.0) · 2074 | 1913.4 (1663.5) · 1179 | 3663.9 (2101.9) · 3470 | 2408.8 (2003.2) · 1635 | 740.8 |
| medxpertqa | 2359.7 (1328.2) · 2099 | 2306.3 (1252.1) · 2067 | 2231.8 (1318.4) · 1920 | 2503.3 (1324.8) · 2279 | 53.4 |
| lexam | 2809.2 (1334.4) · 2704 | 2579.8 (1540.0) · 2308 | 2932.5 (1302.5) · 2854 | 2693.6 (1353.7) · 2552 | 229.3 |
