# CoT token-length dataset-level summary — runs pooled across hint types

- **Model:** `Qwen/Qwen3-4B` (`Qwen_Qwen3-4B`)
- **Datasets:** mmlu_pro, gpqa_diamond, wmdp, codemmlu, lsat_ar, lsat_lr, medxpertqa, lexam
- **Hint types:** professor, grader_hacking, unethical

`token_count` is the full completion (thinking trace + final answer). Condition tables use valid runs only (not truncated, answer parsed), so followed + not-followed partition hinted exactly; bucket tables use every response. Use the truncation rate to set `max_new_tokens`.

## mmlu_pro (pooled over 3 hint types)

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 2727.0 | 2830.8 | 269 | 15779 | 4789 | 600 |
| Unhinted (X=0, all) | 2033.7 | 2443.0 | 259 | 15934 | 4782 | 600 |
| Hinted, followed (X=1, Y=1) | 3041.0 | 2979.8 | 293 | 15779 | 2061 | 362 |
| Hinted, not followed (X=1, Y=0) | 2489.9 | 2689.2 | 269 | 15506 | 2728 | 453 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 763 | 3210 | 2420 | 1533 | 1118 | 527 | 0 | 29 | 9600 | 29 (0.3%) | 16384 |

## gpqa_diamond (pooled over 3 hint types)

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 5147.2 | 3145.0 | 406 | 16374 | 4708 | 594 |
| Unhinted (X=0, all) | 4806.0 | 3173.7 | 428 | 16297 | 4698 | 594 |
| Hinted, followed (X=1, Y=1) | 5489.8 | 3187.0 | 406 | 16374 | 2708 | 452 |
| Hinted, not followed (X=1, Y=0) | 4683.2 | 3026.8 | 406 | 16143 | 2000 | 377 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 27 | 523 | 1378 | 2321 | 3744 | 1413 | 0 | 98 | 9504 | 98 (1.0%) | 16384 |

## wmdp (pooled over 3 hint types)

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 1792.8 | 2629.6 | 249 | 16375 | 4574 | 591 |
| Unhinted (X=0, all) | 1639.7 | 2570.7 | 312 | 16316 | 4548 | 591 |
| Hinted, followed (X=1, Y=1) | 2094.4 | 3007.5 | 338 | 16375 | 2182 | 380 |
| Hinted, not followed (X=1, Y=0) | 1517.7 | 2194.6 | 249 | 16069 | 2392 | 413 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 1201 | 4351 | 2262 | 510 | 325 | 488 | 0 | 463 | 9600 | 463 (4.8%) | 16384 |

## codemmlu (pooled over 3 hint types)

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 2567.2 | 2361.1 | 208 | 14638 | 4751 | 600 |
| Unhinted (X=0, all) | 2047.4 | 2234.7 | 208 | 14367 | 4758 | 600 |
| Hinted, followed (X=1, Y=1) | 3372.7 | 2490.9 | 273 | 14638 | 1694 | 324 |
| Hinted, not followed (X=1, Y=0) | 2120.8 | 2160.7 | 208 | 14500 | 3057 | 479 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 1912 | 1964 | 1996 | 1885 | 1452 | 300 | 0 | 91 | 9600 | 91 (0.9%) | 16384 |

## lsat_ar (pooled over 3 hint types)

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 6650.9 | 3215.6 | 855 | 16356 | 4745 | 600 |
| Unhinted (X=0, all) | 5540.3 | 3418.1 | 888 | 16182 | 4776 | 600 |
| Hinted, followed (X=1, Y=1) | 8548.8 | 2967.7 | 1133 | 16356 | 1471 | 349 |
| Hinted, not followed (X=1, Y=0) | 5798.2 | 2947.1 | 855 | 16200 | 3274 | 542 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 20 | 864 | 2343 | 3878 | 2416 | 0 | 79 | 9600 | 79 (0.8%) | 16384 |

## lsat_lr (pooled over 3 hint types)

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 1868.5 | 1546.6 | 303 | 9461 | 4795 | 600 |
| Unhinted (X=0, all) | 1480.9 | 1224.7 | 295 | 6901 | 4800 | 600 |
| Hinted, followed (X=1, Y=1) | 2644.2 | 1627.6 | 391 | 9214 | 1062 | 229 |
| Hinted, not followed (X=1, Y=0) | 1647.9 | 1449.0 | 303 | 9461 | 3733 | 540 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 757 | 3951 | 2185 | 1943 | 751 | 8 | 0 | 5 | 9600 | 5 (0.1%) | 16384 |

## medxpertqa (pooled over 3 hint types)

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 1666.1 | 999.3 | 351 | 9934 | 4798 | 600 |
| Unhinted (X=0, all) | 1627.4 | 910.9 | 365 | 8914 | 4800 | 600 |
| Hinted, followed (X=1, Y=1) | 1628.0 | 965.2 | 385 | 9934 | 2665 | 434 |
| Hinted, not followed (X=1, Y=0) | 1713.6 | 1038.6 | 351 | 9848 | 2133 | 364 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 95 | 2430 | 4761 | 2085 | 216 | 11 | 0 | 2 | 9600 | 2 (0.0%) | 16384 |

## lexam (pooled over 3 hint types)

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 2085.8 | 1106.7 | 315 | 5919 | 4798 | 600 |
| Unhinted (X=0, all) | 1995.4 | 1264.5 | 329 | 7017 | 4779 | 600 |
| Hinted, followed (X=1, Y=1) | 2215.8 | 1077.9 | 329 | 5919 | 2551 | 444 |
| Hinted, not followed (X=1, Y=0) | 1938.1 | 1120.6 | 315 | 5685 | 2247 | 425 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 435 | 2043 | 2836 | 3705 | 558 | 0 | 0 | 23 | 9600 | 23 (0.2%) | 16384 |

## Summary

| Dataset | Hinted (mean and std) | Unhinted (mean and std) | Hinted & followed (mean and std) | Hinted & not followed (mean and std) |
|---|---|---|---|---|
| mmlu_pro | 2727.0 (2830.8) | 2033.7 (2443.0) | 3041.0 (2979.8) | 2489.9 (2689.2) |
| gpqa_diamond | 5147.2 (3145.0) | 4806.0 (3173.7) | 5489.8 (3187.0) | 4683.2 (3026.8) |
| wmdp | 1792.8 (2629.6) | 1639.7 (2570.7) | 2094.4 (3007.5) | 1517.7 (2194.6) |
| codemmlu | 2567.2 (2361.1) | 2047.4 (2234.7) | 3372.7 (2490.9) | 2120.8 (2160.7) |
| lsat_ar | 6650.9 (3215.6) | 5540.3 (3418.1) | 8548.8 (2967.7) | 5798.2 (2947.1) |
| lsat_lr | 1868.5 (1546.6) | 1480.9 (1224.7) | 2644.2 (1627.6) | 1647.9 (1449.0) |
| medxpertqa | 1666.1 (999.3) | 1627.4 (910.9) | 1628.0 (965.2) | 1713.6 (1038.6) |
| lexam | 2085.8 (1106.7) | 1995.4 (1264.5) | 2215.8 (1077.9) | 1938.1 (1120.6) |
