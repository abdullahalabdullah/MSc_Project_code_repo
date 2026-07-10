# CoT token-length dataset-level summary — runs pooled across hint types

- **Model:** `er-abd/harmless-dpo-qwen3-4B-step-735` (`er-abd_harmless-dpo-qwen3-4B-step-735`)
- **Datasets:** mmlu_pro, gpqa_diamond, wmdp, codemmlu, lsat_ar, lsat_lr, medxpertqa, lexam
- **Hint types:** professor

`token_count` is the full completion (thinking trace + final answer). Condition tables use valid runs only (not truncated, answer parsed), so followed + not-followed partition hinted exactly; bucket tables use every response. Use the truncation rate to set `max_new_tokens`.

## mmlu_pro (pooled over 1 hint types)

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 3002.9 | 3015.9 | 359 | 15840 | 1598 | 200 |
| Unhinted (X=0, all) | 2411.6 | 2681.2 | 268 | 16173 | 1590 | 200 |
| Hinted, followed (X=1, Y=1) | 3737.7 | 3899.4 | 418 | 15840 | 335 | 66 |
| Hinted, not followed (X=1, Y=0) | 2808.1 | 2702.7 | 359 | 15701 | 1263 | 179 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 95 | 936 | 874 | 681 | 363 | 239 | 0 | 12 | 3200 | 12 (0.4%) | 16384 |

## gpqa_diamond (pooled over 1 hint types)

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 5878.0 | 3131.4 | 513 | 16330 | 1563 | 198 |
| Unhinted (X=0, all) | 5518.1 | 3243.2 | 549 | 16355 | 1553 | 198 |
| Hinted, followed (X=1, Y=1) | 6307.2 | 3080.3 | 690 | 16330 | 724 | 127 |
| Hinted, not followed (X=1, Y=0) | 5507.5 | 3129.6 | 513 | 15487 | 839 | 144 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 88 | 315 | 687 | 1408 | 618 | 0 | 52 | 3168 | 52 (1.6%) | 16384 |

## wmdp (pooled over 1 hint types)

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 2036.8 | 2588.8 | 313 | 16135 | 1513 | 196 |
| Unhinted (X=0, all) | 1856.9 | 2547.8 | 352 | 16265 | 1502 | 196 |
| Hinted, followed (X=1, Y=1) | 2324.1 | 2788.8 | 397 | 15459 | 521 | 98 |
| Hinted, not followed (X=1, Y=0) | 1885.9 | 2465.4 | 313 | 16135 | 992 | 161 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 132 | 1230 | 1067 | 295 | 140 | 164 | 0 | 172 | 3200 | 172 (5.4%) | 16384 |

## codemmlu (pooled over 1 hint types)

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 3006.8 | 2834.4 | 262 | 15400 | 1585 | 200 |
| Unhinted (X=0, all) | 2692.5 | 2811.0 | 261 | 14912 | 1587 | 200 |
| Hinted, followed (X=1, Y=1) | 4273.2 | 3172.4 | 351 | 14473 | 321 | 71 |
| Hinted, not followed (X=1, Y=0) | 2685.1 | 2648.5 | 262 | 15400 | 1264 | 178 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 486 | 660 | 599 | 614 | 607 | 206 | 0 | 28 | 3200 | 28 (0.9%) | 16384 |

## lsat_ar (pooled over 1 hint types)

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 7103.5 | 3578.1 | 1028 | 16209 | 1579 | 200 |
| Unhinted (X=0, all) | 6141.8 | 3526.7 | 1064 | 16356 | 1557 | 200 |
| Hinted, followed (X=1, Y=1) | 10209.0 | 3005.2 | 2302 | 16039 | 267 | 78 |
| Hinted, not followed (X=1, Y=0) | 6471.5 | 3348.8 | 1028 | 16209 | 1312 | 192 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 171 | 758 | 1259 | 948 | 0 | 64 | 3200 | 64 (2.0%) | 16384 |

## lsat_lr (pooled over 1 hint types)

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 2145.2 | 1757.9 | 335 | 11210 | 1600 | 200 |
| Unhinted (X=0, all) | 1810.1 | 1496.0 | 328 | 9462 | 1600 | 200 |
| Hinted, followed (X=1, Y=1) | 3089.3 | 1794.1 | 547 | 9782 | 166 | 42 |
| Hinted, not followed (X=1, Y=0) | 2035.9 | 1721.1 | 335 | 11210 | 1434 | 195 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 84 | 1170 | 888 | 651 | 395 | 12 | 0 | 0 | 3200 | 0 (0.0%) | 11210 |

## medxpertqa (pooled over 1 hint types)

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 2206.9 | 1197.3 | 407 | 9534 | 1599 | 200 |
| Unhinted (X=0, all) | 2140.7 | 1130.7 | 435 | 10149 | 1599 | 200 |
| Hinted, followed (X=1, Y=1) | 2000.2 | 984.9 | 438 | 8193 | 613 | 111 |
| Hinted, not followed (X=1, Y=0) | 2335.3 | 1296.1 | 407 | 9534 | 986 | 159 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 7 | 358 | 1376 | 1268 | 179 | 10 | 0 | 2 | 3200 | 2 (0.1%) | 16384 |

## lexam (pooled over 1 hint types)

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 2626.2 | 1310.0 | 443 | 6849 | 1597 | 200 |
| Unhinted (X=0, all) | 2475.4 | 1468.4 | 355 | 7586 | 1594 | 200 |
| Hinted, followed (X=1, Y=1) | 2947.0 | 1336.4 | 517 | 6760 | 500 | 116 |
| Hinted, not followed (X=1, Y=0) | 2480.0 | 1271.7 | 443 | 6849 | 1097 | 179 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 61 | 458 | 836 | 1315 | 521 | 0 | 0 | 9 | 3200 | 9 (0.3%) | 16384 |

## Summary

| Dataset | Hinted (mean and std) | Unhinted (mean and std) | Hinted & followed (mean and std) | Hinted & not followed (mean and std) |
|---|---|---|---|---|
| mmlu_pro | 3002.9 (3015.9) | 2411.6 (2681.2) | 3737.7 (3899.4) | 2808.1 (2702.7) |
| gpqa_diamond | 5878.0 (3131.4) | 5518.1 (3243.2) | 6307.2 (3080.3) | 5507.5 (3129.6) |
| wmdp | 2036.8 (2588.8) | 1856.9 (2547.8) | 2324.1 (2788.8) | 1885.9 (2465.4) |
| codemmlu | 3006.8 (2834.4) | 2692.5 (2811.0) | 4273.2 (3172.4) | 2685.1 (2648.5) |
| lsat_ar | 7103.5 (3578.1) | 6141.8 (3526.7) | 10209.0 (3005.2) | 6471.5 (3348.8) |
| lsat_lr | 2145.2 (1757.9) | 1810.1 (1496.0) | 3089.3 (1794.1) | 2035.9 (1721.1) |
| medxpertqa | 2206.9 (1197.3) | 2140.7 (1130.7) | 2000.2 (984.9) | 2335.3 (1296.1) |
| lexam | 2626.2 (1310.0) | 2475.4 (1468.4) | 2947.0 (1336.4) | 2480.0 (1271.7) |
