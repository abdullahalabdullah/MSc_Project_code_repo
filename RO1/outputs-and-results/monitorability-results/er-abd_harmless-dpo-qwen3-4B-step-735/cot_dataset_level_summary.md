# CoT token-length dataset-level summary — runs pooled across hint types

- **Model:** `er-abd/harmless-dpo-qwen3-4B-step-735` (`er-abd_harmless-dpo-qwen3-4B-step-735`)
- **Datasets:** mmlu_pro, gpqa_diamond, wmdp, codemmlu, lsat_ar, lsat_lr, medxpertqa, lexam
- **Hint types:** professor, unethical

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

## gpqa_diamond (pooled over 2 hint types)

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 5991.9 | 3300.6 | 439 | 16342 | 3125 | 396 |
| Unhinted (X=0, all) | 5518.1 | 3242.6 | 549 | 16355 | 3106 | 396 |
| Hinted, followed (X=1, Y=1) | 6135.4 | 3229.8 | 439 | 16342 | 1968 | 310 |
| Hinted, not followed (X=1, Y=0) | 5747.8 | 3405.3 | 513 | 16221 | 1157 | 221 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 187 | 641 | 1357 | 2730 | 1314 | 0 | 105 | 6336 | 105 (1.7%) | 16384 |

## wmdp (pooled over 2 hint types)

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 2046.5 | 2553.1 | 302 | 16377 | 3030 | 393 |
| Unhinted (X=0, all) | 1860.3 | 2548.5 | 352 | 16265 | 3007 | 393 |
| Hinted, followed (X=1, Y=1) | 2218.8 | 2630.1 | 397 | 16377 | 1645 | 272 |
| Hinted, not followed (X=1, Y=0) | 1841.8 | 2443.7 | 302 | 16135 | 1385 | 245 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 272 | 2428 | 2109 | 650 | 277 | 328 | 0 | 336 | 6400 | 336 (5.2%) | 16384 |

## codemmlu (pooled over 2 hint types)

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 3338.0 | 2820.0 | 242 | 15400 | 3163 | 400 |
| Unhinted (X=0, all) | 2692.5 | 2810.6 | 261 | 14912 | 3174 | 400 |
| Hinted, followed (X=1, Y=1) | 4337.2 | 2910.3 | 311 | 14473 | 1277 | 236 |
| Hinted, not followed (X=1, Y=0) | 2661.4 | 2544.0 | 242 | 15400 | 1886 | 303 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 867 | 1154 | 1266 | 1287 | 1335 | 428 | 0 | 63 | 6400 | 63 (1.0%) | 16384 |

## lsat_ar (pooled over 2 hint types)

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 7752.9 | 3411.8 | 1028 | 16221 | 3145 | 400 |
| Unhinted (X=0, all) | 6141.8 | 3526.2 | 1064 | 16356 | 3114 | 400 |
| Hinted, followed (X=1, Y=1) | 9572.9 | 2916.1 | 1709 | 16221 | 1029 | 245 |
| Hinted, not followed (X=1, Y=0) | 6867.8 | 3282.6 | 1028 | 16209 | 2116 | 359 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 311 | 1251 | 2554 | 2143 | 0 | 141 | 6400 | 141 (2.2%) | 16384 |

## lsat_lr (pooled over 2 hint types)

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 2425.5 | 1861.8 | 335 | 11397 | 3194 | 400 |
| Unhinted (X=0, all) | 1810.1 | 1495.7 | 328 | 9462 | 3200 | 400 |
| Hinted, followed (X=1, Y=1) | 3329.0 | 1823.2 | 470 | 10846 | 758 | 158 |
| Hinted, not followed (X=1, Y=0) | 2144.4 | 1782.9 | 335 | 11397 | 2436 | 353 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 164 | 2204 | 1637 | 1455 | 906 | 28 | 0 | 6 | 6400 | 6 (0.1%) | 16384 |

## medxpertqa (pooled over 2 hint types)

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 2080.7 | 1175.9 | 407 | 12238 | 3199 | 400 |
| Unhinted (X=0, all) | 2140.7 | 1130.6 | 435 | 10149 | 3198 | 400 |
| Hinted, followed (X=1, Y=1) | 1971.1 | 1102.0 | 438 | 12238 | 2072 | 307 |
| Hinted, not followed (X=1, Y=0) | 2282.1 | 1277.2 | 407 | 9534 | 1127 | 201 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 18 | 791 | 2848 | 2388 | 334 | 18 | 0 | 3 | 6400 | 3 (0.0%) | 16384 |

## lexam (pooled over 2 hint types)

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 2539.0 | 1187.2 | 390 | 8128 | 3197 | 400 |
| Unhinted (X=0, all) | 2475.4 | 1468.2 | 355 | 7586 | 3188 | 400 |
| Hinted, followed (X=1, Y=1) | 2585.0 | 1147.3 | 446 | 8128 | 1807 | 310 |
| Hinted, not followed (X=1, Y=0) | 2479.1 | 1234.9 | 390 | 6849 | 1390 | 269 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 127 | 856 | 1717 | 2790 | 895 | 0 | 0 | 15 | 6400 | 15 (0.2%) | 16384 |

## Summary

| Dataset | Hinted (mean and std) | Unhinted (mean and std) | Hinted & followed (mean and std) | Hinted & not followed (mean and std) |
|---|---|---|---|---|
| mmlu_pro | 3002.9 (3015.9) | 2411.6 (2681.2) | 3737.7 (3899.4) | 2808.1 (2702.7) |
| gpqa_diamond | 5991.9 (3300.6) | 5518.1 (3242.6) | 6135.4 (3229.8) | 5747.8 (3405.3) |
| wmdp | 2046.5 (2553.1) | 1860.3 (2548.5) | 2218.8 (2630.1) | 1841.8 (2443.7) |
| codemmlu | 3338.0 (2820.0) | 2692.5 (2810.6) | 4337.2 (2910.3) | 2661.4 (2544.0) |
| lsat_ar | 7752.9 (3411.8) | 6141.8 (3526.2) | 9572.9 (2916.1) | 6867.8 (3282.6) |
| lsat_lr | 2425.5 (1861.8) | 1810.1 (1495.7) | 3329.0 (1823.2) | 2144.4 (1782.9) |
| medxpertqa | 2080.7 (1175.9) | 2140.7 (1130.6) | 1971.1 (1102.0) | 2282.1 (1277.2) |
| lexam | 2539.0 (1187.2) | 2475.4 (1468.2) | 2585.0 (1147.3) | 2479.1 (1234.9) |
