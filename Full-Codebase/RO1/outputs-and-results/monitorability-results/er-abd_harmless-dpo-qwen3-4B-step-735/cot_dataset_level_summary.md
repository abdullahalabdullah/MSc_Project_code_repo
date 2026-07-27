# CoT token-length dataset-level summary — runs pooled across hint types

- **Model:** `er-abd_harmless-dpo-qwen3-4B-step-735` (`er-abd_harmless-dpo-qwen3-4B-step-735`)
- **Datasets:** mmlu_pro, gpqa_diamond, wmdp, codemmlu, lsat_ar, lsat_lr, medxpertqa, lexam
- **Hint types:** professor, grader_hacking, unethical

`token_count` is the full completion (thinking trace + final answer). Condition tables use valid runs only (not truncated, answer parsed), so followed + not-followed partition hinted exactly; bucket tables use every response. Use the truncation rate to set `max_new_tokens`.

## mmlu_pro (pooled over 3 hint types)

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 3350.9 | 3186.4 | 2115 | 322 | 16332 | 4775 | 600 |
| Unhinted (X=0, all) | 2411.6 | 2680.6 | 1350 | 268 | 16173 | 4770 | 600 |
| Hinted, followed (X=1, Y=1) | 3658.1 | 3387.5 | 2244 | 322 | 16285 | 1899 | 347 |
| Hinted, not followed (X=1, Y=0) | 3148.2 | 3029.9 | 2016 | 322 | 16332 | 2876 | 460 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 314 | 2615 | 2508 | 2044 | 1281 | 783 | 0 | 55 | 9600 | 55 (0.6%) | 16384 |

## gpqa_diamond (pooled over 3 hint types)

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 6013.4 | 3305.5 | 5713 | 439 | 16342 | 4682 | 594 |
| Unhinted (X=0, all) | 5518.1 | 3242.5 | 5203 | 549 | 16355 | 4659 | 594 |
| Hinted, followed (X=1, Y=1) | 6245.1 | 3223.4 | 5830 | 439 | 16342 | 2660 | 443 |
| Hinted, not followed (X=1, Y=0) | 5708.6 | 3387.1 | 5425 | 513 | 16273 | 2022 | 371 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 290 | 950 | 2056 | 4061 | 1982 | 0 | 163 | 9504 | 163 (1.7%) | 16384 |

## wmdp (pooled over 3 hint types)

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 2085.7 | 2609.3 | 1237 | 302 | 16377 | 4545 | 590 |
| Unhinted (X=0, all) | 1861.5 | 2548.8 | 1018 | 352 | 16265 | 4512 | 590 |
| Hinted, followed (X=1, Y=1) | 2264.6 | 2673.8 | 1356 | 356 | 16377 | 2163 | 375 |
| Hinted, not followed (X=1, Y=0) | 1923.3 | 2539.1 | 1102 | 302 | 16145 | 2382 | 403 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 433 | 3623 | 3103 | 1014 | 423 | 493 | 0 | 511 | 9600 | 511 (5.3%) | 16384 |

## codemmlu (pooled over 3 hint types)

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 3308.6 | 2912.1 | 2318 | 227 | 16087 | 4734 | 599 |
| Unhinted (X=0, all) | 2693.3 | 2812.7 | 1484 | 261 | 14912 | 4753 | 599 |
| Hinted, followed (X=1, Y=1) | 4376.1 | 3004.7 | 3822 | 276 | 15712 | 1553 | 298 |
| Hinted, not followed (X=1, Y=0) | 2787.4 | 2718.0 | 1758 | 227 | 16087 | 3181 | 483 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 1355 | 1760 | 1873 | 1912 | 1918 | 677 | 0 | 105 | 9600 | 105 (1.1%) | 16384 |

## lsat_ar (pooled over 3 hint types)

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 7688.2 | 3439.4 | 7364 | 1028 | 16221 | 4677 | 600 |
| Unhinted (X=0, all) | 6141.8 | 3526.0 | 5335 | 1064 | 16356 | 4671 | 600 |
| Hinted, followed (X=1, Y=1) | 9702.7 | 2954.1 | 9632 | 1709 | 16221 | 1303 | 337 |
| Hinted, not followed (X=1, Y=0) | 6910.2 | 3295.1 | 6352 | 1028 | 16209 | 3374 | 548 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 472 | 1876 | 3849 | 3151 | 0 | 252 | 9600 | 252 (2.6%) | 16384 |

## lsat_lr (pooled over 3 hint types)

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 2424.3 | 1902.3 | 1749 | 335 | 11397 | 4790 | 600 |
| Unhinted (X=0, all) | 1810.1 | 1495.7 | 1180 | 328 | 9462 | 4800 | 600 |
| Hinted, followed (X=1, Y=1) | 3416.9 | 1937.8 | 3251 | 470 | 10846 | 961 | 202 |
| Hinted, not followed (X=1, Y=0) | 2175.2 | 1809.9 | 1368 | 335 | 11397 | 3829 | 544 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 262 | 3345 | 2426 | 2154 | 1353 | 50 | 0 | 10 | 9600 | 10 (0.1%) | 16384 |

## medxpertqa (pooled over 3 hint types)

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 2172.9 | 1255.1 | 1900 | 407 | 12238 | 4797 | 600 |
| Unhinted (X=0, all) | 2140.7 | 1130.5 | 1906 | 435 | 10149 | 4797 | 600 |
| Hinted, followed (X=1, Y=1) | 2082.7 | 1195.3 | 1811 | 438 | 12238 | 2659 | 424 |
| Hinted, not followed (X=1, Y=0) | 2285.1 | 1317.4 | 2038 | 407 | 12143 | 2138 | 365 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 29 | 1157 | 4134 | 3690 | 552 | 32 | 0 | 6 | 9600 | 6 (0.1%) | 16384 |

## lexam (pooled over 3 hint types)

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 2635.1 | 1286.7 | 2531 | 390 | 8128 | 4796 | 600 |
| Unhinted (X=0, all) | 2475.4 | 1468.1 | 2237 | 355 | 7586 | 4782 | 600 |
| Hinted, followed (X=1, Y=1) | 2793.8 | 1274.9 | 2721 | 446 | 8128 | 2427 | 441 |
| Hinted, not followed (X=1, Y=0) | 2472.5 | 1278.6 | 2326 | 390 | 6849 | 2369 | 437 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 200 | 1299 | 2514 | 4045 | 1520 | 0 | 0 | 22 | 9600 | 22 (0.2%) | 16384 |

## Summary

| Dataset | Hinted (mean (std) · median) | Unhinted (mean (std) · median) | Hinted & followed (mean (std) · median) | Hinted & not followed (mean (std) · median) | Gap (Hinted mean − Unhinted mean) |
|---|---|---|---|---|---|
| mmlu_pro | 3350.9 (3186.4) · 2115 | 2411.6 (2680.6) · 1350 | 3658.1 (3387.5) · 2244 | 3148.2 (3029.9) · 2016 | 939.4 |
| gpqa_diamond | 6013.4 (3305.5) · 5713 | 5518.1 (3242.5) · 5203 | 6245.1 (3223.4) · 5830 | 5708.6 (3387.1) · 5425 | 495.3 |
| wmdp | 2085.7 (2609.3) · 1237 | 1861.5 (2548.8) · 1018 | 2264.6 (2673.8) · 1356 | 1923.3 (2539.1) · 1102 | 224.2 |
| codemmlu | 3308.6 (2912.1) · 2318 | 2693.3 (2812.7) · 1484 | 4376.1 (3004.7) · 3822 | 2787.4 (2718.0) · 1758 | 615.3 |
| lsat_ar | 7688.2 (3439.4) · 7364 | 6141.8 (3526.0) · 5335 | 9702.7 (2954.1) · 9632 | 6910.2 (3295.1) · 6352 | 1546.3 |
| lsat_lr | 2424.3 (1902.3) · 1749 | 1810.1 (1495.7) · 1180 | 3416.9 (1937.8) · 3251 | 2175.2 (1809.9) · 1368 | 614.2 |
| medxpertqa | 2172.9 (1255.1) · 1900 | 2140.7 (1130.5) · 1906 | 2082.7 (1195.3) · 1811 | 2285.1 (1317.4) · 2038 | 32.2 |
| lexam | 2635.1 (1286.7) · 2531 | 2475.4 (1468.1) · 2237 | 2793.8 (1274.9) · 2721 | 2472.5 (1278.6) · 2326 | 159.7 |
