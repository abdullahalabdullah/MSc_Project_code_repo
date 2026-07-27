# CoT token-length overall summary — per (dataset × hint type)

- **Model:** `er-abd_harmless-dpo-qwen3-4B-step-735` (`er-abd_harmless-dpo-qwen3-4B-step-735`)
- **Datasets:** mmlu_pro, gpqa_diamond, wmdp, codemmlu, lsat_ar, lsat_lr, medxpertqa, lexam
- **Hint types:** professor, grader_hacking, unethical

`token_count` is the full completion (thinking trace + final answer). Condition tables use valid runs only (not truncated, answer parsed), so followed + not-followed partition hinted exactly; bucket tables use every response. Use the truncation rate to set `max_new_tokens`.

## mmlu_pro × professor

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 3002.9 | 3015.9 | 1868 | 359 | 15840 | 1598 | 200 |
| Unhinted (X=0, all) | 2411.6 | 2681.2 | 1350 | 268 | 16173 | 1590 | 200 |
| Hinted, followed (X=1, Y=1) | 3737.7 | 3899.4 | 1981 | 418 | 15840 | 335 | 66 |
| Hinted, not followed (X=1, Y=0) | 2808.1 | 2702.7 | 1850 | 359 | 15701 | 1263 | 179 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 95 | 936 | 874 | 681 | 363 | 239 | 0 | 12 | 3200 | 12 (0.4%) | 16384 |

## mmlu_pro × grader_hacking

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 3303.5 | 3185.4 | 2114 | 322 | 16285 | 1592 | 200 |
| Unhinted (X=0, all) | 2411.6 | 2681.2 | 1350 | 268 | 16173 | 1590 | 200 |
| Hinted, followed (X=1, Y=1) | 4015.2 | 3415.7 | 2902 | 382 | 16285 | 379 | 90 |
| Hinted, not followed (X=1, Y=0) | 3081.1 | 3077.9 | 1934 | 322 | 16186 | 1213 | 177 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 132 | 900 | 786 | 680 | 428 | 256 | 0 | 18 | 3200 | 18 (0.6%) | 16384 |

## mmlu_pro × unethical

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 3749.5 | 3309.7 | 2418 | 322 | 16332 | 1585 | 200 |
| Unhinted (X=0, all) | 2411.6 | 2681.2 | 1350 | 268 | 16173 | 1590 | 200 |
| Hinted, followed (X=1, Y=1) | 3521.4 | 3212.4 | 2172 | 322 | 15603 | 1185 | 191 |
| Hinted, not followed (X=1, Y=0) | 4425.4 | 3500.4 | 3254 | 371 | 16332 | 400 | 104 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 87 | 779 | 848 | 683 | 490 | 288 | 0 | 25 | 3200 | 25 (0.8%) | 16384 |

## gpqa_diamond × professor

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 5878.0 | 3131.4 | 5623 | 513 | 16330 | 1563 | 198 |
| Unhinted (X=0, all) | 5518.1 | 3243.2 | 5203 | 549 | 16355 | 1553 | 198 |
| Hinted, followed (X=1, Y=1) | 6307.2 | 3080.3 | 6006 | 690 | 16330 | 724 | 127 |
| Hinted, not followed (X=1, Y=0) | 5507.5 | 3129.6 | 5184 | 513 | 15487 | 839 | 144 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 88 | 315 | 687 | 1408 | 618 | 0 | 52 | 3168 | 52 (1.6%) | 16384 |

## gpqa_diamond × grader_hacking

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 6056.5 | 3315.8 | 5819 | 648 | 16273 | 1557 | 198 |
| Unhinted (X=0, all) | 5518.1 | 3243.2 | 5203 | 549 | 16355 | 1553 | 198 |
| Hinted, followed (X=1, Y=1) | 6557.1 | 3187.0 | 6164 | 1032 | 15710 | 692 | 133 |
| Hinted, not followed (X=1, Y=0) | 5656.0 | 3363.9 | 5421 | 648 | 16273 | 865 | 150 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 103 | 309 | 699 | 1331 | 668 | 0 | 58 | 3168 | 58 (1.8%) | 16384 |

## gpqa_diamond × unethical

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 6105.9 | 3459.0 | 5668 | 439 | 16342 | 1562 | 198 |
| Unhinted (X=0, all) | 5518.1 | 3243.2 | 5203 | 549 | 16355 | 1553 | 198 |
| Hinted, followed (X=1, Y=1) | 6035.3 | 3310.8 | 5558 | 439 | 16342 | 1244 | 183 |
| Hinted, not followed (X=1, Y=0) | 6381.9 | 3980.0 | 6019 | 551 | 16221 | 318 | 77 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 99 | 326 | 670 | 1322 | 696 | 0 | 53 | 3168 | 53 (1.7%) | 16384 |

## wmdp × professor

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 2036.8 | 2588.8 | 1196 | 313 | 16135 | 1513 | 196 |
| Unhinted (X=0, all) | 1856.9 | 2547.8 | 1018 | 352 | 16265 | 1502 | 196 |
| Hinted, followed (X=1, Y=1) | 2324.1 | 2788.8 | 1308 | 397 | 15459 | 521 | 98 |
| Hinted, not followed (X=1, Y=0) | 1885.9 | 2465.4 | 1112 | 313 | 16135 | 992 | 161 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 132 | 1230 | 1067 | 295 | 140 | 164 | 0 | 172 | 3200 | 172 (5.4%) | 16384 |

## wmdp × grader_hacking

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 2164.1 | 2717.5 | 1249 | 334 | 16331 | 1515 | 197 |
| Unhinted (X=0, all) | 1863.8 | 2550.1 | 1019 | 352 | 16265 | 1505 | 197 |
| Hinted, followed (X=1, Y=1) | 2409.9 | 2805.5 | 1428 | 356 | 16331 | 518 | 103 |
| Hinted, not followed (X=1, Y=0) | 2036.4 | 2663.1 | 1111 | 334 | 16145 | 997 | 158 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 161 | 1195 | 994 | 364 | 146 | 165 | 0 | 175 | 3200 | 175 (5.5%) | 16384 |

## wmdp × unethical

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 2056.2 | 2517.8 | 1279 | 302 | 16377 | 1517 | 197 |
| Unhinted (X=0, all) | 1863.8 | 2550.1 | 1019 | 352 | 16265 | 1505 | 197 |
| Hinted, followed (X=1, Y=1) | 2170.1 | 2553.1 | 1344 | 437 | 16377 | 1124 | 174 |
| Hinted, not followed (X=1, Y=0) | 1730.6 | 2387.5 | 1036 | 302 | 15909 | 393 | 84 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 140 | 1198 | 1042 | 355 | 137 | 164 | 0 | 164 | 3200 | 164 (5.1%) | 16384 |

## codemmlu × professor

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 3006.8 | 2834.4 | 1930 | 262 | 15400 | 1585 | 200 |
| Unhinted (X=0, all) | 2692.5 | 2811.0 | 1485 | 261 | 14912 | 1587 | 200 |
| Hinted, followed (X=1, Y=1) | 4273.2 | 3172.4 | 3555 | 351 | 14473 | 321 | 71 |
| Hinted, not followed (X=1, Y=0) | 2685.1 | 2648.5 | 1592 | 262 | 15400 | 1264 | 178 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 486 | 660 | 599 | 614 | 607 | 206 | 0 | 28 | 3200 | 28 (0.9%) | 16384 |

## codemmlu × grader_hacking

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 3249.5 | 3089.4 | 2099 | 227 | 16087 | 1571 | 199 |
| Unhinted (X=0, all) | 2694.9 | 2817.7 | 1477 | 261 | 14912 | 1579 | 199 |
| Hinted, followed (X=1, Y=1) | 4556.2 | 3407.9 | 3716 | 276 | 15712 | 276 | 62 |
| Hinted, not followed (X=1, Y=0) | 2971.0 | 2944.5 | 1850 | 227 | 16087 | 1295 | 180 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 488 | 606 | 607 | 625 | 583 | 249 | 0 | 42 | 3200 | 42 (1.3%) | 16384 |

## codemmlu × unethical

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 3670.6 | 2766.7 | 2988 | 242 | 13376 | 1578 | 200 |
| Unhinted (X=0, all) | 2692.5 | 2811.0 | 1485 | 261 | 14912 | 1587 | 200 |
| Hinted, followed (X=1, Y=1) | 4358.7 | 2818.3 | 3922 | 311 | 13376 | 956 | 165 |
| Hinted, not followed (X=1, Y=0) | 2613.0 | 2318.3 | 1764 | 242 | 11566 | 622 | 125 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 381 | 494 | 667 | 673 | 728 | 222 | 0 | 35 | 3200 | 35 (1.1%) | 16384 |

## lsat_ar × professor

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 7103.5 | 3578.1 | 6586 | 1028 | 16209 | 1579 | 200 |
| Unhinted (X=0, all) | 6141.8 | 3526.7 | 5335 | 1064 | 16356 | 1557 | 200 |
| Hinted, followed (X=1, Y=1) | 10209.0 | 3005.2 | 10200 | 2302 | 16039 | 267 | 78 |
| Hinted, not followed (X=1, Y=0) | 6471.5 | 3348.8 | 5804 | 1028 | 16209 | 1312 | 192 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 171 | 758 | 1259 | 948 | 0 | 64 | 3200 | 64 (2.0%) | 16384 |

## lsat_ar × grader_hacking

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 7555.3 | 3492.8 | 7093 | 1187 | 16174 | 1532 | 200 |
| Unhinted (X=0, all) | 6141.8 | 3526.7 | 5335 | 1064 | 16356 | 1557 | 200 |
| Hinted, followed (X=1, Y=1) | 10190.2 | 3048.8 | 10306 | 1859 | 16174 | 274 | 92 |
| Hinted, not followed (X=1, Y=0) | 6981.4 | 3316.3 | 6356 | 1187 | 16149 | 1258 | 189 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 161 | 625 | 1295 | 1008 | 0 | 111 | 3200 | 111 (3.5%) | 16384 |

## lsat_ar × unethical

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 8407.7 | 3101.8 | 8308 | 1597 | 16221 | 1566 | 200 |
| Unhinted (X=0, all) | 6141.8 | 3526.7 | 5335 | 1064 | 16356 | 1557 | 200 |
| Hinted, followed (X=1, Y=1) | 9350.0 | 2852.9 | 9292 | 1709 | 16221 | 762 | 167 |
| Hinted, not followed (X=1, Y=0) | 7514.6 | 3065.4 | 6980 | 1597 | 15794 | 804 | 167 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 140 | 493 | 1295 | 1195 | 0 | 77 | 3200 | 77 (2.4%) | 16384 |

## lsat_lr × professor

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 2145.2 | 1757.9 | 1420 | 335 | 11210 | 1600 | 200 |
| Unhinted (X=0, all) | 1810.1 | 1496.0 | 1180 | 328 | 9462 | 1600 | 200 |
| Hinted, followed (X=1, Y=1) | 3089.3 | 1794.1 | 2822 | 547 | 9782 | 166 | 42 |
| Hinted, not followed (X=1, Y=0) | 2035.9 | 1721.1 | 1312 | 335 | 11210 | 1434 | 195 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 84 | 1170 | 888 | 651 | 395 | 12 | 0 | 0 | 3200 | 0 (0.0%) | 11210 |

## lsat_lr × grader_hacking

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 2421.8 | 1981.4 | 1735 | 376 | 10531 | 1596 | 200 |
| Unhinted (X=0, all) | 1810.1 | 1496.0 | 1180 | 328 | 9462 | 1600 | 200 |
| Hinted, followed (X=1, Y=1) | 3745.1 | 2291.7 | 3274 | 628 | 10531 | 203 | 44 |
| Hinted, not followed (X=1, Y=0) | 2229.0 | 1855.7 | 1427 | 376 | 9854 | 1393 | 191 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 98 | 1141 | 789 | 699 | 447 | 22 | 0 | 4 | 3200 | 4 (0.1%) | 16384 |

## lsat_lr × unethical

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 2707.0 | 1920.4 | 2430 | 347 | 11397 | 1594 | 200 |
| Unhinted (X=0, all) | 1810.1 | 1496.0 | 1180 | 328 | 9462 | 1600 | 200 |
| Hinted, followed (X=1, Y=1) | 3396.2 | 1827.1 | 3322 | 470 | 10846 | 592 | 116 |
| Hinted, not followed (X=1, Y=0) | 2299.7 | 1857.7 | 1465 | 347 | 11397 | 1002 | 158 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 80 | 1034 | 749 | 804 | 511 | 16 | 0 | 6 | 3200 | 6 (0.2%) | 16384 |

## medxpertqa × professor

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 2206.9 | 1197.3 | 1977 | 407 | 9534 | 1599 | 200 |
| Unhinted (X=0, all) | 2140.7 | 1130.7 | 1906 | 435 | 10149 | 1599 | 200 |
| Hinted, followed (X=1, Y=1) | 2000.2 | 984.9 | 1819 | 438 | 8193 | 613 | 111 |
| Hinted, not followed (X=1, Y=0) | 2335.3 | 1296.1 | 2071 | 407 | 9534 | 986 | 159 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 7 | 358 | 1376 | 1268 | 179 | 10 | 0 | 2 | 3200 | 2 (0.1%) | 16384 |

## medxpertqa × grader_hacking

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 2357.6 | 1382.2 | 2092 | 437 | 12143 | 1598 | 200 |
| Unhinted (X=0, all) | 2140.7 | 1130.7 | 1906 | 435 | 10149 | 1599 | 200 |
| Hinted, followed (X=1, Y=1) | 2476.6 | 1410.5 | 2173 | 453 | 10074 | 587 | 117 |
| Hinted, not followed (X=1, Y=0) | 2288.5 | 1361.4 | 2037 | 437 | 12143 | 1011 | 164 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 11 | 366 | 1286 | 1302 | 218 | 14 | 0 | 3 | 3200 | 3 (0.1%) | 16384 |

## medxpertqa × unethical

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 1954.6 | 1140.7 | 1692 | 413 | 12238 | 1600 | 200 |
| Unhinted (X=0, all) | 2140.7 | 1130.7 | 1906 | 435 | 10149 | 1599 | 200 |
| Hinted, followed (X=1, Y=1) | 1958.9 | 1147.7 | 1697 | 468 | 12238 | 1459 | 196 |
| Hinted, not followed (X=1, Y=0) | 1909.6 | 1068.3 | 1610 | 413 | 5106 | 141 | 42 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 11 | 433 | 1472 | 1120 | 155 | 8 | 0 | 1 | 3200 | 1 (0.0%) | 16384 |

## lexam × professor

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 2626.2 | 1310.0 | 2534 | 443 | 6849 | 1597 | 200 |
| Unhinted (X=0, all) | 2475.4 | 1468.4 | 2237 | 355 | 7586 | 1594 | 200 |
| Hinted, followed (X=1, Y=1) | 2947.0 | 1336.4 | 3024 | 517 | 6760 | 500 | 116 |
| Hinted, not followed (X=1, Y=0) | 2480.0 | 1271.7 | 2325 | 443 | 6849 | 1097 | 179 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 61 | 458 | 836 | 1315 | 521 | 0 | 0 | 9 | 3200 | 9 (0.3%) | 16384 |

## lexam × grader_hacking

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 2827.3 | 1446.9 | 2770 | 406 | 7874 | 1599 | 200 |
| Unhinted (X=0, all) | 2475.4 | 1468.4 | 2237 | 355 | 7586 | 1594 | 200 |
| Hinted, followed (X=1, Y=1) | 3402.4 | 1425.3 | 3606 | 562 | 7874 | 620 | 131 |
| Hinted, not followed (X=1, Y=0) | 2463.1 | 1338.8 | 2259 | 406 | 6448 | 979 | 168 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 73 | 443 | 797 | 1255 | 625 | 0 | 0 | 7 | 3200 | 7 (0.2%) | 16384 |

## lexam × unethical

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 2451.8 | 1043.5 | 2392 | 390 | 8128 | 1600 | 200 |
| Unhinted (X=0, all) | 2475.4 | 1468.4 | 2237 | 355 | 7586 | 1594 | 200 |
| Hinted, followed (X=1, Y=1) | 2446.5 | 1033.7 | 2380 | 446 | 8128 | 1307 | 194 |
| Hinted, not followed (X=1, Y=0) | 2475.9 | 1087.9 | 2449 | 390 | 5514 | 293 | 90 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 66 | 398 | 881 | 1475 | 374 | 0 | 0 | 6 | 3200 | 6 (0.2%) | 16384 |

## Summary

| Dataset | Hint type | Hinted (mean (std) · median) | Unhinted (mean (std) · median) | Hinted & followed (mean (std) · median) | Hinted & not followed (mean (std) · median) | Gap (Hinted mean − Unhinted mean) |
|---|---|---|---|---|---|---|
| mmlu_pro | professor | 3002.9 (3015.9) · 1868 | 2411.6 (2681.2) · 1350 | 3737.7 (3899.4) · 1981 | 2808.1 (2702.7) · 1850 | 591.4 |
| mmlu_pro | grader_hacking | 3303.5 (3185.4) · 2114 | 2411.6 (2681.2) · 1350 | 4015.2 (3415.7) · 2902 | 3081.1 (3077.9) · 1934 | 891.9 |
| mmlu_pro | unethical | 3749.5 (3309.7) · 2418 | 2411.6 (2681.2) · 1350 | 3521.4 (3212.4) · 2172 | 4425.4 (3500.4) · 3254 | 1337.9 |
| gpqa_diamond | professor | 5878.0 (3131.4) · 5623 | 5518.1 (3243.2) · 5203 | 6307.2 (3080.3) · 6006 | 5507.5 (3129.6) · 5184 | 359.9 |
| gpqa_diamond | grader_hacking | 6056.5 (3315.8) · 5819 | 5518.1 (3243.2) · 5203 | 6557.1 (3187.0) · 6164 | 5656.0 (3363.9) · 5421 | 538.4 |
| gpqa_diamond | unethical | 6105.9 (3459.0) · 5668 | 5518.1 (3243.2) · 5203 | 6035.3 (3310.8) · 5558 | 6381.9 (3980.0) · 6019 | 587.8 |
| wmdp | professor | 2036.8 (2588.8) · 1196 | 1856.9 (2547.8) · 1018 | 2324.1 (2788.8) · 1308 | 1885.9 (2465.4) · 1112 | 179.9 |
| wmdp | grader_hacking | 2164.1 (2717.5) · 1249 | 1863.8 (2550.1) · 1019 | 2409.9 (2805.5) · 1428 | 2036.4 (2663.1) · 1111 | 300.3 |
| wmdp | unethical | 2056.2 (2517.8) · 1279 | 1863.8 (2550.1) · 1019 | 2170.1 (2553.1) · 1344 | 1730.6 (2387.5) · 1036 | 192.4 |
| codemmlu | professor | 3006.8 (2834.4) · 1930 | 2692.5 (2811.0) · 1485 | 4273.2 (3172.4) · 3555 | 2685.1 (2648.5) · 1592 | 314.3 |
| codemmlu | grader_hacking | 3249.5 (3089.4) · 2099 | 2694.9 (2817.7) · 1477 | 4556.2 (3407.9) · 3716 | 2971.0 (2944.5) · 1850 | 554.6 |
| codemmlu | unethical | 3670.6 (2766.7) · 2988 | 2692.5 (2811.0) · 1485 | 4358.7 (2818.3) · 3922 | 2613.0 (2318.3) · 1764 | 978.2 |
| lsat_ar | professor | 7103.5 (3578.1) · 6586 | 6141.8 (3526.7) · 5335 | 10209.0 (3005.2) · 10200 | 6471.5 (3348.8) · 5804 | 961.6 |
| lsat_ar | grader_hacking | 7555.3 (3492.8) · 7093 | 6141.8 (3526.7) · 5335 | 10190.2 (3048.8) · 10306 | 6981.4 (3316.3) · 6356 | 1413.5 |
| lsat_ar | unethical | 8407.7 (3101.8) · 8308 | 6141.8 (3526.7) · 5335 | 9350.0 (2852.9) · 9292 | 7514.6 (3065.4) · 6980 | 2265.9 |
| lsat_lr | professor | 2145.2 (1757.9) · 1420 | 1810.1 (1496.0) · 1180 | 3089.3 (1794.1) · 2822 | 2035.9 (1721.1) · 1312 | 335.1 |
| lsat_lr | grader_hacking | 2421.8 (1981.4) · 1735 | 1810.1 (1496.0) · 1180 | 3745.1 (2291.7) · 3274 | 2229.0 (1855.7) · 1427 | 611.7 |
| lsat_lr | unethical | 2707.0 (1920.4) · 2430 | 1810.1 (1496.0) · 1180 | 3396.2 (1827.1) · 3322 | 2299.7 (1857.7) · 1465 | 896.9 |
| medxpertqa | professor | 2206.9 (1197.3) · 1977 | 2140.7 (1130.7) · 1906 | 2000.2 (984.9) · 1819 | 2335.3 (1296.1) · 2071 | 66.2 |
| medxpertqa | grader_hacking | 2357.6 (1382.2) · 2092 | 2140.7 (1130.7) · 1906 | 2476.6 (1410.5) · 2173 | 2288.5 (1361.4) · 2037 | 216.9 |
| medxpertqa | unethical | 1954.6 (1140.7) · 1692 | 2140.7 (1130.7) · 1906 | 1958.9 (1147.7) · 1697 | 1909.6 (1068.3) · 1610 | -186.1 |
| lexam | professor | 2626.2 (1310.0) · 2534 | 2475.4 (1468.4) · 2237 | 2947.0 (1336.4) · 3024 | 2480.0 (1271.7) · 2325 | 150.8 |
| lexam | grader_hacking | 2827.3 (1446.9) · 2770 | 2475.4 (1468.4) · 2237 | 3402.4 (1425.3) · 3606 | 2463.1 (1338.8) · 2259 | 351.9 |
| lexam | unethical | 2451.8 (1043.5) · 2392 | 2475.4 (1468.4) · 2237 | 2446.5 (1033.7) · 2380 | 2475.9 (1087.9) · 2449 | -23.5 |
