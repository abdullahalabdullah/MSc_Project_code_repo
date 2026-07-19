# CoT token-length overall summary — per (dataset × hint type)

- **Model:** `er-abd/anti-syc-dpo-qwen3-4B-step-520` (`er-abd_anti-syc-dpo-qwen3-4B-step-520`)
- **Datasets:** mmlu_pro, gpqa_diamond, wmdp, codemmlu, lsat_ar, lsat_lr, medxpertqa, lexam
- **Hint types:** professor, grader_hacking, unethical

`token_count` is the full completion (thinking trace + final answer). Condition tables use valid runs only (not truncated, answer parsed), so followed + not-followed partition hinted exactly; bucket tables use every response. Use the truncation rate to set `max_new_tokens`.

## mmlu_pro × professor

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 3074.1 | 2928.0 | 1965 | 356 | 15937 | 1592 | 200 |
| Unhinted (X=0, all) | 2562.7 | 2816.4 | 1434 | 278 | 16347 | 1587 | 200 |
| Hinted, followed (X=1, Y=1) | 3521.9 | 3467.2 | 1961 | 365 | 14824 | 279 | 62 |
| Hinted, not followed (X=1, Y=0) | 2979.0 | 2792.4 | 1966 | 356 | 15937 | 1313 | 183 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 121 | 871 | 841 | 696 | 400 | 250 | 0 | 21 | 3200 | 21 (0.7%) | 16384 |

## mmlu_pro × grader_hacking

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 3599.0 | 3295.1 | 2430 | 325 | 16133 | 1574 | 200 |
| Unhinted (X=0, all) | 2562.7 | 2816.4 | 1434 | 278 | 16347 | 1587 | 200 |
| Hinted, followed (X=1, Y=1) | 4461.1 | 3545.6 | 3316 | 334 | 15620 | 370 | 98 |
| Hinted, not followed (X=1, Y=0) | 3334.0 | 3168.9 | 2182 | 325 | 16133 | 1204 | 182 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 141 | 839 | 700 | 711 | 477 | 293 | 0 | 39 | 3200 | 39 (1.2%) | 16384 |

## mmlu_pro × unethical

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 3855.5 | 3314.5 | 2506 | 393 | 16155 | 1576 | 200 |
| Unhinted (X=0, all) | 2562.7 | 2816.4 | 1434 | 278 | 16347 | 1587 | 200 |
| Hinted, followed (X=1, Y=1) | 3660.1 | 3230.8 | 2247 | 393 | 16081 | 1145 | 190 |
| Hinted, not followed (X=1, Y=0) | 4374.7 | 3477.9 | 3233 | 397 | 16155 | 431 | 108 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 102 | 726 | 802 | 716 | 533 | 284 | 0 | 37 | 3200 | 37 (1.2%) | 16384 |

## gpqa_diamond × professor

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 6227.3 | 3296.6 | 6033 | 571 | 16323 | 1560 | 198 |
| Unhinted (X=0, all) | 5816.5 | 3351.9 | 5504 | 503 | 16365 | 1546 | 198 |
| Hinted, followed (X=1, Y=1) | 6699.2 | 3241.2 | 6420 | 749 | 16323 | 733 | 131 |
| Hinted, not followed (X=1, Y=0) | 5809.0 | 3290.7 | 5453 | 571 | 16103 | 827 | 144 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 77 | 267 | 665 | 1367 | 729 | 0 | 62 | 3168 | 62 (2.0%) | 16384 |

## gpqa_diamond × grader_hacking

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 6362.4 | 3336.9 | 6151 | 558 | 16297 | 1549 | 198 |
| Unhinted (X=0, all) | 5816.5 | 3351.9 | 5504 | 503 | 16365 | 1546 | 198 |
| Hinted, followed (X=1, Y=1) | 6800.5 | 3201.9 | 6460 | 798 | 16297 | 664 | 133 |
| Hinted, not followed (X=1, Y=0) | 6033.7 | 3399.6 | 5761 | 558 | 16159 | 885 | 161 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 91 | 232 | 665 | 1342 | 764 | 0 | 73 | 3168 | 73 (2.3%) | 16384 |

## gpqa_diamond × unethical

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 6204.0 | 3372.3 | 5754 | 441 | 16375 | 1553 | 198 |
| Unhinted (X=0, all) | 5816.5 | 3351.9 | 5504 | 503 | 16365 | 1546 | 198 |
| Hinted, followed (X=1, Y=1) | 6151.4 | 3253.4 | 5675 | 441 | 16375 | 1243 | 184 |
| Hinted, not followed (X=1, Y=0) | 6415.1 | 3810.2 | 6240 | 472 | 15656 | 310 | 76 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 89 | 248 | 677 | 1326 | 756 | 0 | 69 | 3168 | 69 (2.2%) | 16384 |

## wmdp × professor

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 2216.7 | 2847.1 | 1304 | 324 | 16351 | 1514 | 197 |
| Unhinted (X=0, all) | 1982.6 | 2695.0 | 1092 | 349 | 16330 | 1503 | 197 |
| Hinted, followed (X=1, Y=1) | 2618.1 | 3259.3 | 1385 | 408 | 16351 | 471 | 90 |
| Hinted, not followed (X=1, Y=0) | 2035.4 | 2621.6 | 1255 | 324 | 16297 | 1043 | 165 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 168 | 1076 | 1079 | 400 | 107 | 192 | 0 | 178 | 3200 | 178 (5.6%) | 16384 |

## wmdp × grader_hacking

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 2327.1 | 2769.2 | 1470 | 352 | 16244 | 1489 | 198 |
| Unhinted (X=0, all) | 2015.1 | 2765.0 | 1093 | 349 | 16330 | 1507 | 198 |
| Hinted, followed (X=1, Y=1) | 2620.1 | 2861.3 | 1701 | 445 | 15630 | 483 | 97 |
| Hinted, not followed (X=1, Y=0) | 2186.4 | 2714.1 | 1341 | 352 | 16244 | 1006 | 161 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 196 | 991 | 989 | 511 | 123 | 187 | 0 | 203 | 3200 | 203 (6.3%) | 16384 |

## wmdp × unethical

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 2200.4 | 2659.5 | 1384 | 323 | 16335 | 1512 | 197 |
| Unhinted (X=0, all) | 1995.8 | 2731.5 | 1092 | 349 | 16330 | 1504 | 197 |
| Hinted, followed (X=1, Y=1) | 2315.4 | 2741.6 | 1427 | 397 | 16335 | 1106 | 172 |
| Hinted, not followed (X=1, Y=0) | 1887.0 | 2397.3 | 1008 | 323 | 15356 | 406 | 96 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 156 | 1064 | 1036 | 478 | 103 | 184 | 0 | 179 | 3200 | 179 (5.6%) | 16384 |

## codemmlu × professor

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 3272.3 | 2968.2 | 2146 | 238 | 14788 | 1570 | 200 |
| Unhinted (X=0, all) | 2768.6 | 2860.5 | 1598 | 249 | 14313 | 1563 | 200 |
| Hinted, followed (X=1, Y=1) | 4516.2 | 3325.2 | 3814 | 351 | 14788 | 312 | 71 |
| Hinted, not followed (X=1, Y=0) | 2963.7 | 2789.6 | 1870 | 238 | 14412 | 1258 | 182 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 508 | 573 | 576 | 578 | 669 | 229 | 0 | 67 | 3200 | 67 (2.1%) | 16384 |

## codemmlu × grader_hacking

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 3387.2 | 3101.4 | 2236 | 211 | 15518 | 1552 | 200 |
| Unhinted (X=0, all) | 2768.6 | 2860.5 | 1598 | 249 | 14313 | 1563 | 200 |
| Hinted, followed (X=1, Y=1) | 4622.4 | 3399.6 | 3856 | 280 | 14967 | 301 | 75 |
| Hinted, not followed (X=1, Y=0) | 3090.0 | 2950.5 | 2000 | 211 | 15518 | 1251 | 177 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 494 | 555 | 575 | 598 | 635 | 258 | 0 | 85 | 3200 | 85 (2.7%) | 16384 |

## codemmlu × unethical

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 3789.9 | 2862.9 | 2951 | 231 | 14617 | 1553 | 200 |
| Unhinted (X=0, all) | 2768.6 | 2860.5 | 1598 | 249 | 14313 | 1563 | 200 |
| Hinted, followed (X=1, Y=1) | 4462.5 | 2876.9 | 4038 | 363 | 14617 | 918 | 162 |
| Hinted, not followed (X=1, Y=0) | 2817.4 | 2547.6 | 1941 | 231 | 13344 | 635 | 129 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 378 | 447 | 638 | 679 | 719 | 255 | 0 | 84 | 3200 | 84 (2.6%) | 16384 |

## lsat_ar × professor

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 7340.5 | 3604.8 | 6733 | 1317 | 16374 | 1577 | 200 |
| Unhinted (X=0, all) | 6241.3 | 3579.2 | 5382 | 1123 | 16331 | 1560 | 200 |
| Hinted, followed (X=1, Y=1) | 10334.6 | 3165.7 | 10386 | 2905 | 16374 | 230 | 64 |
| Hinted, not followed (X=1, Y=0) | 6829.2 | 3423.0 | 6154 | 1317 | 16195 | 1347 | 193 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 128 | 742 | 1265 | 1002 | 0 | 63 | 3200 | 63 (2.0%) | 16384 |

## lsat_ar × grader_hacking

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 8089.6 | 3596.5 | 7659 | 1390 | 16379 | 1511 | 200 |
| Unhinted (X=0, all) | 6241.3 | 3579.2 | 5382 | 1123 | 16331 | 1560 | 200 |
| Hinted, followed (X=1, Y=1) | 10655.8 | 3102.9 | 10794 | 2054 | 16330 | 273 | 85 |
| Hinted, not followed (X=1, Y=0) | 7523.7 | 3449.4 | 6887 | 1390 | 16379 | 1238 | 191 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 121 | 621 | 1228 | 1101 | 0 | 129 | 3200 | 129 (4.0%) | 16384 |

## lsat_ar × unethical

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 8553.1 | 3111.6 | 8319 | 1541 | 16357 | 1515 | 200 |
| Unhinted (X=0, all) | 6241.3 | 3579.2 | 5382 | 1123 | 16331 | 1560 | 200 |
| Hinted, followed (X=1, Y=1) | 9707.1 | 2873.5 | 9832 | 1703 | 16357 | 671 | 152 |
| Hinted, not followed (X=1, Y=0) | 7635.7 | 2987.2 | 7092 | 1541 | 16114 | 844 | 162 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 101 | 515 | 1272 | 1187 | 0 | 125 | 3200 | 125 (3.9%) | 16384 |

## lsat_lr × professor

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 2302.8 | 1922.1 | 1478 | 323 | 10492 | 1597 | 200 |
| Unhinted (X=0, all) | 1913.4 | 1663.8 | 1179 | 274 | 10129 | 1598 | 200 |
| Hinted, followed (X=1, Y=1) | 3412.0 | 2133.4 | 3195 | 544 | 10492 | 168 | 38 |
| Hinted, not followed (X=1, Y=0) | 2172.4 | 1853.4 | 1343 | 323 | 10318 | 1429 | 191 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 143 | 1152 | 750 | 644 | 482 | 24 | 0 | 5 | 3200 | 5 (0.2%) | 16384 |

## lsat_lr × grader_hacking

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 2737.0 | 2210.9 | 2090 | 358 | 11811 | 1590 | 200 |
| Unhinted (X=0, all) | 1913.4 | 1663.8 | 1179 | 274 | 10129 | 1598 | 200 |
| Hinted, followed (X=1, Y=1) | 4112.3 | 2370.6 | 3796 | 504 | 11811 | 203 | 49 |
| Hinted, not followed (X=1, Y=0) | 2535.8 | 2113.6 | 1780 | 358 | 11742 | 1387 | 191 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 148 | 1061 | 681 | 680 | 571 | 47 | 0 | 12 | 3200 | 12 (0.4%) | 16384 |

## lsat_lr × unethical

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 2924.5 | 2058.8 | 2766 | 346 | 12752 | 1589 | 200 |
| Unhinted (X=0, all) | 1913.4 | 1663.8 | 1179 | 274 | 10129 | 1598 | 200 |
| Hinted, followed (X=1, Y=1) | 3577.3 | 1965.4 | 3433 | 499 | 10526 | 563 | 109 |
| Hinted, not followed (X=1, Y=0) | 2566.2 | 2021.5 | 2286 | 346 | 12752 | 1026 | 161 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 133 | 995 | 620 | 808 | 595 | 36 | 0 | 13 | 3200 | 13 (0.4%) | 16384 |

## medxpertqa × professor

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 2411.1 | 1281.1 | 2178 | 443 | 10205 | 1595 | 200 |
| Unhinted (X=0, all) | 2306.3 | 1252.3 | 2067 | 370 | 11311 | 1598 | 200 |
| Hinted, followed (X=1, Y=1) | 2147.9 | 1105.9 | 1936 | 528 | 7289 | 554 | 108 |
| Hinted, not followed (X=1, Y=0) | 2551.2 | 1344.9 | 2287 | 443 | 10205 | 1041 | 165 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 9 | 312 | 1182 | 1428 | 252 | 10 | 0 | 7 | 3200 | 7 (0.2%) | 16384 |

## medxpertqa × grader_hacking

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 2569.9 | 1425.7 | 2317 | 483 | 12191 | 1590 | 200 |
| Unhinted (X=0, all) | 2306.3 | 1252.3 | 2067 | 370 | 11311 | 1598 | 200 |
| Hinted, followed (X=1, Y=1) | 2710.4 | 1609.7 | 2374 | 483 | 12191 | 528 | 111 |
| Hinted, not followed (X=1, Y=0) | 2500.0 | 1320.0 | 2292 | 498 | 10348 | 1062 | 172 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 12 | 303 | 1137 | 1434 | 286 | 16 | 0 | 12 | 3200 | 12 (0.4%) | 16384 |

## medxpertqa × unethical

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 2099.1 | 1227.6 | 1813 | 339 | 11394 | 1597 | 200 |
| Unhinted (X=0, all) | 2306.3 | 1252.3 | 2067 | 370 | 11311 | 1598 | 200 |
| Hinted, followed (X=1, Y=1) | 2089.3 | 1232.9 | 1803 | 429 | 11394 | 1447 | 196 |
| Hinted, not followed (X=1, Y=0) | 2194.3 | 1175.3 | 2074 | 339 | 6196 | 150 | 41 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 13 | 390 | 1325 | 1265 | 192 | 10 | 0 | 5 | 3200 | 5 (0.2%) | 16384 |

## lexam × professor

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 2773.1 | 1346.5 | 2608 | 371 | 7335 | 1579 | 200 |
| Unhinted (X=0, all) | 2579.8 | 1540.3 | 2308 | 318 | 7207 | 1563 | 200 |
| Hinted, followed (X=1, Y=1) | 3126.0 | 1361.3 | 3252 | 453 | 7335 | 479 | 110 |
| Hinted, not followed (X=1, Y=0) | 2619.5 | 1311.3 | 2444 | 371 | 7197 | 1100 | 176 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 66 | 408 | 756 | 1286 | 626 | 0 | 0 | 58 | 3200 | 58 (1.8%) | 16384 |

## lexam × grader_hacking

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 3073.8 | 1506.5 | 3154 | 396 | 7464 | 1572 | 200 |
| Unhinted (X=0, all) | 2579.8 | 1540.3 | 2308 | 318 | 7207 | 1563 | 200 |
| Hinted, followed (X=1, Y=1) | 3573.2 | 1400.6 | 3716 | 506 | 7464 | 578 | 134 |
| Hinted, not followed (X=1, Y=0) | 2783.4 | 1490.3 | 2643 | 396 | 7463 | 994 | 168 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 72 | 396 | 694 | 1225 | 748 | 0 | 0 | 65 | 3200 | 65 (2.0%) | 16384 |

## lexam × unethical

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 2583.7 | 1070.8 | 2554 | 413 | 6690 | 1593 | 200 |
| Unhinted (X=0, all) | 2579.8 | 1540.3 | 2308 | 318 | 7207 | 1563 | 200 |
| Hinted, followed (X=1, Y=1) | 2558.5 | 1081.0 | 2512 | 428 | 6690 | 1238 | 191 |
| Hinted, not followed (X=1, Y=0) | 2671.6 | 1030.9 | 2687 | 413 | 5836 | 355 | 102 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 59 | 362 | 805 | 1462 | 468 | 0 | 0 | 44 | 3200 | 44 (1.4%) | 16384 |

## Summary

| Dataset | Hint type | Hinted (mean (std) · median) | Unhinted (mean (std) · median) | Hinted & followed (mean (std) · median) | Hinted & not followed (mean (std) · median) | Gap (Hinted mean − Unhinted mean) |
|---|---|---|---|---|---|---|
| mmlu_pro | professor | 3074.1 (2928.0) · 1965 | 2562.7 (2816.4) · 1434 | 3521.9 (3467.2) · 1961 | 2979.0 (2792.4) · 1966 | 511.4 |
| mmlu_pro | grader_hacking | 3599.0 (3295.1) · 2430 | 2562.7 (2816.4) · 1434 | 4461.1 (3545.6) · 3316 | 3334.0 (3168.9) · 2182 | 1036.3 |
| mmlu_pro | unethical | 3855.5 (3314.5) · 2506 | 2562.7 (2816.4) · 1434 | 3660.1 (3230.8) · 2247 | 4374.7 (3477.9) · 3233 | 1292.8 |
| gpqa_diamond | professor | 6227.3 (3296.6) · 6033 | 5816.5 (3351.9) · 5504 | 6699.2 (3241.2) · 6420 | 5809.0 (3290.7) · 5453 | 410.8 |
| gpqa_diamond | grader_hacking | 6362.4 (3336.9) · 6151 | 5816.5 (3351.9) · 5504 | 6800.5 (3201.9) · 6460 | 6033.7 (3399.6) · 5761 | 546.0 |
| gpqa_diamond | unethical | 6204.0 (3372.3) · 5754 | 5816.5 (3351.9) · 5504 | 6151.4 (3253.4) · 5675 | 6415.1 (3810.2) · 6240 | 387.6 |
| wmdp | professor | 2216.7 (2847.1) · 1304 | 1982.6 (2695.0) · 1092 | 2618.1 (3259.3) · 1385 | 2035.4 (2621.6) · 1255 | 234.1 |
| wmdp | grader_hacking | 2327.1 (2769.2) · 1470 | 2015.1 (2765.0) · 1093 | 2620.1 (2861.3) · 1701 | 2186.4 (2714.1) · 1341 | 312.0 |
| wmdp | unethical | 2200.4 (2659.5) · 1384 | 1995.8 (2731.5) · 1092 | 2315.4 (2741.6) · 1427 | 1887.0 (2397.3) · 1008 | 204.6 |
| codemmlu | professor | 3272.3 (2968.2) · 2146 | 2768.6 (2860.5) · 1598 | 4516.2 (3325.2) · 3814 | 2963.7 (2789.6) · 1870 | 503.7 |
| codemmlu | grader_hacking | 3387.2 (3101.4) · 2236 | 2768.6 (2860.5) · 1598 | 4622.4 (3399.6) · 3856 | 3090.0 (2950.5) · 2000 | 618.6 |
| codemmlu | unethical | 3789.9 (2862.9) · 2951 | 2768.6 (2860.5) · 1598 | 4462.5 (2876.9) · 4038 | 2817.4 (2547.6) · 1941 | 1021.3 |
| lsat_ar | professor | 7340.5 (3604.8) · 6733 | 6241.3 (3579.2) · 5382 | 10334.6 (3165.7) · 10386 | 6829.2 (3423.0) · 6154 | 1099.2 |
| lsat_ar | grader_hacking | 8089.6 (3596.5) · 7659 | 6241.3 (3579.2) · 5382 | 10655.8 (3102.9) · 10794 | 7523.7 (3449.4) · 6887 | 1848.3 |
| lsat_ar | unethical | 8553.1 (3111.6) · 8319 | 6241.3 (3579.2) · 5382 | 9707.1 (2873.5) · 9832 | 7635.7 (2987.2) · 7092 | 2311.9 |
| lsat_lr | professor | 2302.8 (1922.1) · 1478 | 1913.4 (1663.8) · 1179 | 3412.0 (2133.4) · 3195 | 2172.4 (1853.4) · 1343 | 389.4 |
| lsat_lr | grader_hacking | 2737.0 (2210.9) · 2090 | 1913.4 (1663.8) · 1179 | 4112.3 (2370.6) · 3796 | 2535.8 (2113.6) · 1780 | 823.6 |
| lsat_lr | unethical | 2924.5 (2058.8) · 2766 | 1913.4 (1663.8) · 1179 | 3577.3 (1965.4) · 3433 | 2566.2 (2021.5) · 2286 | 1011.1 |
| medxpertqa | professor | 2411.1 (1281.1) · 2178 | 2306.3 (1252.3) · 2067 | 2147.9 (1105.9) · 1936 | 2551.2 (1344.9) · 2287 | 104.8 |
| medxpertqa | grader_hacking | 2569.9 (1425.7) · 2317 | 2306.3 (1252.3) · 2067 | 2710.4 (1609.7) · 2374 | 2500.0 (1320.0) · 2292 | 263.6 |
| medxpertqa | unethical | 2099.1 (1227.6) · 1813 | 2306.3 (1252.3) · 2067 | 2089.3 (1232.9) · 1803 | 2194.3 (1175.3) · 2074 | -207.2 |
| lexam | professor | 2773.1 (1346.5) · 2608 | 2579.8 (1540.3) · 2308 | 3126.0 (1361.3) · 3252 | 2619.5 (1311.3) · 2444 | 193.3 |
| lexam | grader_hacking | 3073.8 (1506.5) · 3154 | 2579.8 (1540.3) · 2308 | 3573.2 (1400.6) · 3716 | 2783.4 (1490.3) · 2643 | 494.0 |
| lexam | unethical | 2583.7 (1070.8) · 2554 | 2579.8 (1540.3) · 2308 | 2558.5 (1081.0) · 2512 | 2671.6 (1030.9) · 2687 | 3.9 |
