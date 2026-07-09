# CoT token-length overall summary — per (dataset × hint type)

- **Model:** `Qwen/Qwen3-4B` (`Qwen_Qwen3-4B`)
- **Datasets:** mmlu_pro, gpqa_diamond, wmdp, codemmlu, lsat_ar, lsat_lr, medxpertqa, lexam
- **Hint types:** professor, grader_hacking, unethical

`token_count` is the full completion (thinking trace + final answer). Condition tables use valid runs only (not truncated, answer parsed), so followed + not-followed partition hinted exactly; bucket tables use every response. Use the truncation rate to set `max_new_tokens`.

## mmlu_pro × professor

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 2424.9 | 2632.5 | 269 | 15011 | 1599 | 200 |
| Unhinted (X=0, all) | 2033.7 | 2443.5 | 259 | 15934 | 1594 | 200 |
| Hinted, followed (X=1, Y=1) | 2756.6 | 3118.1 | 326 | 14484 | 341 | 70 |
| Hinted, not followed (X=1, Y=0) | 2335.0 | 2478.4 | 269 | 15011 | 1258 | 180 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 251 | 1127 | 834 | 520 | 297 | 164 | 0 | 7 | 3200 | 7 (0.2%) | 16384 |

## mmlu_pro × grader_hacking

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 2728.5 | 2868.9 | 291 | 15506 | 1595 | 200 |
| Unhinted (X=0, all) | 2033.7 | 2443.5 | 259 | 15934 | 1594 | 200 |
| Hinted, followed (X=1, Y=1) | 3590.5 | 3212.4 | 300 | 15111 | 390 | 93 |
| Hinted, not followed (X=1, Y=0) | 2449.6 | 2691.5 | 291 | 15506 | 1205 | 185 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 282 | 1081 | 756 | 522 | 362 | 186 | 0 | 11 | 3200 | 11 (0.3%) | 16384 |

## mmlu_pro × unethical

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 3028.5 | 2951.3 | 293 | 15779 | 1595 | 200 |
| Unhinted (X=0, all) | 2033.7 | 2443.5 | 259 | 15934 | 1594 | 200 |
| Hinted, followed (X=1, Y=1) | 2952.8 | 2852.8 | 293 | 15779 | 1330 | 199 |
| Hinted, not followed (X=1, Y=0) | 3408.2 | 3383.8 | 295 | 14774 | 265 | 88 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 230 | 1002 | 830 | 491 | 459 | 177 | 0 | 11 | 3200 | 11 (0.3%) | 16384 |

## gpqa_diamond × professor

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 5058.2 | 3083.7 | 406 | 16374 | 1571 | 198 |
| Unhinted (X=0, all) | 4806.0 | 3174.4 | 428 | 16297 | 1566 | 198 |
| Hinted, followed (X=1, Y=1) | 5568.8 | 3220.4 | 406 | 16374 | 762 | 131 |
| Hinted, not followed (X=1, Y=0) | 4577.3 | 2869.1 | 406 | 15806 | 809 | 145 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 9 | 166 | 478 | 768 | 1276 | 440 | 0 | 31 | 3168 | 31 (1.0%) | 16384 |

## gpqa_diamond × grader_hacking

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 5215.2 | 3156.7 | 410 | 16313 | 1567 | 198 |
| Unhinted (X=0, all) | 4806.0 | 3174.4 | 428 | 16297 | 1566 | 198 |
| Hinted, followed (X=1, Y=1) | 5757.8 | 3191.1 | 410 | 16313 | 690 | 135 |
| Hinted, not followed (X=1, Y=0) | 4788.2 | 3064.3 | 452 | 16143 | 877 | 153 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 7 | 174 | 446 | 779 | 1250 | 477 | 0 | 35 | 3168 | 35 (1.1%) | 16384 |

## gpqa_diamond × unethical

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 5168.3 | 3193.7 | 435 | 16338 | 1570 | 198 |
| Unhinted (X=0, all) | 4806.0 | 3174.4 | 428 | 16297 | 1566 | 198 |
| Hinted, followed (X=1, Y=1) | 5294.7 | 3154.1 | 435 | 16338 | 1256 | 186 |
| Hinted, not followed (X=1, Y=0) | 4662.8 | 3304.5 | 443 | 14473 | 314 | 79 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 11 | 183 | 454 | 774 | 1218 | 496 | 0 | 32 | 3168 | 32 (1.0%) | 16384 |

## wmdp × professor

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 1800.8 | 2679.1 | 249 | 16375 | 1524 | 197 |
| Unhinted (X=0, all) | 1639.7 | 2571.3 | 312 | 16316 | 1516 | 197 |
| Hinted, followed (X=1, Y=1) | 2183.5 | 3274.6 | 388 | 16375 | 525 | 101 |
| Hinted, not followed (X=1, Y=0) | 1599.7 | 2281.6 | 249 | 16069 | 999 | 157 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 389 | 1465 | 758 | 161 | 107 | 163 | 0 | 157 | 3200 | 157 (4.9%) | 16384 |

## wmdp × grader_hacking

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 1820.4 | 2586.9 | 301 | 16242 | 1519 | 197 |
| Unhinted (X=0, all) | 1639.7 | 2571.3 | 312 | 16316 | 1516 | 197 |
| Hinted, followed (X=1, Y=1) | 2227.6 | 3074.5 | 338 | 16242 | 495 | 98 |
| Hinted, not followed (X=1, Y=0) | 1623.5 | 2290.5 | 301 | 15945 | 1024 | 167 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 428 | 1401 | 733 | 209 | 110 | 158 | 0 | 161 | 3200 | 161 (5.0%) | 16384 |

## wmdp × unethical

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 1757.4 | 2623.4 | 332 | 16285 | 1531 | 197 |
| Unhinted (X=0, all) | 1639.7 | 2571.3 | 312 | 16316 | 1516 | 197 |
| Hinted, followed (X=1, Y=1) | 1997.4 | 2847.7 | 359 | 16285 | 1162 | 181 |
| Hinted, not followed (X=1, Y=0) | 1002.0 | 1508.1 | 332 | 14596 | 369 | 89 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 384 | 1485 | 771 | 140 | 108 | 167 | 0 | 145 | 3200 | 145 (4.5%) | 16384 |

## codemmlu × professor

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 2328.7 | 2217.4 | 213 | 12636 | 1585 | 200 |
| Unhinted (X=0, all) | 2047.4 | 2235.2 | 208 | 14367 | 1586 | 200 |
| Hinted, followed (X=1, Y=1) | 3281.3 | 2446.2 | 277 | 10628 | 330 | 73 |
| Hinted, not followed (X=1, Y=0) | 2078.2 | 2083.1 | 213 | 12636 | 1255 | 181 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 683 | 690 | 656 | 605 | 449 | 88 | 0 | 29 | 3200 | 29 (0.9%) | 16384 |

## codemmlu × grader_hacking

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 2492.5 | 2457.4 | 208 | 14500 | 1578 | 200 |
| Unhinted (X=0, all) | 2047.4 | 2235.2 | 208 | 14367 | 1586 | 200 |
| Hinted, followed (X=1, Y=1) | 3597.5 | 2902.8 | 273 | 13168 | 292 | 72 |
| Hinted, not followed (X=1, Y=0) | 2241.6 | 2272.1 | 208 | 14500 | 1286 | 182 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 673 | 664 | 652 | 635 | 424 | 116 | 0 | 36 | 3200 | 36 (1.1%) | 16384 |

## codemmlu × unethical

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 2879.5 | 2370.0 | 219 | 14638 | 1588 | 200 |
| Unhinted (X=0, all) | 2047.4 | 2235.2 | 208 | 14367 | 1586 | 200 |
| Hinted, followed (X=1, Y=1) | 3339.7 | 2379.7 | 336 | 14638 | 1072 | 179 |
| Hinted, not followed (X=1, Y=0) | 1923.5 | 2043.4 | 219 | 11571 | 516 | 116 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 556 | 610 | 688 | 645 | 579 | 96 | 0 | 26 | 3200 | 26 (0.8%) | 16384 |

## lsat_ar × professor

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 6106.3 | 3341.4 | 959 | 16356 | 1589 | 200 |
| Unhinted (X=0, all) | 5540.3 | 3418.9 | 888 | 16182 | 1592 | 200 |
| Hinted, followed (X=1, Y=1) | 9030.4 | 3244.9 | 1691 | 16356 | 306 | 85 |
| Hinted, not followed (X=1, Y=0) | 5408.9 | 2966.0 | 959 | 16200 | 1283 | 190 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 7 | 345 | 885 | 1214 | 730 | 0 | 19 | 3200 | 19 (0.6%) | 16384 |

## lsat_ar × grader_hacking

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 6654.5 | 3353.2 | 855 | 16299 | 1577 | 200 |
| Unhinted (X=0, all) | 5540.3 | 3418.9 | 888 | 16182 | 1592 | 200 |
| Hinted, followed (X=1, Y=1) | 9247.9 | 3144.8 | 1729 | 16299 | 329 | 94 |
| Hinted, not followed (X=1, Y=0) | 5970.8 | 3060.5 | 855 | 15972 | 1248 | 194 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 7 | 281 | 807 | 1283 | 791 | 0 | 31 | 3200 | 31 (1.0%) | 16384 |

## lsat_ar × unethical

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 7195.4 | 2832.8 | 1133 | 15014 | 1579 | 200 |
| Unhinted (X=0, all) | 5540.3 | 3418.9 | 888 | 16182 | 1592 | 200 |
| Hinted, followed (X=1, Y=1) | 8097.4 | 2700.5 | 1133 | 15014 | 836 | 170 |
| Hinted, not followed (X=1, Y=0) | 6180.6 | 2629.1 | 1261 | 14793 | 743 | 158 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 6 | 238 | 651 | 1381 | 895 | 0 | 29 | 3200 | 29 (0.9%) | 16384 |

## lsat_lr × professor

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 1701.9 | 1447.9 | 303 | 9461 | 1598 | 200 |
| Unhinted (X=0, all) | 1480.9 | 1225.0 | 295 | 6901 | 1600 | 200 |
| Hinted, followed (X=1, Y=1) | 2415.7 | 1576.4 | 499 | 8081 | 200 | 49 |
| Hinted, not followed (X=1, Y=0) | 1599.7 | 1399.7 | 303 | 9461 | 1398 | 193 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 253 | 1371 | 776 | 573 | 223 | 2 | 0 | 2 | 3200 | 2 (0.1%) | 16384 |

## lsat_lr × grader_hacking

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 1868.8 | 1630.2 | 320 | 8752 | 1599 | 200 |
| Unhinted (X=0, all) | 1480.9 | 1225.0 | 295 | 6901 | 1600 | 200 |
| Hinted, followed (X=1, Y=1) | 2823.3 | 1812.1 | 443 | 8371 | 225 | 56 |
| Hinted, not followed (X=1, Y=0) | 1712.5 | 1543.9 | 320 | 8752 | 1374 | 189 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 263 | 1344 | 721 | 593 | 275 | 3 | 0 | 1 | 3200 | 1 (0.0%) | 16384 |

## lsat_lr × unethical

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 2035.0 | 1539.2 | 325 | 9214 | 1598 | 200 |
| Unhinted (X=0, all) | 1480.9 | 1225.0 | 295 | 6901 | 1600 | 200 |
| Hinted, followed (X=1, Y=1) | 2652.7 | 1567.6 | 391 | 9214 | 637 | 124 |
| Hinted, not followed (X=1, Y=0) | 1625.5 | 1375.6 | 325 | 7874 | 961 | 158 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 241 | 1236 | 688 | 777 | 253 | 3 | 0 | 2 | 3200 | 2 (0.1%) | 16384 |

## medxpertqa × professor

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 1690.1 | 984.7 | 380 | 7482 | 1599 | 200 |
| Unhinted (X=0, all) | 1627.4 | 911.1 | 365 | 8914 | 1600 | 200 |
| Hinted, followed (X=1, Y=1) | 1556.8 | 843.8 | 439 | 6464 | 626 | 118 |
| Hinted, not followed (X=1, Y=0) | 1775.8 | 1057.1 | 380 | 7482 | 973 | 158 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 29 | 789 | 1589 | 716 | 75 | 1 | 0 | 1 | 3200 | 1 (0.0%) | 16384 |

## medxpertqa × grader_hacking

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 1765.3 | 1062.6 | 371 | 9848 | 1600 | 200 |
| Unhinted (X=0, all) | 1627.4 | 911.1 | 365 | 8914 | 1600 | 200 |
| Hinted, followed (X=1, Y=1) | 1868.8 | 1089.1 | 425 | 8707 | 567 | 118 |
| Hinted, not followed (X=1, Y=0) | 1708.5 | 1044.0 | 371 | 9848 | 1033 | 165 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 770 | 1556 | 762 | 74 | 6 | 0 | 0 | 3200 | 0 (0.0%) | 9848 |

## medxpertqa × unethical

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 1542.7 | 934.1 | 351 | 9934 | 1599 | 200 |
| Unhinted (X=0, all) | 1627.4 | 911.1 | 365 | 8914 | 1600 | 200 |
| Hinted, followed (X=1, Y=1) | 1565.5 | 948.6 | 385 | 9934 | 1472 | 198 |
| Hinted, not followed (X=1, Y=0) | 1278.9 | 696.9 | 351 | 3486 | 127 | 41 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 34 | 871 | 1616 | 607 | 67 | 4 | 0 | 1 | 3200 | 1 (0.0%) | 16384 |

## lexam × professor

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 2064.8 | 1105.0 | 372 | 5681 | 1600 | 200 |
| Unhinted (X=0, all) | 1995.4 | 1264.8 | 329 | 7017 | 1593 | 200 |
| Hinted, followed (X=1, Y=1) | 2377.1 | 1127.7 | 444 | 5681 | 564 | 123 |
| Hinted, not followed (X=1, Y=0) | 1894.8 | 1054.8 | 372 | 5517 | 1036 | 177 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 139 | 698 | 958 | 1218 | 180 | 0 | 0 | 7 | 3200 | 7 (0.2%) | 16384 |

## lexam × grader_hacking

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 2304.8 | 1266.3 | 315 | 5919 | 1599 | 200 |
| Unhinted (X=0, all) | 1995.4 | 1264.8 | 329 | 7017 | 1593 | 200 |
| Hinted, followed (X=1, Y=1) | 2805.6 | 1187.4 | 494 | 5919 | 602 | 126 |
| Hinted, not followed (X=1, Y=0) | 2002.5 | 1216.1 | 315 | 5685 | 997 | 172 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 163 | 676 | 814 | 1282 | 257 | 0 | 0 | 8 | 3200 | 8 (0.2%) | 16384 |

## lexam × unethical

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 1887.7 | 873.7 | 329 | 5336 | 1599 | 200 |
| Unhinted (X=0, all) | 1995.4 | 1264.8 | 329 | 7017 | 1593 | 200 |
| Hinted, followed (X=1, Y=1) | 1893.8 | 863.4 | 329 | 5336 | 1385 | 195 |
| Hinted, not followed (X=1, Y=0) | 1848.2 | 939.0 | 337 | 4114 | 214 | 76 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 133 | 669 | 1064 | 1205 | 121 | 0 | 0 | 8 | 3200 | 8 (0.2%) | 16384 |

## Summary

| Dataset | Hint type | Hinted (mean and std) | Unhinted (mean and std) | Hinted & followed (mean and std) | Hinted & not followed (mean and std) |
|---|---|---|---|---|---|
| mmlu_pro | professor | 2424.9 (2632.5) | 2033.7 (2443.5) | 2756.6 (3118.1) | 2335.0 (2478.4) |
| mmlu_pro | grader_hacking | 2728.5 (2868.9) | 2033.7 (2443.5) | 3590.5 (3212.4) | 2449.6 (2691.5) |
| mmlu_pro | unethical | 3028.5 (2951.3) | 2033.7 (2443.5) | 2952.8 (2852.8) | 3408.2 (3383.8) |
| gpqa_diamond | professor | 5058.2 (3083.7) | 4806.0 (3174.4) | 5568.8 (3220.4) | 4577.3 (2869.1) |
| gpqa_diamond | grader_hacking | 5215.2 (3156.7) | 4806.0 (3174.4) | 5757.8 (3191.1) | 4788.2 (3064.3) |
| gpqa_diamond | unethical | 5168.3 (3193.7) | 4806.0 (3174.4) | 5294.7 (3154.1) | 4662.8 (3304.5) |
| wmdp | professor | 1800.8 (2679.1) | 1639.7 (2571.3) | 2183.5 (3274.6) | 1599.7 (2281.6) |
| wmdp | grader_hacking | 1820.4 (2586.9) | 1639.7 (2571.3) | 2227.6 (3074.5) | 1623.5 (2290.5) |
| wmdp | unethical | 1757.4 (2623.4) | 1639.7 (2571.3) | 1997.4 (2847.7) | 1002.0 (1508.1) |
| codemmlu | professor | 2328.7 (2217.4) | 2047.4 (2235.2) | 3281.3 (2446.2) | 2078.2 (2083.1) |
| codemmlu | grader_hacking | 2492.5 (2457.4) | 2047.4 (2235.2) | 3597.5 (2902.8) | 2241.6 (2272.1) |
| codemmlu | unethical | 2879.5 (2370.0) | 2047.4 (2235.2) | 3339.7 (2379.7) | 1923.5 (2043.4) |
| lsat_ar | professor | 6106.3 (3341.4) | 5540.3 (3418.9) | 9030.4 (3244.9) | 5408.9 (2966.0) |
| lsat_ar | grader_hacking | 6654.5 (3353.2) | 5540.3 (3418.9) | 9247.9 (3144.8) | 5970.8 (3060.5) |
| lsat_ar | unethical | 7195.4 (2832.8) | 5540.3 (3418.9) | 8097.4 (2700.5) | 6180.6 (2629.1) |
| lsat_lr | professor | 1701.9 (1447.9) | 1480.9 (1225.0) | 2415.7 (1576.4) | 1599.7 (1399.7) |
| lsat_lr | grader_hacking | 1868.8 (1630.2) | 1480.9 (1225.0) | 2823.3 (1812.1) | 1712.5 (1543.9) |
| lsat_lr | unethical | 2035.0 (1539.2) | 1480.9 (1225.0) | 2652.7 (1567.6) | 1625.5 (1375.6) |
| medxpertqa | professor | 1690.1 (984.7) | 1627.4 (911.1) | 1556.8 (843.8) | 1775.8 (1057.1) |
| medxpertqa | grader_hacking | 1765.3 (1062.6) | 1627.4 (911.1) | 1868.8 (1089.1) | 1708.5 (1044.0) |
| medxpertqa | unethical | 1542.7 (934.1) | 1627.4 (911.1) | 1565.5 (948.6) | 1278.9 (696.9) |
| lexam | professor | 2064.8 (1105.0) | 1995.4 (1264.8) | 2377.1 (1127.7) | 1894.8 (1054.8) |
| lexam | grader_hacking | 2304.8 (1266.3) | 1995.4 (1264.8) | 2805.6 (1187.4) | 2002.5 (1216.1) |
| lexam | unethical | 1887.7 (873.7) | 1995.4 (1264.8) | 1893.8 (863.4) | 1848.2 (939.0) |
