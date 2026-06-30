# CoT token-length analysis: qwen3_4b

**HuggingFace ID:** `Qwen/Qwen3-4B`  
**Hint type:** `professor`  
**Datasets included:** mmlu_redux, gpqa_diamond, mathbench_primary, mathbench_middle, mathbench_high, mathbench_college, compmath, wmdp, codemmlu

---

## 1. Overall CoT token length (pooled across all datasets)

Output tokens in the chain-of-thought, pooled over every valid run (run level) across all datasets above. Same methodology as each dataset's `gmean2_summary.md`. Runs = pooled responses; Instances = distinct examples.

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 1884.6 | 2262.8 | 239 | 13338 | 1782 | 448 |
| Unhinted (X=0, all) | 1597.0 | 2176.0 | 200 | 16344 | 1785 | 448 |
| Hinted — hint followed (X=1, Y=1) | 3256.2 | 3188.1 | 302 | 13338 | 275 | 93 |
| Hinted — hint not followed (X=1, Y=0) | 1634.3 | 1949.2 | 239 | 12901 | 1507 | 400 |

---

## 2. CoT token length per dataset

Same four conditions, computed separately for each dataset.

### mmlu_redux

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 1419.4 | 1610.7 | 295 | 12901 | 199 | 50 |
| Unhinted (X=0, all) | 1114.4 | 1265.5 | 293 | 9137 | 199 | 50 |
| Hinted — hint followed (X=1, Y=1) | 1791.3 | 1963.2 | 575 | 9477 | 20 | 7 |
| Hinted — hint not followed (X=1, Y=0) | 1377.8 | 1567.6 | 295 | 12901 | 179 | 47 |

### gpqa_diamond

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 4376.4 | 3015.1 | 423 | 13338 | 197 | 50 |
| Unhinted (X=0, all) | 4315.9 | 3445.1 | 397 | 16344 | 199 | 50 |
| Hinted — hint followed (X=1, Y=1) | 5068.5 | 3526.1 | 423 | 13338 | 93 | 29 |
| Hinted — hint not followed (X=1, Y=0) | 3757.6 | 2318.1 | 609 | 11493 | 104 | 31 |

### mathbench_primary

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 573.3 | 410.4 | 251 | 2552 | 200 | 50 |
| Unhinted (X=0, all) | 510.4 | 580.0 | 225 | 5901 | 200 | 50 |
| Hinted — hint followed (X=1, Y=1) | 1188.2 | 668.6 | 392 | 2413 | 13 | 5 |
| Hinted — hint not followed (X=1, Y=0) | 530.6 | 350.8 | 251 | 2552 | 187 | 49 |

### mathbench_middle

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 1694.4 | 1794.5 | 239 | 11789 | 200 | 50 |
| Unhinted (X=0, all) | 1136.7 | 1200.5 | 200 | 8401 | 200 | 50 |
| Hinted — hint followed (X=1, Y=1) | 846.8 | 591.1 | 302 | 2295 | 10 | 4 |
| Hinted — hint not followed (X=1, Y=0) | 1739.0 | 1825.9 | 239 | 11789 | 190 | 48 |

### mathbench_high

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 2119.1 | 2718.1 | 252 | 11863 | 195 | 49 |
| Unhinted (X=0, all) | 1556.9 | 2034.5 | 240 | 15205 | 196 | 49 |
| Hinted — hint followed (X=1, Y=1) | 2618.6 | 2284.1 | 717 | 8850 | 11 | 4 |
| Hinted — hint not followed (X=1, Y=0) | 2089.2 | 2744.3 | 252 | 11863 | 184 | 47 |

### mathbench_college

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 1645.0 | 1557.0 | 379 | 10546 | 200 | 50 |
| Unhinted (X=0, all) | 1340.9 | 1139.9 | 303 | 7718 | 199 | 50 |
| Hinted — hint followed (X=1, Y=1) | 3276.0 | 2620.5 | 512 | 6303 | 7 | 3 |
| Hinted — hint not followed (X=1, Y=0) | 1585.8 | 1482.4 | 379 | 10546 | 193 | 50 |

### compmath

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 1122.2 | 908.4 | 320 | 4694 | 200 | 50 |
| Unhinted (X=0, all) | 798.3 | 498.3 | 275 | 2926 | 200 | 50 |
| Hinted — hint followed (X=1, Y=1) | 1983.1 | 1467.0 | 621 | 4487 | 12 | 3 |
| Hinted — hint not followed (X=1, Y=0) | 1067.3 | 837.3 | 320 | 4694 | 188 | 47 |

### wmdp

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 1946.3 | 2833.2 | 406 | 13326 | 194 | 49 |
| Unhinted (X=0, all) | 1833.3 | 2926.8 | 378 | 14973 | 194 | 49 |
| Hinted — hint followed (X=1, Y=1) | 2363.8 | 3090.4 | 466 | 13326 | 76 | 26 |
| Hinted — hint not followed (X=1, Y=0) | 1677.4 | 2633.0 | 406 | 12775 | 118 | 37 |

### codemmlu

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 2111.1 | 1960.4 | 274 | 9175 | 197 | 50 |
| Unhinted (X=0, all) | 1783.9 | 1891.4 | 256 | 9964 | 198 | 50 |
| Hinted — hint followed (X=1, Y=1) | 3307.8 | 2368.3 | 489 | 9175 | 33 | 12 |
| Hinted — hint not followed (X=1, Y=0) | 1870.3 | 1781.0 | 274 | 7544 | 164 | 44 |

---

## 3. Token-length buckets per dataset

Number of individual responses (every sample, both arms) whose `token_count` falls in each length band. **Truncated responses (those that hit `max_new_tokens`) are counted in the top `32768+` bucket regardless of their cap** — their true length is censored above the cap, so they represent the overflow tail, not a normal-length trace at the cap value. The `Truncated` column repeats that count explicitly. A high truncation rate means the cap is too low — use this to pick `max_new_tokens`.

| Dataset | 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mmlu_redux | 112 | 130 | 94 | 48 | 8 | 6 | 0 | 2 | 400 | 2 (0.5%) | 16384 |
| gpqa_diamond | 6 | 36 | 79 | 104 | 114 | 57 | 0 | 4 | 400 | 4 (1.0%) | 16384 |
| mathbench_primary | 302 | 62 | 30 | 5 | 1 | 0 | 0 | 0 | 400 | 0 (0.0%) | 5901 |
| mathbench_middle | 120 | 114 | 76 | 64 | 22 | 4 | 0 | 0 | 400 | 0 (0.0%) | 11789 |
| mathbench_high | 100 | 137 | 71 | 27 | 39 | 21 | 0 | 5 | 400 | 5 (1.2%) | 16384 |
| mathbench_college | 51 | 159 | 103 | 66 | 19 | 1 | 0 | 1 | 400 | 1 (0.2%) | 16384 |
| compmath | 124 | 152 | 90 | 31 | 3 | 0 | 0 | 0 | 400 | 0 (0.0%) | 4694 |
| wmdp | 45 | 189 | 101 | 11 | 16 | 27 | 0 | 11 | 400 | 11 (2.8%) | 16384 |
| codemmlu | 69 | 115 | 86 | 70 | 50 | 5 | 0 | 5 | 400 | 5 (1.2%) | 16384 |
| **ALL** | **929** | **1094** | **730** | **426** | **272** | **121** | **0** | **28** | **3600** | **28 (0.8%)** | **16384** |
