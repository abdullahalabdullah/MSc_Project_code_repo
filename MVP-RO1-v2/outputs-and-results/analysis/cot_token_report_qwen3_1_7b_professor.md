# CoT token-length analysis: qwen3_1_7b

**HuggingFace ID:** `Qwen/Qwen3-1.7B`  
**Hint type:** `professor`  
**Datasets included:** mmlu_redux, gpqa_diamond, mathbench_primary, mathbench_middle, mathbench_high, mathbench_college, compmath, wmdp, codemmlu

---

## 1. Overall CoT token length (pooled across all datasets)

Output tokens in the chain-of-thought, pooled over every valid run (run level) across all datasets above. Same methodology as each dataset's `gmean2_summary.md`. Runs = pooled responses; Instances = distinct examples.

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 2244.0 | 2690.8 | 237 | 15889 | 1772 | 450 |
| Unhinted (X=0, all) | 2076.5 | 2640.4 | 225 | 16061 | 1771 | 450 |
| Hinted — hint followed (X=1, Y=1) | 3239.8 | 3194.3 | 426 | 15507 | 253 | 107 |
| Hinted — hint not followed (X=1, Y=0) | 2078.1 | 2561.3 | 237 | 15889 | 1519 | 423 |

---

## 2. CoT token length per dataset

Same four conditions, computed separately for each dataset.

### mmlu_redux

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 1824.6 | 1865.7 | 315 | 12695 | 197 | 50 |
| Unhinted (X=0, all) | 1610.1 | 1713.0 | 351 | 15438 | 198 | 50 |
| Hinted — hint followed (X=1, Y=1) | 1679.6 | 1061.4 | 426 | 3826 | 27 | 11 |
| Hinted — hint not followed (X=1, Y=0) | 1847.6 | 1964.6 | 315 | 12695 | 170 | 47 |

### gpqa_diamond

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 5277.8 | 3420.2 | 578 | 15507 | 197 | 50 |
| Unhinted (X=0, all) | 5206.6 | 3585.7 | 608 | 15535 | 198 | 50 |
| Hinted — hint followed (X=1, Y=1) | 4536.1 | 3340.0 | 578 | 15507 | 72 | 29 |
| Hinted — hint not followed (X=1, Y=0) | 5705.1 | 3405.7 | 937 | 13468 | 125 | 41 |

### mathbench_primary

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 721.9 | 585.5 | 237 | 5423 | 200 | 50 |
| Unhinted (X=0, all) | 664.2 | 608.3 | 225 | 5431 | 199 | 50 |
| Hinted — hint followed (X=1, Y=1) | 1402.5 | 1204.4 | 601 | 4517 | 11 | 3 |
| Hinted — hint not followed (X=1, Y=0) | 682.3 | 507.0 | 237 | 5423 | 189 | 48 |

### mathbench_middle

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 1644.5 | 1908.2 | 315 | 14504 | 197 | 50 |
| Unhinted (X=0, all) | 1447.7 | 1890.7 | 262 | 15261 | 200 | 50 |
| Hinted — hint followed (X=1, Y=1) | 2108.1 | 2519.5 | 522 | 8097 | 9 | 3 |
| Hinted — hint not followed (X=1, Y=0) | 1622.3 | 1879.9 | 315 | 14504 | 188 | 49 |

### mathbench_high

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 2345.4 | 3180.7 | 375 | 15889 | 197 | 50 |
| Unhinted (X=0, all) | 1871.2 | 2574.9 | 362 | 16061 | 194 | 50 |
| Hinted — hint followed (X=1, Y=1) | 5266.8 | 5631.5 | 1638 | 14607 | 8 | 4 |
| Hinted — hint not followed (X=1, Y=0) | 2221.7 | 2998.0 | 375 | 15889 | 189 | 49 |

### mathbench_college

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 2052.6 | 1916.0 | 451 | 13831 | 197 | 50 |
| Unhinted (X=0, all) | 2160.9 | 2526.7 | 403 | 14444 | 199 | 50 |
| Hinted — hint followed (X=1, Y=1) | 2836.4 | 1517.1 | 1036 | 4923 | 10 | 4 |
| Hinted — hint not followed (X=1, Y=0) | 2010.7 | 1929.3 | 451 | 13831 | 187 | 49 |

### compmath

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 1267.7 | 1030.0 | 347 | 9222 | 198 | 50 |
| Unhinted (X=0, all) | 1017.2 | 694.7 | 349 | 5689 | 195 | 50 |
| Hinted — hint followed (X=1, Y=1) | 1497.0 | 1415.0 | 546 | 6749 | 19 | 9 |
| Hinted — hint not followed (X=1, Y=0) | 1243.3 | 982.6 | 347 | 9222 | 179 | 47 |

### wmdp

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 2601.3 | 3555.6 | 312 | 15203 | 199 | 50 |
| Unhinted (X=0, all) | 2171.6 | 2964.2 | 304 | 15792 | 194 | 50 |
| Hinted — hint followed (X=1, Y=1) | 3352.1 | 4007.3 | 537 | 15203 | 54 | 25 |
| Hinted — hint not followed (X=1, Y=0) | 2321.6 | 3344.1 | 312 | 14810 | 145 | 48 |

### codemmlu

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 2493.6 | 2338.6 | 246 | 11948 | 190 | 50 |
| Unhinted (X=0, all) | 2542.8 | 2673.8 | 248 | 14396 | 194 | 50 |
| Hinted — hint followed (X=1, Y=1) | 3101.8 | 2318.6 | 454 | 8886 | 43 | 19 |
| Hinted — hint not followed (X=1, Y=0) | 2315.7 | 2322.2 | 246 | 11948 | 147 | 45 |

---

## 3. Token-length buckets per dataset

Number of individual responses (every sample, both arms) whose `token_count` falls in each length band. **Truncated responses (those that hit `max_new_tokens`) are counted in the top `32768+` bucket regardless of their cap** — their true length is censored above the cap, so they represent the overflow tail, not a normal-length trace at the cap value. The `Truncated` column repeats that count explicitly. A high truncation rate means the cap is too low — use this to pick `max_new_tokens`.

| Dataset | 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mmlu_redux | 44 | 137 | 115 | 76 | 17 | 6 | 0 | 5 | 400 | 5 (1.2%) | 16384 |
| gpqa_diamond | 0 | 27 | 61 | 97 | 117 | 93 | 0 | 5 | 400 | 5 (1.2%) | 16384 |
| mathbench_primary | 206 | 141 | 39 | 10 | 3 | 0 | 0 | 1 | 400 | 1 (0.2%) | 16384 |
| mathbench_middle | 76 | 132 | 122 | 39 | 20 | 8 | 0 | 3 | 400 | 3 (0.8%) | 16384 |
| mathbench_high | 55 | 133 | 110 | 45 | 29 | 19 | 0 | 9 | 400 | 9 (2.2%) | 16384 |
| mathbench_college | 12 | 143 | 110 | 89 | 26 | 16 | 0 | 4 | 400 | 4 (1.0%) | 16384 |
| compmath | 43 | 204 | 105 | 37 | 3 | 1 | 0 | 7 | 400 | 7 (1.8%) | 16384 |
| wmdp | 30 | 156 | 107 | 49 | 12 | 39 | 0 | 7 | 400 | 7 (1.8%) | 16384 |
| codemmlu | 50 | 100 | 84 | 65 | 69 | 16 | 0 | 16 | 400 | 16 (4.0%) | 16384 |
| **ALL** | **516** | **1173** | **853** | **507** | **296** | **198** | **0** | **57** | **3600** | **57 (1.6%)** | **16384** |
