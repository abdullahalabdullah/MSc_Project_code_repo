# CoT token-length analysis: qwen3_4b_saferl

**HuggingFace ID:** `Qwen/Qwen3-4B-SafeRL`  
**Hint type:** `professor`  
**Datasets included:** mmlu_redux, gpqa_diamond, wmdp, codemmlu, lsat_ar, lsat_lr

---

## 1. Overall CoT token length (pooled across all datasets)

Output tokens in the chain-of-thought, pooled over every valid run (run level) across all datasets above. Same methodology as each dataset's `gmean2_summary.md`. Runs = pooled responses; Instances = distinct examples.

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 2412.5 | 2440.6 | 289 | 15911 | 1197 | 300 |
| Unhinted (X=0, all) | 2189.8 | 2400.1 | 289 | 16238 | 1194 | 300 |
| Hinted — hint followed (X=1, Y=1) | 3149.6 | 2953.4 | 382 | 15911 | 309 | 108 |
| Hinted — hint not followed (X=1, Y=0) | 2156.1 | 2179.0 | 289 | 12214 | 888 | 254 |

---

## 2. CoT token length per dataset

Same four conditions, computed separately for each dataset.

### mmlu_redux

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 1282.5 | 1559.3 | 289 | 12214 | 200 | 50 |
| Unhinted (X=0, all) | 1079.1 | 1441.9 | 289 | 11527 | 200 | 50 |
| Hinted — hint followed (X=1, Y=1) | 1948.1 | 2453.3 | 389 | 9615 | 19 | 7 |
| Hinted — hint not followed (X=1, Y=0) | 1212.6 | 1426.3 | 289 | 12214 | 181 | 49 |

### gpqa_diamond

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 3365.3 | 2569.0 | 382 | 12428 | 199 | 50 |
| Unhinted (X=0, all) | 3215.9 | 2484.1 | 394 | 13790 | 198 | 50 |
| Hinted — hint followed (X=1, Y=1) | 3560.8 | 2817.5 | 382 | 12428 | 99 | 31 |
| Hinted — hint not followed (X=1, Y=0) | 3171.9 | 2294.6 | 528 | 11012 | 100 | 32 |

### wmdp

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 1605.5 | 2332.8 | 408 | 15911 | 198 | 50 |
| Unhinted (X=0, all) | 1390.9 | 2098.6 | 303 | 13694 | 196 | 50 |
| Hinted — hint followed (X=1, Y=1) | 2166.0 | 2898.8 | 467 | 15911 | 74 | 24 |
| Hinted — hint not followed (X=1, Y=0) | 1271.1 | 1851.3 | 408 | 10779 | 124 | 37 |

### codemmlu

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 1818.7 | 1680.3 | 330 | 7198 | 200 | 50 |
| Unhinted (X=0, all) | 1572.4 | 1633.1 | 310 | 9829 | 200 | 50 |
| Hinted — hint followed (X=1, Y=1) | 2849.2 | 2068.1 | 398 | 7198 | 44 | 15 |
| Hinted — hint not followed (X=1, Y=0) | 1528.0 | 1432.4 | 330 | 6496 | 156 | 42 |

### lsat_ar

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 4968.0 | 2577.4 | 1094 | 14192 | 200 | 50 |
| Unhinted (X=0, all) | 4618.9 | 2918.9 | 984 | 16238 | 200 | 50 |
| Hinted — hint followed (X=1, Y=1) | 6642.1 | 2744.1 | 2515 | 14192 | 38 | 19 |
| Hinted — hint not followed (X=1, Y=0) | 4575.3 | 2379.8 | 1094 | 11753 | 162 | 48 |

### lsat_lr

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 1432.0 | 1107.2 | 386 | 5446 | 200 | 50 |
| Unhinted (X=0, all) | 1256.1 | 858.4 | 322 | 4261 | 200 | 50 |
| Hinted — hint followed (X=1, Y=1) | 1304.2 | 802.4 | 520 | 4356 | 35 | 12 |
| Hinted — hint not followed (X=1, Y=0) | 1459.1 | 1161.8 | 386 | 5446 | 165 | 46 |

---

## 3. Token-length buckets per dataset

Number of individual responses (every sample, both arms) whose `token_count` falls in each length band. **Truncated responses (those that hit `max_new_tokens`) are counted in the top `32768+` bucket regardless of their cap** — their true length is censored above the cap, so they represent the overflow tail, not a normal-length trace at the cap value. The `Truncated` column repeats that count explicitly. A high truncation rate means the cap is too low — use this to pick `max_new_tokens`.

| Dataset | 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mmlu_redux | 116 | 147 | 94 | 29 | 8 | 6 | 0 | 0 | 400 | 0 (0.0%) | 12214 |
| gpqa_diamond | 9 | 57 | 93 | 123 | 90 | 25 | 0 | 3 | 400 | 3 (0.8%) | 16384 |
| wmdp | 51 | 237 | 60 | 9 | 23 | 14 | 0 | 6 | 400 | 6 (1.5%) | 16384 |
| codemmlu | 83 | 126 | 72 | 82 | 35 | 2 | 0 | 0 | 400 | 0 (0.0%) | 9829 |
| lsat_ar | 0 | 2 | 64 | 130 | 149 | 55 | 0 | 0 | 400 | 0 (0.0%) | 16238 |
| lsat_lr | 32 | 188 | 102 | 67 | 11 | 0 | 0 | 0 | 400 | 0 (0.0%) | 5446 |
| **ALL** | **291** | **757** | **485** | **440** | **316** | **102** | **0** | **9** | **2400** | **9 (0.4%)** | **16384** |
