# CoT token-length analysis: qwen3_4b

**HuggingFace ID:** `Qwen/Qwen3-4B`  
**Hint type:** `professor`  
**Datasets included:** lsat_ar, lsat_lr, aqua_rat, logiqa_en

---

## 1. Overall CoT token length (pooled across all datasets)

Output tokens in the chain-of-thought, pooled over every valid run (run level) across all datasets above. Same methodology as each dataset's `gmean2_summary.md`. Runs = pooled responses; Instances = distinct examples.

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 3206.5 | 2941.3 | 409 | 15333 | 798 | 200 |
| Unhinted (X=0, all) | 2682.7 | 2841.4 | 383 | 16363 | 797 | 200 |
| Hinted — hint followed (X=1, Y=1) | 5268.7 | 3998.5 | 613 | 15333 | 125 | 46 |
| Hinted — hint not followed (X=1, Y=0) | 2823.5 | 2524.4 | 409 | 14672 | 673 | 180 |

---

## 2. CoT token length per dataset

Same four conditions, computed separately for each dataset.

### lsat_ar

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 5920.1 | 3251.1 | 1268 | 14672 | 199 | 50 |
| Unhinted (X=0, all) | 5274.1 | 3409.1 | 966 | 16267 | 198 | 50 |
| Hinted — hint followed (X=1, Y=1) | 8529.7 | 2801.0 | 3588 | 12441 | 38 | 16 |
| Hinted — hint not followed (X=1, Y=0) | 5304.2 | 3044.1 | 1268 | 14672 | 161 | 46 |

### lsat_lr

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 1859.3 | 1468.0 | 448 | 8254 | 200 | 50 |
| Unhinted (X=0, all) | 1683.5 | 1449.2 | 388 | 9117 | 200 | 50 |
| Hinted — hint followed (X=1, Y=1) | 2447.8 | 1977.4 | 882 | 8254 | 24 | 10 |
| Hinted — hint not followed (X=1, Y=0) | 1779.1 | 1372.2 | 448 | 6124 | 176 | 47 |

### aqua_rat

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 2547.4 | 2467.4 | 409 | 11187 | 200 | 50 |
| Unhinted (X=0, all) | 1670.5 | 2207.1 | 383 | 11144 | 199 | 50 |
| Hinted — hint followed (X=1, Y=1) | 6092.1 | 3974.5 | 613 | 10645 | 20 | 7 |
| Hinted — hint not followed (X=1, Y=0) | 2153.5 | 1879.3 | 409 | 11187 | 180 | 47 |

### logiqa_en

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 2509.4 | 2410.0 | 429 | 15333 | 199 | 50 |
| Unhinted (X=0, all) | 2123.4 | 2195.0 | 469 | 16363 | 200 | 50 |
| Hinted — hint followed (X=1, Y=1) | 3578.2 | 3685.8 | 745 | 15333 | 43 | 13 |
| Hinted — hint not followed (X=1, Y=0) | 2214.8 | 1825.9 | 429 | 8650 | 156 | 40 |

---

## 3. Token-length buckets per dataset

Number of individual responses (every sample, both arms) whose `token_count` falls in each length band. **Truncated responses (those that hit `max_new_tokens`) are counted in the top `32768+` bucket regardless of their cap** — their true length is censored above the cap, so they represent the overflow tail, not a normal-length trace at the cap value. The `Truncated` column repeats that count explicitly. A high truncation rate means the cap is too low — use this to pick `max_new_tokens`.

| Dataset | 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lsat_ar | 0 | 1 | 55 | 102 | 152 | 87 | 0 | 3 | 400 | 3 (0.8%) | 16384 |
| lsat_lr | 16 | 160 | 101 | 83 | 38 | 2 | 0 | 0 | 400 | 0 (0.0%) | 9117 |
| aqua_rat | 35 | 131 | 128 | 60 | 20 | 25 | 0 | 1 | 400 | 1 (0.2%) | 16384 |
| logiqa_en | 8 | 119 | 134 | 75 | 46 | 17 | 0 | 1 | 400 | 1 (0.2%) | 16384 |
| **ALL** | **59** | **411** | **418** | **320** | **256** | **131** | **0** | **5** | **1600** | **5 (0.3%)** | **16384** |
