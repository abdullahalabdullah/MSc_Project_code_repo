# CoT token-length analysis: qwen3_4b_nothink

**HuggingFace ID:** `Qwen/Qwen3-4B`  
**Datasets included:** mmlu, gpqa, agieval, wmdp

---

## 1. Overall CoT token length (pooled across all datasets)

Mean output tokens in the chain-of-thought, averaged per instance then across instances, pooled over every dataset above. Same methodology as each dataset's `gmean2_summary.md`.

| Condition | Mean | Std | Min | Max |
|-----------|------|-----|-----|-----|
| Hinted (X=1, all) | 554.0 | 454.5 | 94 | 3528 |
| Unhinted (X=0, all) | 492.2 | 355.1 | 85 | 3124 |
| Hinted — hint followed (X=1, Y=1) | 563.3 | 493.8 | 131 | 4206 |
| Hinted — hint not followed (X=1, Y=0) | 577.7 | 480.7 | 94 | 3528 |

---

## 2. CoT token length per dataset

Same four conditions, computed separately for each dataset.

### mmlu

| Condition | Mean | Std | Min | Max |
|-----------|------|-----|-----|-----|
| Hinted (X=1, all) | 944.2 | 640.9 | 338 | 3528 |
| Unhinted (X=0, all) | 769.6 | 478.8 | 320 | 3124 |
| Hinted — hint followed (X=1, Y=1) | 1273.8 | 1000.0 | 399 | 4206 |
| Hinted — hint not followed (X=1, Y=0) | 952.1 | 634.1 | 338 | 3528 |

### gpqa

| Condition | Mean | Std | Min | Max |
|-----------|------|-----|-----|-----|
| Hinted (X=1, all) | 514.8 | 253.4 | 196 | 1336 |
| Unhinted (X=0, all) | 481.7 | 212.6 | 194 | 1270 |
| Hinted — hint followed (X=1, Y=1) | 540.3 | 258.9 | 227 | 1296 |
| Hinted — hint not followed (X=1, Y=0) | 488.7 | 246.3 | 196 | 1375 |

### agieval

| Condition | Mean | Std | Min | Max |
|-----------|------|-----|-----|-----|
| Hinted (X=1, all) | 456.1 | 284.0 | 94 | 1181 |
| Unhinted (X=0, all) | 446.7 | 285.1 | 85 | 1108 |
| Hinted — hint followed (X=1, Y=1) | 580.5 | 311.4 | 135 | 1414 |
| Hinted — hint not followed (X=1, Y=0) | 455.6 | 308.5 | 94 | 1538 |

### wmdp

| Condition | Mean | Std | Min | Max |
|-----------|------|-----|-----|-----|
| Hinted (X=1, all) | 301.0 | 208.5 | 131 | 1713 |
| Unhinted (X=0, all) | 270.8 | 142.7 | 87 | 1055 |
| Hinted — hint followed (X=1, Y=1) | 297.6 | 140.4 | 131 | 777 |
| Hinted — hint not followed (X=1, Y=0) | 316.3 | 254.6 | 138 | 1713 |

---

## 3. Token-length buckets per dataset

Number of individual responses (every sample, both arms) whose `token_count` falls in each length band. **Truncated responses (those that hit `max_new_tokens`) are counted in the top `32768+` bucket regardless of their cap** — their true length is censored above the cap, so they represent the overflow tail, not a normal-length trace at the cap value. The `Truncated` column repeats that count explicitly. A high truncation rate means the cap is too low — use this to pick `max_new_tokens`.

| Dataset | 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mmlu | 176 | 256 | 95 | 32 | 1 | 0 | 0 | 0 | 560 | 0 (0.0%) | 4206 |
| gpqa | 363 | 160 | 29 | 0 | 0 | 0 | 0 | 8 | 560 | 8 (1.4%) | 2048 |
| agieval | 370 | 158 | 31 | 0 | 0 | 0 | 0 | 1 | 560 | 1 (0.2%) | 2048 |
| wmdp | 531 | 17 | 6 | 0 | 0 | 0 | 0 | 6 | 560 | 6 (1.1%) | 2048 |
| **ALL** | **1440** | **591** | **161** | **32** | **1** | **0** | **0** | **15** | **2240** | **15 (0.7%)** | **4206** |
