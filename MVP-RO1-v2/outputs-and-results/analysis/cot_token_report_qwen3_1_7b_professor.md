# CoT token-length analysis: qwen3_1_7b

**HuggingFace ID:** `Qwen/Qwen3-1.7B`  
**Hint type:** `professor`  
**Datasets included:** codemmlu

---

## 1. Overall CoT token length (pooled across all datasets)

Output tokens in the chain-of-thought, pooled over every valid run (run level) across all datasets above. Same methodology as each dataset's `gmean2_summary.md`. Runs = pooled responses; Instances = distinct examples.

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 2493.6 | 2338.6 | 246 | 11948 | 190 | 50 |
| Unhinted (X=0, all) | 2542.8 | 2673.8 | 248 | 14396 | 194 | 50 |
| Hinted — hint followed (X=1, Y=1) | 3101.8 | 2318.6 | 454 | 8886 | 43 | 19 |
| Hinted — hint not followed (X=1, Y=0) | 2315.7 | 2322.2 | 246 | 11948 | 147 | 45 |

---

## 2. CoT token length per dataset

Same four conditions, computed separately for each dataset.

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
| codemmlu | 50 | 100 | 84 | 65 | 69 | 16 | 0 | 16 | 400 | 16 (4.0%) | 16384 |
| **ALL** | **50** | **100** | **84** | **65** | **69** | **16** | **0** | **16** | **400** | **16 (4.0%)** | **16384** |
