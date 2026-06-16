# CoT token-length analysis: qwen3_4b

**HuggingFace ID:** `Qwen/Qwen3-4B`  
**Datasets included:** mmlu

---

## 1. Overall CoT token length (pooled across all datasets)

Mean output tokens in the chain-of-thought, averaged per instance then across instances, pooled over every dataset above. Same methodology as each dataset's `gmean2_summary.md`.

| Condition | Mean | Std | Min | Max |
|-----------|------|-----|-----|-----|
| Hinted (X=1, all) | 980.9 | 669.9 | 307 | 3278 |
| Unhinted (X=0, all) | 779.5 | 468.5 | 288 | 2547 |
| Hinted — hint followed (X=1, Y=1) | 1054.2 | 499.5 | 394 | 2274 |
| Hinted — hint not followed (X=1, Y=0) | 1010.9 | 706.3 | 307 | 3278 |

---

## 2. CoT token length per dataset

Same four conditions, computed separately for each dataset.

### mmlu

| Condition | Mean | Std | Min | Max |
|-----------|------|-----|-----|-----|
| Hinted (X=1, all) | 980.9 | 669.9 | 307 | 3278 |
| Unhinted (X=0, all) | 779.5 | 468.5 | 288 | 2547 |
| Hinted — hint followed (X=1, Y=1) | 1054.2 | 499.5 | 394 | 2274 |
| Hinted — hint not followed (X=1, Y=0) | 1010.9 | 706.3 | 307 | 3278 |

---

## 3. Token-length buckets per dataset

Number of individual responses (every sample, both arms) whose `token_count` falls in each length band. **Truncated responses (those that hit `max_new_tokens`) are counted in the top `32768+` bucket regardless of their cap** — their true length is censored above the cap, so they represent the overflow tail, not a normal-length trace at the cap value. The `Truncated` column repeats that count explicitly. A high truncation rate means the cap is too low — use this to pick `max_new_tokens`.

| Dataset | 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mmlu | 174 | 243 | 105 | 37 | 1 | 0 | 0 | 0 | 560 | 0 (0.0%) | 4269 |
| **ALL** | **174** | **243** | **105** | **37** | **1** | **0** | **0** | **0** | **560** | **0 (0.0%)** | **4269** |
