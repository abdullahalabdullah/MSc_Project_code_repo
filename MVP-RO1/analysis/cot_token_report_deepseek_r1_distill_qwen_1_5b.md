# CoT token-length analysis: deepseek_r1_distill_qwen_1_5b

**HuggingFace ID:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`  
**Datasets included:** mmlu

---

## 1. Overall CoT token length (pooled across all datasets)

Mean output tokens in the chain-of-thought, averaged per instance then across instances, pooled over every dataset above. Same methodology as each dataset's `gmean2_summary.md`.

| Condition | Mean | Std | Min | Max |
|-----------|------|-----|-----|-----|
| Hinted (X=1, all) | 1650.3 | 1204.6 | 354 | 5886 |
| Unhinted (X=0, all) | 1741.5 | 1330.1 | 408 | 6190 |
| Hinted — hint followed (X=1, Y=1) | 1779.0 | 1630.2 | 362 | 7924 |
| Hinted — hint not followed (X=1, Y=0) | 1536.8 | 1236.1 | 346 | 5886 |

---

## 2. CoT token length per dataset

Same four conditions, computed separately for each dataset.

### mmlu

| Condition | Mean | Std | Min | Max |
|-----------|------|-----|-----|-----|
| Hinted (X=1, all) | 1650.3 | 1204.6 | 354 | 5886 |
| Unhinted (X=0, all) | 1741.5 | 1330.1 | 408 | 6190 |
| Hinted — hint followed (X=1, Y=1) | 1779.0 | 1630.2 | 362 | 7924 |
| Hinted — hint not followed (X=1, Y=0) | 1536.8 | 1236.1 | 346 | 5886 |

---

## 3. Token-length buckets per dataset

Number of individual responses (every sample, both arms) whose `token_count` falls in each length band. **Truncated responses (those that hit `max_new_tokens`) are counted in the top `32768+` bucket regardless of their cap** — their true length is censored above the cap, so they represent the overflow tail, not a normal-length trace at the cap value. The `Truncated` column repeats that count explicitly. A high truncation rate means the cap is too low — use this to pick `max_new_tokens`.

| Dataset | 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mmlu | 72 | 226 | 117 | 92 | 45 | 5 | 0 | 3 | 560 | 3 (0.5%) | 32768 |
| **ALL** | **72** | **226** | **117** | **92** | **45** | **5** | **0** | **3** | **560** | **3 (0.5%)** | **32768** |
