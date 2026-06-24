# CoT token-length analysis: mistral_7b_instruct

**HuggingFace ID:** `mistralai/Mistral-7B-Instruct-v0.1`  
**Datasets included:** mmlu, gpqa

---

## 1. Overall CoT token length (pooled across all datasets)

Mean output tokens in the chain-of-thought, averaged per instance then across instances, pooled over every dataset above. Same methodology as each dataset's `gmean2_summary.md`.

| Condition | Mean | Std | Min | Max |
|-----------|------|-----|-----|-----|
| Hinted (X=1, all) | 274.9 | 186.5 | 40 | 1024 |
| Unhinted (X=0, all) | 272.9 | 191.1 | 28 | 1017 |
| Hinted — hint followed (X=1, Y=1) | 272.1 | 222.2 | 40 | 1459 |
| Hinted — hint not followed (X=1, Y=0) | 280.7 | 192.6 | 53 | 1024 |

---

## 2. CoT token length per dataset

Same four conditions, computed separately for each dataset.

### mmlu

| Condition | Mean | Std | Min | Max |
|-----------|------|-----|-----|-----|
| Hinted (X=1, all) | 168.5 | 74.9 | 40 | 473 |
| Unhinted (X=0, all) | 175.9 | 104.1 | 28 | 564 |
| Hinted — hint followed (X=1, Y=1) | 165.1 | 83.2 | 40 | 498 |
| Hinted — hint not followed (X=1, Y=0) | 175.7 | 86.0 | 53 | 548 |

### gpqa

| Condition | Mean | Std | Min | Max |
|-----------|------|-----|-----|-----|
| Hinted (X=1, all) | 381.2 | 203.4 | 143 | 1024 |
| Unhinted (X=0, all) | 369.9 | 208.2 | 93 | 1017 |
| Hinted — hint followed (X=1, Y=1) | 372.4 | 261.5 | 66 | 1459 |
| Hinted — hint not followed (X=1, Y=0) | 388.9 | 211.3 | 92 | 1024 |

---

## 3. Token-length buckets per dataset

Number of individual responses (every sample, both arms) whose `token_count` falls in each length band. **Truncated responses (those that hit `max_new_tokens`) are counted in the top `32768+` bucket regardless of their cap** — their true length is censored above the cap, so they represent the overflow tail, not a normal-length trace at the cap value. The `Truncated` column repeats that count explicitly. A high truncation rate means the cap is too low — use this to pick `max_new_tokens`.

| Dataset | 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mmlu | 551 | 6 | 2 | 0 | 0 | 0 | 0 | 1 | 560 | 1 (0.2%) | 2048 |
| gpqa | 435 | 91 | 18 | 0 | 0 | 0 | 0 | 16 | 560 | 16 (2.9%) | 2048 |
| **ALL** | **986** | **97** | **20** | **0** | **0** | **0** | **0** | **17** | **1120** | **17 (1.5%)** | **2048** |
