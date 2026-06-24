# CoT token-length analysis: llama32_3b_sft_harmless_e4

**HuggingFace ID:** `er-abd/llama-3.2-3b-sft-harmlessness-hh-merged-epoch4`  
**Datasets included:** mmlu

---

## 1. Overall CoT token length (pooled across all datasets)

Mean output tokens in the chain-of-thought, averaged per instance then across instances, pooled over every dataset above. Same methodology as each dataset's `gmean2_summary.md`.

| Condition | Mean | Std | Min | Max |
|-----------|------|-----|-----|-----|
| Hinted (X=1, all) | 205.3 | 73.6 | 59 | 408 |
| Unhinted (X=0, all) | 208.7 | 128.8 | 73 | 934 |
| Hinted — hint followed (X=1, Y=1) | 244.2 | 171.8 | 93 | 980 |
| Hinted — hint not followed (X=1, Y=0) | 203.8 | 110.4 | 30 | 743 |

---

## 2. CoT token length per dataset

Same four conditions, computed separately for each dataset.

### mmlu

| Condition | Mean | Std | Min | Max |
|-----------|------|-----|-----|-----|
| Hinted (X=1, all) | 205.3 | 73.6 | 59 | 408 |
| Unhinted (X=0, all) | 208.7 | 128.8 | 73 | 934 |
| Hinted — hint followed (X=1, Y=1) | 244.2 | 171.8 | 93 | 980 |
| Hinted — hint not followed (X=1, Y=0) | 203.8 | 110.4 | 30 | 743 |

---

## 3. Token-length buckets per dataset

Number of individual responses (every sample, both arms) whose `token_count` falls in each length band. **Truncated responses (those that hit `max_new_tokens`) are counted in the top `32768+` bucket regardless of their cap** — their true length is censored above the cap, so they represent the overflow tail, not a normal-length trace at the cap value. The `Truncated` column repeats that count explicitly. A high truncation rate means the cap is too low — use this to pick `max_new_tokens`.

| Dataset | 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mmlu | 536 | 17 | 2 | 0 | 0 | 0 | 0 | 5 | 560 | 5 (0.9%) | 2048 |
| **ALL** | **536** | **17** | **2** | **0** | **0** | **0** | **0** | **5** | **560** | **5 (0.9%)** | **2048** |
