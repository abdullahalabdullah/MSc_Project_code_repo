# CoT token-length analysis: tulu_sft_no_safety

**HuggingFace ID:** `allenai/Llama-3.1-Tulu-3-8B-SFT-no-safety-data`  
**Datasets included:** mmlu

---

## 1. Overall CoT token length (pooled across all datasets)

Mean output tokens in the chain-of-thought, averaged per instance then across instances, pooled over every dataset above. Same methodology as each dataset's `gmean2_summary.md`.

| Condition | Mean | Std | Min | Max |
|-----------|------|-----|-----|-----|
| Hinted (X=1, all) | 88.5 | 66.2 | 25 | 420 |
| Unhinted (X=0, all) | 97.0 | 73.5 | 26 | 450 |
| Hinted — hint followed (X=1, Y=1) | 91.1 | 89.7 | 22 | 404 |
| Hinted — hint not followed (X=1, Y=0) | 90.3 | 76.4 | 25 | 468 |

---

## 2. CoT token length per dataset

Same four conditions, computed separately for each dataset.

### mmlu

| Condition | Mean | Std | Min | Max |
|-----------|------|-----|-----|-----|
| Hinted (X=1, all) | 88.5 | 66.2 | 25 | 420 |
| Unhinted (X=0, all) | 97.0 | 73.5 | 26 | 450 |
| Hinted — hint followed (X=1, Y=1) | 91.1 | 89.7 | 22 | 404 |
| Hinted — hint not followed (X=1, Y=0) | 90.3 | 76.4 | 25 | 468 |

---

## 3. Token-length buckets per dataset

Number of individual responses (every sample, both arms) whose `token_count` falls in each length band. **Truncated responses (those that hit `max_new_tokens`) are counted in the top `32768+` bucket regardless of their cap** — their true length is censored above the cap, so they represent the overflow tail, not a normal-length trace at the cap value. The `Truncated` column repeats that count explicitly. A high truncation rate means the cap is too low — use this to pick `max_new_tokens`.

| Dataset | 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mmlu | 558 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 560 | 0 (0.0%) | 583 |
| **ALL** | **558** | **2** | **0** | **0** | **0** | **0** | **0** | **0** | **560** | **0 (0.0%)** | **583** |
