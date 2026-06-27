# CoT token-length analysis: qwen3_1_7b

**HuggingFace ID:** `Qwen/Qwen3-1.7B`  
**Datasets included:** mmlu, agieval, gpqa, wmdp

---

## 1. Overall CoT token length (pooled across all datasets)

Mean output tokens in the chain-of-thought, averaged per instance then across instances, pooled over every dataset above. Same methodology as each dataset's `gmean2_summary.md`.

| Condition | Mean | Std | Min | Max |
|-----------|------|-----|-----|-----|
| Hinted (X=1, all) | 3220.7 | 2985.1 | 431 | 13510 |
| Unhinted (X=0, all) | 3153.1 | 3118.2 | 352 | 14229 |
| Hinted — hint followed (X=1, Y=1) | 4395.6 | 3438.4 | 311 | 13594 |
| Hinted — hint not followed (X=1, Y=0) | 3182.1 | 3012.8 | 431 | 13873 |

---

## 2. CoT token length per dataset

Same four conditions, computed separately for each dataset.

### mmlu

| Condition | Mean | Std | Min | Max |
|-----------|------|-----|-----|-----|
| Hinted (X=1, all) | 1289.9 | 808.2 | 431 | 4732 |
| Unhinted (X=0, all) | 1012.5 | 579.5 | 352 | 2902 |
| Hinted — hint followed (X=1, Y=1) | 1444.7 | 1042.3 | 434 | 4172 |
| Hinted — hint not followed (X=1, Y=0) | 1293.4 | 796.5 | 431 | 4732 |

### agieval

| Condition | Mean | Std | Min | Max |
|-----------|------|-----|-----|-----|
| Hinted (X=1, all) | 3697.7 | 3053.8 | 522 | 12352 |
| Unhinted (X=0, all) | 3591.6 | 3074.0 | 388 | 13819 |
| Hinted — hint followed (X=1, Y=1) | 5467.4 | 3729.5 | 708 | 13594 |
| Hinted — hint not followed (X=1, Y=0) | 3743.5 | 3091.8 | 522 | 12573 |

### gpqa

| Condition | Mean | Std | Min | Max |
|-----------|------|-----|-----|-----|
| Hinted (X=1, all) | 5257.4 | 2828.6 | 578 | 13510 |
| Unhinted (X=0, all) | 5433.2 | 2989.3 | 606 | 12920 |
| Hinted — hint followed (X=1, Y=1) | 6403.6 | 2844.7 | 708 | 13182 |
| Hinted — hint not followed (X=1, Y=0) | 5162.3 | 3010.9 | 578 | 13873 |

### wmdp

| Condition | Mean | Std | Min | Max |
|-----------|------|-----|-----|-----|
| Hinted (X=1, all) | 2627.6 | 3039.4 | 457 | 12586 |
| Unhinted (X=0, all) | 2564.4 | 3140.9 | 408 | 14229 |
| Hinted — hint followed (X=1, Y=1) | 2694.8 | 2660.0 | 311 | 11343 |
| Hinted — hint not followed (X=1, Y=0) | 2617.3 | 3051.2 | 457 | 12586 |

---

## 3. Token-length buckets per dataset

Number of individual responses (every sample, both arms) whose `token_count` falls in each length band. **Truncated responses (those that hit `max_new_tokens`) are counted in the top `32768+` bucket regardless of their cap** — their true length is censored above the cap, so they represent the overflow tail, not a normal-length trace at the cap value. The `Truncated` column repeats that count explicitly. A high truncation rate means the cap is too low — use this to pick `max_new_tokens`.

| Dataset | 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mmlu | 76 | 272 | 136 | 60 | 7 | 0 | 0 | 9 | 560 | 9 (1.6%) | 16384 |
| agieval | 15 | 116 | 102 | 133 | 117 | 60 | 0 | 17 | 560 | 17 (3.0%) | 16384 |
| gpqa | 2 | 27 | 72 | 134 | 195 | 119 | 0 | 11 | 560 | 11 (2.0%) | 16384 |
| wmdp | 32 | 156 | 181 | 93 | 30 | 34 | 0 | 34 | 560 | 34 (6.1%) | 16384 |
| **ALL** | **125** | **571** | **491** | **420** | **349** | **213** | **0** | **71** | **2240** | **71 (3.2%)** | **16384** |
