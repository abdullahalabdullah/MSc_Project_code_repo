# CoT token-length analysis: aqua_dpo_qwen3_1_7b_s0

**HuggingFace ID:** `er-abd/aqua-dpo-qwen3-1.7B-sk-llama_original-seed-0-merged`  
**Datasets included:** mmlu, gpqa, wmdp, agieval

---

## 1. Overall CoT token length (pooled across all datasets)

Mean output tokens in the chain-of-thought, averaged per instance then across instances, pooled over every dataset above. Same methodology as each dataset's `gmean2_summary.md`.

| Condition | Mean | Std | Min | Max |
|-----------|------|-----|-----|-----|
| Hinted (X=1, all) | 3483.2 | 3284.6 | 473 | 16264 |
| Unhinted (X=0, all) | 3540.1 | 3485.3 | 398 | 16312 |
| Hinted — hint followed (X=1, Y=1) | 4442.7 | 3467.9 | 476 | 13434 |
| Hinted — hint not followed (X=1, Y=0) | 3420.4 | 3309.3 | 473 | 16264 |

---

## 2. CoT token length per dataset

Same four conditions, computed separately for each dataset.

### mmlu

| Condition | Mean | Std | Min | Max |
|-----------|------|-----|-----|-----|
| Hinted (X=1, all) | 1351.9 | 807.4 | 476 | 4032 |
| Unhinted (X=0, all) | 1193.9 | 758.9 | 428 | 4642 |
| Hinted — hint followed (X=1, Y=1) | 1811.1 | 1791.1 | 476 | 8306 |
| Hinted — hint not followed (X=1, Y=0) | 1374.3 | 781.3 | 504 | 4032 |

### gpqa

| Condition | Mean | Std | Min | Max |
|-----------|------|-----|-----|-----|
| Hinted (X=1, all) | 5706.9 | 3016.2 | 689 | 13994 |
| Unhinted (X=0, all) | 5943.9 | 3174.7 | 708 | 13636 |
| Hinted — hint followed (X=1, Y=1) | 6702.9 | 3150.9 | 864 | 13434 |
| Hinted — hint not followed (X=1, Y=0) | 5630.1 | 3118.6 | 689 | 14394 |

### wmdp

| Condition | Mean | Std | Min | Max |
|-----------|------|-----|-----|-----|
| Hinted (X=1, all) | 3238.0 | 3889.9 | 473 | 16264 |
| Unhinted (X=0, all) | 3108.6 | 3813.6 | 398 | 15112 |
| Hinted — hint followed (X=1, Y=1) | 3639.5 | 3530.7 | 480 | 13421 |
| Hinted — hint not followed (X=1, Y=0) | 2887.4 | 3835.3 | 473 | 16264 |

### agieval

| Condition | Mean | Std | Min | Max |
|-----------|------|-----|-----|-----|
| Hinted (X=1, all) | 3636.1 | 2950.3 | 560 | 13494 |
| Unhinted (X=0, all) | 3914.2 | 3432.4 | 417 | 16312 |
| Hinted — hint followed (X=1, Y=1) | 4470.9 | 2754.0 | 916 | 11713 |
| Hinted — hint not followed (X=1, Y=0) | 3846.8 | 3140.0 | 560 | 13494 |

---

## 3. Token-length buckets per dataset

Number of individual responses (every sample, both arms) whose `token_count` falls in each length band. **Truncated responses (those that hit `max_new_tokens`) are counted in the top `32768+` bucket regardless of their cap** — their true length is censored above the cap, so they represent the overflow tail, not a normal-length trace at the cap value. The `Truncated` column repeats that count explicitly. A high truncation rate means the cap is too low — use this to pick `max_new_tokens`.

| Dataset | 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mmlu | 47 | 270 | 164 | 69 | 8 | 2 | 0 | 0 | 560 | 0 (0.0%) | 9165 |
| gpqa | 0 | 21 | 70 | 112 | 209 | 140 | 0 | 8 | 560 | 8 (1.4%) | 16384 |
| wmdp | 21 | 126 | 206 | 105 | 29 | 42 | 0 | 31 | 560 | 31 (5.5%) | 16384 |
| agieval | 10 | 92 | 149 | 128 | 110 | 66 | 0 | 5 | 560 | 5 (0.9%) | 16384 |
| **ALL** | **78** | **509** | **589** | **414** | **356** | **250** | **0** | **44** | **2240** | **44 (2.0%)** | **16384** |
