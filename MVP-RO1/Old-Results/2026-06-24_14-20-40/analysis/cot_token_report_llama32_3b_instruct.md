# CoT token-length analysis: llama32_3b_instruct

**HuggingFace ID:** `meta-llama/Llama-3.2-3B-Instruct`  
**Datasets included:** mmlu, gpqa, gsm_mc, math_mc, agieval, wmdp

---

## 1. Overall CoT token length (pooled across all datasets)

Mean output tokens in the chain-of-thought, averaged per instance then across instances, pooled over every dataset above. Same methodology as each dataset's `gmean2_summary.md`.

| Condition | Mean | Std | Min | Max |
|-----------|------|-----|-----|-----|
| Hinted (X=1, all) | 374.2 | 208.9 | 102 | 1444 |
| Unhinted (X=0, all) | 391.2 | 243.6 | 105 | 1818 |
| Hinted — hint followed (X=1, Y=1) | 431.2 | 243.0 | 146 | 1516 |
| Hinted — hint not followed (X=1, Y=0) | 366.3 | 202.8 | 102 | 1373 |

---

## 2. CoT token length per dataset

Same four conditions, computed separately for each dataset.

### mmlu

| Condition | Mean | Std | Min | Max |
|-----------|------|-----|-----|-----|
| Hinted (X=1, all) | 287.1 | 108.2 | 122 | 985 |
| Unhinted (X=0, all) | 298.6 | 114.2 | 126 | 1038 |
| Hinted — hint followed (X=1, Y=1) | 270.1 | 60.3 | 148 | 382 |
| Hinted — hint not followed (X=1, Y=0) | 285.5 | 112.7 | 122 | 985 |

### gpqa

| Condition | Mean | Std | Min | Max |
|-----------|------|-----|-----|-----|
| Hinted (X=1, all) | 510.7 | 211.4 | 240 | 1171 |
| Unhinted (X=0, all) | 549.0 | 254.1 | 250 | 1410 |
| Hinted — hint followed (X=1, Y=1) | 519.3 | 267.2 | 198 | 1492 |
| Hinted — hint not followed (X=1, Y=0) | 514.8 | 211.9 | 204 | 1171 |

### gsm_mc

| Condition | Mean | Std | Min | Max |
|-----------|------|-----|-----|-----|
| Hinted (X=1, all) | 219.5 | 81.0 | 102 | 604 |
| Unhinted (X=0, all) | 228.0 | 96.7 | 105 | 633 |
| Hinted — hint followed (X=1, Y=1) | 272.8 | 109.9 | 155 | 566 |
| Hinted — hint not followed (X=1, Y=0) | 219.0 | 84.0 | 102 | 604 |

### math_mc

| Condition | Mean | Std | Min | Max |
|-----------|------|-----|-----|-----|
| Hinted (X=1, all) | 460.9 | 280.3 | 129 | 1411 |
| Unhinted (X=0, all) | 483.6 | 359.8 | 138 | 1818 |
| Hinted — hint followed (X=1, Y=1) | 649.1 | 310.6 | 146 | 1411 |
| Hinted — hint not followed (X=1, Y=0) | 424.5 | 251.2 | 129 | 1041 |

### agieval

| Condition | Mean | Std | Min | Max |
|-----------|------|-----|-----|-----|
| Hinted (X=1, all) | 401.4 | 140.7 | 123 | 864 |
| Unhinted (X=0, all) | 400.9 | 149.1 | 109 | 1024 |
| Hinted — hint followed (X=1, Y=1) | 424.6 | 147.7 | 208 | 864 |
| Hinted — hint not followed (X=1, Y=0) | 401.1 | 143.5 | 123 | 828 |

### wmdp

| Condition | Mean | Std | Min | Max |
|-----------|------|-----|-----|-----|
| Hinted (X=1, all) | 372.4 | 208.9 | 128 | 1444 |
| Unhinted (X=0, all) | 394.4 | 225.7 | 141 | 1256 |
| Hinted — hint followed (X=1, Y=1) | 408.4 | 256.1 | 180 | 1516 |
| Hinted — hint not followed (X=1, Y=0) | 372.9 | 215.7 | 128 | 1373 |

---

## 3. Token-length buckets per dataset

Number of individual responses (every sample, both arms) whose `token_count` falls in each length band. **Truncated responses (those that hit `max_new_tokens`) are counted in the top `32768+` bucket regardless of their cap** — their true length is censored above the cap, so they represent the overflow tail, not a normal-length trace at the cap value. The `Truncated` column repeats that count explicitly. A high truncation rate means the cap is too low — use this to pick `max_new_tokens`.

| Dataset | 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mmlu | 540 | 14 | 3 | 0 | 0 | 0 | 0 | 3 | 560 | 3 (0.5%) | 2048 |
| gpqa | 344 | 144 | 33 | 0 | 0 | 0 | 0 | 39 | 560 | 39 (7.0%) | 2048 |
| gsm_mc | 540 | 15 | 2 | 0 | 0 | 0 | 0 | 3 | 560 | 3 (0.5%) | 2048 |
| math_mc | 357 | 91 | 36 | 0 | 0 | 0 | 0 | 76 | 560 | 76 (13.6%) | 2048 |
| agieval | 472 | 71 | 10 | 0 | 0 | 0 | 0 | 7 | 560 | 7 (1.2%) | 2048 |
| wmdp | 486 | 46 | 20 | 0 | 0 | 0 | 0 | 8 | 560 | 8 (1.4%) | 2048 |
| **ALL** | **2739** | **381** | **104** | **0** | **0** | **0** | **0** | **136** | **3360** | **136 (4.0%)** | **2048** |
