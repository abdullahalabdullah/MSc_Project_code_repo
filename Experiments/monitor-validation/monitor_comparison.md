# Monitor Comparison

Subject model: `qwen3_1_7b`

Monitors compared: `gpt-5-mini (mon-2)`, `gemma-4-E4B (local)`, `qwen3_8b (local)`

## Summary (overall agreement)

| Pair | followed_hint | acknowledged | Compared |
| ---- | ------------- | ------------ | -------- |
| gpt-5-mini (mon-2) vs gemma-4-E4B (local) | 100.0% | 93.8% | 2164 |
| gpt-5-mini (mon-2) vs qwen3_8b (local) | 100.0% | 93.7% | 2164 |
| gemma-4-E4B (local) vs qwen3_8b (local) | 100.0% | 89.1% | 2164 |

## `gpt-5-mini (mon-2)` vs `gemma-4-E4B (local)`

### Overlap & removal

| Dataset | gpt-5-mini (mon-2) records | gemma-4-E4B (local) records | Only in A | Only in B | Compared |
| ------- | ---------------- | ---------------- | --------- | --------- | -------- |
| agieval | 538 | 538 | 0 | 0 | 538 |
| gpqa | 549 | 549 | 0 | 0 | 549 |
| mmlu | 551 | 551 | 0 | 0 | 551 |
| wmdp | 526 | 526 | 0 | 0 | 526 |
| **Total** | 2164 | 2164 | 0 | 0 | **2164** |

### Agreement per dataset

| Dataset | followed_hint agreement | acknowledged agreement |
| ------- | ----------------------- | ---------------------- |
| agieval | 100.0% (538/538) | 89.0% (479/538) |
| gpqa | 100.0% (549/549) | 90.2% (495/549) |
| mmlu | 100.0% (551/551) | 99.5% (548/551) |
| wmdp | 100.0% (526/526) | 96.4% (507/526) |
| **Average** | **100.0%** (2164/2164) | **93.8%** (2029/2164) |

## `gpt-5-mini (mon-2)` vs `qwen3_8b (local)`

### Overlap & removal

| Dataset | gpt-5-mini (mon-2) records | qwen3_8b (local) records | Only in A | Only in B | Compared |
| ------- | ---------------- | ---------------- | --------- | --------- | -------- |
| agieval | 538 | 538 | 0 | 0 | 538 |
| gpqa | 549 | 549 | 0 | 0 | 549 |
| mmlu | 551 | 551 | 0 | 0 | 551 |
| wmdp | 526 | 526 | 0 | 0 | 526 |
| **Total** | 2164 | 2164 | 0 | 0 | **2164** |

### Agreement per dataset

| Dataset | followed_hint agreement | acknowledged agreement |
| ------- | ----------------------- | ---------------------- |
| agieval | 100.0% (538/538) | 94.4% (508/538) |
| gpqa | 100.0% (549/549) | 90.9% (499/549) |
| mmlu | 100.0% (551/551) | 96.9% (534/551) |
| wmdp | 100.0% (526/526) | 92.6% (487/526) |
| **Average** | **100.0%** (2164/2164) | **93.7%** (2028/2164) |

## `gemma-4-E4B (local)` vs `qwen3_8b (local)`

### Overlap & removal

| Dataset | gemma-4-E4B (local) records | qwen3_8b (local) records | Only in A | Only in B | Compared |
| ------- | ---------------- | ---------------- | --------- | --------- | -------- |
| agieval | 538 | 538 | 0 | 0 | 538 |
| gpqa | 549 | 549 | 0 | 0 | 549 |
| mmlu | 551 | 551 | 0 | 0 | 551 |
| wmdp | 526 | 526 | 0 | 0 | 526 |
| **Total** | 2164 | 2164 | 0 | 0 | **2164** |

### Agreement per dataset

| Dataset | followed_hint agreement | acknowledged agreement |
| ------- | ----------------------- | ---------------------- |
| agieval | 100.0% (538/538) | 85.7% (461/538) |
| gpqa | 100.0% (549/549) | 83.1% (456/549) |
| mmlu | 100.0% (551/551) | 96.9% (534/551) |
| wmdp | 100.0% (526/526) | 90.7% (477/526) |
| **Average** | **100.0%** (2164/2164) | **89.1%** (1928/2164) |
