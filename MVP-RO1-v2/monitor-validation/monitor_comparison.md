# Monitor Comparison: monitor_outputs-1 vs monitor_outputs-2

Model: `qwen3_1_7b`

## Overlap & removal

| Dataset | Mon-1 records | Mon-2 records | Removed from Mon-1 | Removed from Mon-2 | Compared |
| ------- | ------------- | ------------- | ------------------ | ------------------ | -------- |
| agieval | 538 | 538 | 0 | 0 | 538 |
| gpqa | 549 | 549 | 0 | 0 | 549 |
| mmlu | 551 | 551 | 0 | 0 | 551 |
| wmdp | 526 | 526 | 0 | 0 | 526 |
| **Total** | 2164 | 2164 | 0 | 0 | **2164** |

## Agreement per dataset

| Dataset | followed_hint agreement | acknowledged agreement |
| ------- | ----------------------- | ---------------------- |
| agieval | 100.0% (538/538) | 100.0% (538/538) |
| gpqa | 100.0% (549/549) | 100.0% (549/549) |
| mmlu | 100.0% (551/551) | 100.0% (551/551) |
| wmdp | 100.0% (526/526) | 100.0% (526/526) |
| **Average** | **100.0%** (2164/2164) | **100.0%** (2164/2164) |
