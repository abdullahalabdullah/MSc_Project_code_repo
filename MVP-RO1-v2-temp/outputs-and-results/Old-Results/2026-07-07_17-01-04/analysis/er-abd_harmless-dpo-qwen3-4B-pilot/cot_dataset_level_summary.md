# CoT token-length dataset-level summary — runs pooled across hint types

- **Model:** `er-abd_harmless-dpo-qwen3-4B-pilot` (`er-abd_harmless-dpo-qwen3-4B-pilot`)
- **Datasets:** lsat_ar, lsat_lr, medxpertqa
- **Hint types:** professor

`token_count` is the full completion (thinking trace + final answer). Condition tables use valid runs only (not truncated, answer parsed), so followed + not-followed partition hinted exactly; bucket tables use every response. Use the truncation rate to set `max_new_tokens`.

## lsat_ar (pooled over 1 hint types)

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 6105.5 | 3339.1 | 1243 | 15823 | 199 | 50 |
| Unhinted (X=0, all) | 5299.9 | 3342.2 | 963 | 15436 | 197 | 50 |
| Hinted, followed (X=1, Y=1) | 9055.2 | 3299.4 | 3344 | 15823 | 35 | 15 |
| Hinted, not followed (X=1, Y=0) | 5476.0 | 3001.0 | 1243 | 14644 | 164 | 46 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 48 | 113 | 149 | 85 | 0 | 4 | 400 | 4 (1.0%) | 16384 |

## lsat_lr (pooled over 1 hint types)

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 1944.8 | 1446.1 | 397 | 5933 | 200 | 50 |
| Unhinted (X=0, all) | 1666.8 | 1384.8 | 412 | 8593 | 200 | 50 |
| Hinted, followed (X=1, Y=1) | 2158.1 | 1071.6 | 834 | 4189 | 24 | 9 |
| Hinted, not followed (X=1, Y=0) | 1915.7 | 1490.0 | 397 | 5933 | 176 | 47 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 12 | 162 | 102 | 84 | 39 | 1 | 0 | 0 | 400 | 0 (0.0%) | 8593 |

## medxpertqa (pooled over 1 hint types)

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 1760.7 | 1119.7 | 444 | 6416 | 200 | 50 |
| Unhinted (X=0, all) | 1733.5 | 1028.1 | 399 | 7171 | 200 | 50 |
| Hinted, followed (X=1, Y=1) | 1654.4 | 1095.0 | 459 | 6177 | 79 | 29 |
| Hinted, not followed (X=1, Y=0) | 1830.2 | 1134.6 | 444 | 6416 | 121 | 40 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 11 | 88 | 194 | 93 | 14 | 0 | 0 | 0 | 400 | 0 (0.0%) | 7171 |

## Summary

| Dataset | Hinted (mean and std) | Unhinted (mean and std) | Hinted & followed (mean and std) | Hinted & not followed (mean and std) |
|---|---|---|---|---|
| lsat_ar | 6105.5 (3339.1) | 5299.9 (3342.2) | 9055.2 (3299.4) | 5476.0 (3001.0) |
| lsat_lr | 1944.8 (1446.1) | 1666.8 (1384.8) | 2158.1 (1071.6) | 1915.7 (1490.0) |
| medxpertqa | 1760.7 (1119.7) | 1733.5 (1028.1) | 1654.4 (1095.0) | 1830.2 (1134.6) |
