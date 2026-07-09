# CoT token-length overall summary — per (dataset × hint type)

- **Model:** `Qwen/Qwen3-4B` (`Qwen_Qwen3-4B`)
- **Datasets:** mmlu_pro, medxpertqa, lexam
- **Hint types:** professor

`token_count` is the full completion (thinking trace + final answer). Condition tables use valid runs only (not truncated, answer parsed), so followed + not-followed partition hinted exactly; bucket tables use every response. Use the truncation rate to set `max_new_tokens`.

## mmlu_pro × professor

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 2663.2 | 2887.3 | 390 | 15402 | 198 | 50 |
| Unhinted (X=0, all) | 2266.1 | 2981.6 | 361 | 16131 | 200 | 50 |
| Hinted, followed (X=1, Y=1) | 3713.7 | 3943.7 | 511 | 15402 | 35 | 16 |
| Hinted, not followed (X=1, Y=0) | 2437.6 | 2565.9 | 390 | 12032 | 163 | 45 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 35 | 142 | 93 | 64 | 32 | 32 | 0 | 2 | 400 | 2 (0.5%) | 16384 |

## medxpertqa × professor

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 1685.8 | 968.4 | 438 | 4760 | 200 | 50 |
| Unhinted (X=0, all) | 1632.8 | 934.0 | 390 | 5658 | 200 | 50 |
| Hinted, followed (X=1, Y=1) | 1598.9 | 984.2 | 514 | 4530 | 89 | 30 |
| Hinted, not followed (X=1, Y=0) | 1755.4 | 954.4 | 438 | 4760 | 111 | 36 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 107 | 178 | 92 | 13 | 0 | 0 | 0 | 400 | 0 (0.0%) | 5658 |

## lexam × professor

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 1803.9 | 991.2 | 475 | 5134 | 200 | 50 |
| Unhinted (X=0, all) | 1695.0 | 1171.9 | 361 | 5767 | 200 | 50 |
| Hinted, followed (X=1, Y=1) | 1879.9 | 984.1 | 492 | 4253 | 72 | 23 |
| Hinted, not followed (X=1, Y=0) | 1761.1 | 996.5 | 475 | 5134 | 128 | 36 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 15 | 112 | 142 | 119 | 12 | 0 | 0 | 0 | 400 | 0 (0.0%) | 5767 |

## Summary

| Dataset | Hint type | Hinted (mean and std) | Unhinted (mean and std) | Hinted & followed (mean and std) | Hinted & not followed (mean and std) |
|---|---|---|---|---|---|
| mmlu_pro | professor | 2663.2 (2887.3) | 2266.1 (2981.6) | 3713.7 (3943.7) | 2437.6 (2565.9) |
| medxpertqa | professor | 1685.8 (968.4) | 1632.8 (934.0) | 1598.9 (984.2) | 1755.4 (954.4) |
| lexam | professor | 1803.9 (991.2) | 1695.0 (1171.9) | 1879.9 (984.1) | 1761.1 (996.5) |
