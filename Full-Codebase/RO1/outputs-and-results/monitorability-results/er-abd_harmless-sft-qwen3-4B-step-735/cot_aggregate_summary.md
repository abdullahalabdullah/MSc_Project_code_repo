# CoT token-length aggregate summary — runs pooled across all datasets and hint types

- **Model:** `er-abd/harmless-sft-qwen3-4B-step-735` (`er-abd_harmless-sft-qwen3-4B-step-735`)
- **Datasets:** mmlu_pro
- **Hint types:** grader_hacking

`token_count` is the full completion (thinking trace + final answer). Condition tables use valid runs only (not truncated, answer parsed), so followed + not-followed partition hinted exactly; bucket tables use every response. Use the truncation rate to set `max_new_tokens`.

## All (1 dataset × hint-type runs)

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 2653.2 | 2701.0 | 1528 | 282 | 14750 | 1585 | 200 |
| Unhinted (X=0, all) | 1981.3 | 2349.9 | 1019 | 262 | 15183 | 1596 | 200 |
| Hinted, followed (X=1, Y=1) | 3580.9 | 3174.2 | 2381 | 343 | 14750 | 385 | 93 |
| Hinted, not followed (X=1, Y=0) | 2355.6 | 2458.8 | 1374 | 282 | 14285 | 1200 | 184 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 259 | 1120 | 751 | 506 | 384 | 161 | 0 | 19 | 3200 | 19 (0.6%) | 16384 |
