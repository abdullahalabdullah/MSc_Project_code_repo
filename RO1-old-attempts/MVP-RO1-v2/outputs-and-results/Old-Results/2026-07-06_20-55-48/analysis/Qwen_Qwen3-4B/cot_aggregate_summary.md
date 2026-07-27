# CoT token-length aggregate summary — runs pooled across all datasets and hint types

- **Model:** `Qwen/Qwen3-4B` (`Qwen_Qwen3-4B`)
- **Datasets:** mmlu_pro, lexam
- **Hint types:** professor

`token_count` is the full completion (thinking trace + final answer). Condition tables use valid runs only (not truncated, answer parsed), so followed + not-followed partition hinted exactly; bucket tables use every response. Use the truncation rate to set `max_new_tokens`.

## All (2 dataset × hint-type runs)

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 2231.4 | 2194.2 | 390 | 15402 | 398 | 100 |
| Unhinted (X=0, all) | 1980.5 | 2280.5 | 361 | 16131 | 400 | 100 |
| Hinted, followed (X=1, Y=1) | 2479.7 | 2526.8 | 492 | 15402 | 107 | 39 |
| Hinted, not followed (X=1, Y=0) | 2140.1 | 2055.7 | 390 | 12032 | 291 | 81 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 254 | 235 | 183 | 44 | 32 | 0 | 2 | 800 | 2 (0.2%) | 16384 |
