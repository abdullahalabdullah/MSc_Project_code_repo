# CoT token-length aggregate summary — runs pooled across all datasets and hint types

- **Model:** `er-abd/harmless-dpo-qwen3-4B-step-735` (`er-abd_harmless-dpo-qwen3-4B-step-735`)
- **Datasets:** mmlu_pro, gpqa_diamond, wmdp, codemmlu, lsat_ar, lsat_lr, medxpertqa, lexam
- **Hint types:** professor

`token_count` is the full completion (thinking trace + final answer). Condition tables use valid runs only (not truncated, answer parsed), so followed + not-followed partition hinted exactly; bucket tables use every response. Use the truncation rate to set `max_new_tokens`.

## All (8 dataset × hint-type runs)

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 3498.9 | 3121.2 | 262 | 16330 | 12634 | 1594 |
| Unhinted (X=0, all) | 3123.0 | 2957.0 | 261 | 16356 | 12582 | 1594 |
| Hinted, followed (X=1, Y=1) | 4160.0 | 3505.0 | 351 | 16330 | 3447 | 709 |
| Hinted, not followed (X=1, Y=0) | 3250.8 | 2926.3 | 262 | 16209 | 9187 | 1387 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 865 | 4900 | 6126 | 6269 | 4872 | 2197 | 0 | 339 | 25568 | 339 (1.3%) | 16384 |
