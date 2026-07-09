# CoT token-length hint-type-level summary — runs pooled across datasets

- **Model:** `Qwen/Qwen3-4B` (`Qwen_Qwen3-4B`)
- **Datasets:** mmlu_pro, medxpertqa, lexam
- **Hint types:** professor

`token_count` is the full completion (thinking trace + final answer). Condition tables use valid runs only (not truncated, answer parsed), so followed + not-followed partition hinted exactly; bucket tables use every response. Use the truncation rate to set `max_new_tokens`.

## professor (pooled over 3 datasets)

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 2048.9 | 1892.2 | 390 | 15402 | 598 | 150 |
| Unhinted (X=0, all) | 1864.6 | 1944.4 | 361 | 16131 | 600 | 150 |
| Hinted, followed (X=1, Y=1) | 2079.8 | 2025.1 | 492 | 15402 | 196 | 69 |
| Hinted, not followed (X=1, Y=0) | 2033.8 | 1826.4 | 390 | 12032 | 402 | 117 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 60 | 361 | 413 | 275 | 57 | 32 | 0 | 2 | 1200 | 2 (0.2%) | 16384 |

## Summary

| Hint type | Hinted (mean and std) | Unhinted (mean and std) | Hinted & followed (mean and std) | Hinted & not followed (mean and std) |
|---|---|---|---|---|
| professor | 2048.9 (1892.2) | 1864.6 (1944.4) | 2079.8 (2025.1) | 2033.8 (1826.4) |
