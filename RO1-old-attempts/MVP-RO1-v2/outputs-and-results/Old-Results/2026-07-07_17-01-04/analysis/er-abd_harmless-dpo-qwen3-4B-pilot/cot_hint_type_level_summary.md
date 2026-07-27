# CoT token-length hint-type-level summary — runs pooled across datasets

- **Model:** `er-abd_harmless-dpo-qwen3-4B-pilot` (`er-abd_harmless-dpo-qwen3-4B-pilot`)
- **Datasets:** lsat_ar, lsat_lr, medxpertqa
- **Hint types:** professor

`token_count` is the full completion (thinking trace + final answer). Condition tables use valid runs only (not truncated, answer parsed), so followed + not-followed partition hinted exactly; bucket tables use every response. Use the truncation rate to set `max_new_tokens`.

## professor (pooled over 3 datasets)

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 3265.6 | 2971.5 | 397 | 15823 | 599 | 150 |
| Unhinted (X=0, all) | 2888.0 | 2745.4 | 399 | 15436 | 597 | 150 |
| Hinted, followed (X=1, Y=1) | 3619.0 | 3704.9 | 459 | 15823 | 138 | 53 |
| Hinted, not followed (X=1, Y=0) | 3159.8 | 2709.5 | 397 | 14644 | 461 | 133 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 23 | 251 | 344 | 290 | 202 | 86 | 0 | 4 | 1200 | 4 (0.3%) | 16384 |

## Summary

| Hint type | Hinted (mean and std) | Unhinted (mean and std) | Hinted & followed (mean and std) | Hinted & not followed (mean and std) |
|---|---|---|---|---|
| professor | 3265.6 (2971.5) | 2888.0 (2745.4) | 3619.0 (3704.9) | 3159.8 (2709.5) |
