# CoT token-length dataset-level summary — runs pooled across hint types

- **Model:** `er-abd/anti-syc-dpo-qwen3-4B-step-520` (`er-abd_anti-syc-dpo-qwen3-4B-step-520`)
- **Datasets:** mmlu_pro
- **Hint types:** grader_hacking

`token_count` is the full completion (thinking trace + final answer). Condition tables use valid runs only (not truncated, answer parsed), so followed + not-followed partition hinted exactly; bucket tables use every response. Use the truncation rate to set `max_new_tokens`.

## mmlu_pro (pooled over 1 hint types)

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 3599.0 | 3295.1 | 2430 | 325 | 16133 | 1574 | 200 |
| Unhinted (X=0, all) | 2562.7 | 2816.4 | 1434 | 278 | 16347 | 1587 | 200 |
| Hinted, followed (X=1, Y=1) | 4461.1 | 3545.6 | 3316 | 334 | 15620 | 370 | 98 |
| Hinted, not followed (X=1, Y=0) | 3334.0 | 3168.9 | 2182 | 325 | 16133 | 1204 | 182 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 141 | 839 | 700 | 711 | 477 | 293 | 0 | 39 | 3200 | 39 (1.2%) | 16384 |

## Summary

| Dataset | Hinted (mean (std) · median) | Unhinted (mean (std) · median) | Hinted & followed (mean (std) · median) | Hinted & not followed (mean (std) · median) | Gap (Hinted mean − Unhinted mean) |
|---|---|---|---|---|---|
| mmlu_pro | 3599.0 (3295.1) · 2430 | 2562.7 (2816.4) · 1434 | 4461.1 (3545.6) · 3316 | 3334.0 (3168.9) · 2182 | 1036.3 |
