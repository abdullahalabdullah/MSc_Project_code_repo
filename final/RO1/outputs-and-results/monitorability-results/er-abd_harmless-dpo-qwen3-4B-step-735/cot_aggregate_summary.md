# CoT token-length aggregate summary — runs pooled across all datasets and hint types

- **Model:** `er-abd_harmless-dpo-qwen3-4B-step-735` (`er-abd_harmless-dpo-qwen3-4B-step-735`)
- **Datasets:** mmlu_pro, gpqa_diamond, wmdp, codemmlu, lsat_ar, lsat_lr, medxpertqa, lexam
- **Hint types:** professor, grader_hacking, unethical

`token_count` is the full completion (thinking trace + final answer). Condition tables use valid runs only (not truncated, answer parsed), so followed + not-followed partition hinted exactly; bucket tables use every response. Use the truncation rate to set `max_new_tokens`.

## All (23 dataset × hint-type runs)

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 3700.2 | 3233.1 | 2588 | 227 | 16377 | 36211 | 4583 |
| Unhinted (X=0, all) | 3154.9 | 2964.8 | 2058 | 261 | 16356 | 36154 | 4583 |
| Hinted, followed (X=1, Y=1) | 4108.4 | 3383.5 | 2942 | 276 | 16377 | 14440 | 2676 |
| Hinted, not followed (X=1, Y=0) | 3429.4 | 3099.9 | 2372 | 227 | 16273 | 21771 | 3507 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 2508 | 13310 | 17132 | 18108 | 14467 | 6880 | 0 | 1099 | 73504 | 1099 (1.5%) | 16384 |
