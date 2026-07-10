# CoT token-length aggregate summary — runs pooled across all datasets and hint types

- **Model:** `er-abd/harmless-dpo-qwen3-4B-step-735` (`er-abd_harmless-dpo-qwen3-4B-step-735`)
- **Datasets:** mmlu_pro, gpqa_diamond, wmdp, codemmlu, lsat_ar, lsat_lr, medxpertqa, lexam
- **Hint types:** professor, unethical

`token_count` is the full completion (thinking trace + final answer). Condition tables use valid runs only (not truncated, answer parsed), so followed + not-followed partition hinted exactly; bucket tables use every response. Use the truncation rate to set `max_new_tokens`.

## All (15 dataset × hint-type runs)

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 3686.3 | 3221.7 | 242 | 16377 | 23651 | 2989 |
| Unhinted (X=0, all) | 3171.3 | 2968.5 | 261 | 16356 | 23577 | 2989 |
| Hinted, followed (X=1, Y=1) | 4007.4 | 3357.1 | 311 | 16377 | 10891 | 1904 |
| Hinted, not followed (X=1, Y=0) | 3412.3 | 3075.2 | 242 | 16221 | 12760 | 2130 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 1545 | 8556 | 11403 | 11859 | 9394 | 4498 | 0 | 681 | 47936 | 681 (1.4%) | 16384 |
