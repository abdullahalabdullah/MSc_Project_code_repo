# CoT token-length aggregate summary — runs pooled across all datasets and hint types

- **Model:** `er-abd_harmless-dpo-qwen3-4B-step-735` (`er-abd_harmless-dpo-qwen3-4B-step-735`)
- **Datasets:** mmlu_pro, gpqa_diamond, wmdp, codemmlu, lsat_ar, lsat_lr, medxpertqa, lexam
- **Hint types:** professor, grader_hacking, unethical

`token_count` is the full completion (thinking trace + final answer). Condition tables use valid runs only (not truncated, answer parsed), so followed + not-followed partition hinted exactly; bucket tables use every response. Use the truncation rate to set `max_new_tokens`.

## All (24 dataset × hint-type runs)

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 3702.2 | 3236.3 | 2579 | 227 | 16377 | 37796 | 4783 |
| Unhinted (X=0, all) | 3123.6 | 2957.1 | 2020 | 261 | 16356 | 37744 | 4783 |
| Hinted, followed (X=1, Y=1) | 4063.9 | 3374.3 | 2875 | 276 | 16377 | 15625 | 2867 |
| Hinted, not followed (X=1, Y=0) | 3447.4 | 3110.3 | 2386 | 227 | 16332 | 22171 | 3611 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 2595 | 14089 | 17980 | 18791 | 14957 | 7168 | 0 | 1124 | 76704 | 1124 (1.5%) | 16384 |
