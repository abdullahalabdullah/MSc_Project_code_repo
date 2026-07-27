# CoT token-length aggregate summary — runs pooled across all datasets and hint types

- **Model:** `er-abd/anti-syc-dpo-qwen3-4B-step-520` (`er-abd_anti-syc-dpo-qwen3-4B-step-520`)
- **Datasets:** mmlu_pro, gpqa_diamond, wmdp, codemmlu, lsat_ar, lsat_lr, medxpertqa, lexam
- **Hint types:** professor, grader_hacking, unethical

`token_count` is the full completion (thinking trace + final answer). Condition tables use valid runs only (not truncated, answer parsed), so followed + not-followed partition hinted exactly; bucket tables use every response. Use the truncation rate to set `max_new_tokens`.

## All (24 dataset × hint-type runs)

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 3901.8 | 3310.8 | 2806 | 211 | 16379 | 37499 | 4786 |
| Unhinted (X=0, all) | 3267.2 | 3058.7 | 2187 | 249 | 16365 | 37559 | 4786 |
| Hinted, followed (X=1, Y=1) | 4242.1 | 3440.2 | 3048 | 280 | 16375 | 14957 | 2812 |
| Hinted, not followed (X=1, Y=0) | 3676.0 | 3202.3 | 2668 | 211 | 16379 | 22542 | 3677 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 2924 | 12778 | 16283 | 19484 | 15786 | 7814 | 0 | 1635 | 76704 | 1635 (2.1%) | 16384 |
