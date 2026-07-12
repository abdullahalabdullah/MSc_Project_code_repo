# CoT token-length aggregate summary — runs pooled across all datasets and hint types

- **Model:** `Qwen/Qwen3-4B` (`Qwen_Qwen3-4B`)
- **Datasets:** mmlu_pro, gpqa_diamond, wmdp, codemmlu, lsat_ar, lsat_lr, medxpertqa, lexam
- **Hint types:** professor, grader_hacking, unethical

`token_count` is the full completion (thinking trace + final answer). Condition tables use valid runs only (not truncated, answer parsed), so followed + not-followed partition hinted exactly; bucket tables use every response. Use the truncation rate to set `max_new_tokens`.

## All (24 dataset × hint-type runs)

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 3061.5 | 2928.9 | 1933 | 208 | 16375 | 37958 | 4785 |
| Unhinted (X=0, all) | 2646.7 | 2754.3 | 1562 | 208 | 16316 | 37941 | 4785 |
| Hinted, followed (X=1, Y=1) | 3464.2 | 3167.9 | 2208 | 273 | 16375 | 16394 | 2974 |
| Hinted, not followed (X=1, Y=0) | 2755.4 | 2693.4 | 1748 | 208 | 16200 | 21564 | 3593 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 5190 | 18492 | 18702 | 16325 | 12042 | 5163 | 0 | 790 | 76704 | 790 (1.0%) | 16384 |
