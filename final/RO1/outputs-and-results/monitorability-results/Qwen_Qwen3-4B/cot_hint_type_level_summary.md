# CoT token-length hint-type-level summary — runs pooled across datasets

- **Model:** `Qwen/Qwen3-4B` (`Qwen_Qwen3-4B`)
- **Datasets:** mmlu_pro, gpqa_diamond, wmdp, codemmlu, lsat_ar, lsat_lr, medxpertqa, lexam
- **Hint types:** professor, grader_hacking, unethical

`token_count` is the full completion (thinking trace + final answer). Condition tables use valid runs only (not truncated, answer parsed), so followed + not-followed partition hinted exactly; bucket tables use every response. Use the truncation rate to set `max_new_tokens`.

## professor (pooled over 8 datasets)

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 2896.8 | 2830.0 | 1818 | 213 | 16375 | 12665 | 1595 |
| Unhinted (X=0, all) | 2646.7 | 2754.3 | 1562 | 208 | 16316 | 12647 | 1595 |
| Hinted, followed (X=1, Y=1) | 3550.7 | 3337.4 | 2232 | 277 | 16375 | 3654 | 750 |
| Hinted, not followed (X=1, Y=0) | 2631.6 | 2549.0 | 1691 | 213 | 16200 | 9011 | 1381 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 1753 | 6313 | 6394 | 5446 | 3821 | 1588 | 0 | 253 | 25568 | 253 (1.0%) | 16384 |

## grader_hacking (pooled over 8 datasets)

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 3103.9 | 2967.5 | 2025 | 208 | 16313 | 12634 | 1595 |
| Unhinted (X=0, all) | 2646.7 | 2754.3 | 1562 | 208 | 16316 | 12647 | 1595 |
| Hinted, followed (X=1, Y=1) | 3886.5 | 3347.2 | 2786 | 273 | 16313 | 3590 | 792 |
| Hinted, not followed (X=1, Y=0) | 2793.2 | 2741.6 | 1768 | 208 | 16143 | 9044 | 1407 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 1848 | 6117 | 5959 | 5589 | 4035 | 1737 | 0 | 283 | 25568 | 283 (1.1%) | 16384 |

## unethical (pooled over 8 datasets)

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 3184.0 | 2979.7 | 1996 | 219 | 16338 | 12659 | 1595 |
| Unhinted (X=0, all) | 2646.7 | 2754.3 | 1562 | 208 | 16316 | 12647 | 1595 |
| Hinted, followed (X=1, Y=1) | 3263.9 | 3004.4 | 2026 | 293 | 16338 | 9150 | 1432 |
| Hinted, not followed (X=1, Y=0) | 2975.7 | 2904.6 | 1871 | 219 | 14793 | 3509 | 805 |

Responses per token-length bucket (every response, both arms; truncated responses count in the top bucket):

| 0-512 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-32768 | 32768+ | Total | Truncated (rate) | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| 1589 | 6062 | 6349 | 5290 | 4186 | 1838 | 0 | 254 | 25568 | 254 (1.0%) | 16384 |

## Summary

| Hint type | Hinted (mean (std) · median) | Unhinted (mean (std) · median) | Hinted & followed (mean (std) · median) | Hinted & not followed (mean (std) · median) | Gap (Hinted mean − Unhinted mean) |
|---|---|---|---|---|---|
| professor | 2896.8 (2830.0) · 1818 | 2646.7 (2754.3) · 1562 | 3550.7 (3337.4) · 2232 | 2631.6 (2549.0) · 1691 | 250.1 |
| grader_hacking | 3103.9 (2967.5) · 2025 | 2646.7 (2754.3) · 1562 | 3886.5 (3347.2) · 2786 | 2793.2 (2741.6) · 1768 | 457.2 |
| unethical | 3184.0 (2979.7) · 1996 | 2646.7 (2754.3) · 1562 | 3263.9 (3004.4) · 2026 | 2975.7 (2904.6) · 1871 | 537.3 |
