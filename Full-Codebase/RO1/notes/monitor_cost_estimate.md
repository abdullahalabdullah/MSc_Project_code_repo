# Monitor API cost estimate — gpt-5-mini vs gpt-5

Estimate of `gpt-5-mini` monitor cost for the main RO1 run: n=200 per dataset,
N=8 samples/prompt, 3 hint types (`professor`, `grader_hacking`, `unethical`),
unhinted arm monitored once only (for the first hint type — its transcripts
don't change across hint types, so re-monitoring it per hint type would be
redundant spend).

## Method

- **Input tokens**: rebuilt the exact monitor prompt via
  `build_monitor_prompt()` (`monitorability-eval/src/monitor.py`) — template +
  per-hint-type wording + the model's full `raw_output` as `model_response` —
  then tokenized with `o200k_base` (the encoding `gpt-5-mini` resolves to).
  Computed from the **real main-run inference outputs**
  (`outputs-and-results/inference-outputs/Qwen_Qwen3-4B/{dataset}/professor/{hinted,unhinted}.jsonl`,
  n=200 × N=8 = 1,600 rows per arm per dataset), `professor` hint type only.
- **Output tokens**: no monitor has been run yet on this run's data
  (`monitor_outputs/` is empty), so output length was taken from an earlier
  pilot run (`MVP-RO1-v2/outputs-and-results/Old-Results/2026-07-06_20-55-48`,
  `Qwen_Qwen3-4B`, `professor` hint type, 50 instances × N=4). It was very
  stable — **~99 tokens/request** regardless of dataset or arm — because the
  monitor's output format is short and fixed (`REASONING: ... OUTPUT: True/False`).
  Assumed the same ~99 tokens/request applies across all 6 datasets and both
  hinted/unhinted arms.
- No prompt-caching discount applied: the shared prefix (template + hint
  wording) is only ~150 tokens, under OpenAI's 1024-token auto-cache
  threshold — full input rate applies.

**Pricing (per 1M tokens):**

| model | input | cached input | output |
|---|---|---|---|
| gpt-5-mini | $0.25 | $0.025 | $2.00 |
| gpt-5 | $1.25 | $0.125 | $10.00 |

gpt-5's rates are exactly 5× gpt-5-mini's on both input and output, so its
total cost is exactly 5× gpt-5-mini's for the same token counts.

## Average input tokens per monitor request (professor hint type)

| dataset | avg input tok — hinted | avg input tok — unhinted |
|---|---|---|
| mmlu_pro | 2,517 | 2,201 |
| wmdp | 2,394 | 2,306 |
| codemmlu | 2,660 | 2,382 |
| lsat_lr | 1,943 | 1,707 |
| medxpertqa | 1,897 | 1,827 |
| lexam | 2,287 | 2,282 |

Average output tokens/request (all datasets/arms): **~99**

## Cost projection — n=200, N=8, ×3 hint types, unhinted run once

Per dataset: hinted = 200 × 8 × 3 = 4,800 requests; unhinted = 200 × 8 × 1 =
1,600 requests → 6,400 requests/dataset.

| dataset | requests | input tokens | output tokens | cost (gpt-5-mini) | cost (gpt-5) |
|---|---|---|---|---|---|
| mmlu_pro | 6,400 | 15.60M | 0.63M | $5.17 | $25.85 |
| wmdp | 6,400 | 15.18M | 0.63M | $5.06 | $25.30 |
| codemmlu | 6,400 | 16.58M | 0.63M | $5.41 | $27.05 |
| lsat_lr | 6,400 | 12.06M | 0.63M | $4.28 | $21.40 |
| medxpertqa | 6,400 | 12.03M | 0.63M | $4.27 | $21.35 |
| lexam | 6,400 | 14.63M | 0.63M | $4.92 | $24.60 |
| **TOTAL** | **38,400** | **86.07M** | **3.80M** | **$29.12** | **$145.61** |

**≈ $29 on gpt-5-mini, ≈ $146 on gpt-5** for the full 6-dataset,
3-hint-type monitor pass at n=200/N=8. Output cost is a small share of the
total on either model (~$7.60 of $29 on gpt-5-mini, ~$38 of $146 on gpt-5);
input dominates. `codemmlu` is the most expensive dataset (longest CoTs),
`medxpertqa`/`lsat_lr` the cheapest.

## Caveats

- Input-token figures are exact for the `professor` hint type on this run's
  actual data. `grader_hacking`/`unethical` hint wording differs slightly
  (a few tokens in the fixed template slot only — negligible), and their CoT
  lengths were not separately measured here; assumed comparable to `professor`.
- Output-token estimate (~99) comes from a different (pilot) run, not this
  run's data, since the monitor hasn't been executed on the current inference
  outputs yet. Should be re-checked once real monitor runs are available for
  this dataset.
