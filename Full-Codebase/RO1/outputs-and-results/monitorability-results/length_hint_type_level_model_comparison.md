# Response length model comparison — per hint type (datasets pooled)

- **Base**: `Qwen/Qwen3-4B`
- **Harmlessness DPO**: `er-abd/harmless-dpo-qwen3-4B-step-735`
- **Anti-sycophancy DPO**: `er-abd/anti-syc-dpo-qwen3-4B-step-520`

Columns: three models' **Median [Q1–Q3]** (skew-robust central tendency + spread, in tokens), then the two trained-vs-base **paired GMR** contrasts — the per-question geometric-mean ratio: each question's runs are collapsed to their per-question MEDIAN length per model, then GMR = exp(mean over questions of [ln median(trained_q) − ln median(base_q)]) (>1× = the trained model is typically longer on the same question; point estimates only — descriptive summaries of observed runs, no confidence intervals). The pairing is exact at every level: all three models ran the identical instances, and questions are keyed by (dataset, instance_id) so pooled levels pair correctly. Note the GMR is NOT the ratio of the displayed pooled medians (that is the marginal contrast; the paired GMR weights every question equally). Length stats INCLUDE truncated runs at the 16384-token cap (truncation is caused by length, so dropping them would bias length down; the cap understates the true tail, so estimates are conservative). Hint-effect (Hinted − Unhinted) and followed/not-followed rows are omitted; per-model hint effects, p90 and truncation live in the step-44 `length_*_level_tables` files.

*Datasets pooled; each dataset's unhinted arm is distinct, so no double counting. Note the pool mixes very different length regimes — read as a summary, per-dataset tables are the meaningful unit. Accordingly, BOTH contrast cells are the geometric mean of the hint type's eight per-dataset cell GMRs (equal weight per dataset — verifiable from the dataset×hint file), not a GMR over the pooled question set.*

## Professor hint

| Metric | Base | Harmlessness DPO | Anti-sycophancy DPO | Harmlessness DPO / Base | Anti-sycophancy DPO / Base |
|---|---|---|---|---|---|
| Hinted median tokens (X=1, all) | 1838 [912–3892] | 2436 [1209–4863] | 2676 [1304–5221] | 1.26× | 1.34× |
| Unhinted median tokens (X=0, all) | 1583 [808–3527] | 2068 [1016–4347] | 2242 [1047–4644] | 1.23× | 1.30× |

*n runs (hinted/unhinted): Base 12784/12784 · Harmlessness DPO 12784/12784 · Anti-sycophancy DPO 12784/12784*

## Grader-hacking hint

| Metric | Base | Harmlessness DPO | Anti-sycophancy DPO | Harmlessness DPO / Base | Anti-sycophancy DPO / Base |
|---|---|---|---|---|---|
| Hinted median tokens (X=1, all) | 2058 [920–4294] | 2708 [1258–5402] | 3099 [1456–5924] | 1.26× | 1.39× |
| Unhinted median tokens (X=0, all) | 1583 [808–3527] | 2068 [1016–4347] | 2242 [1047–4644] | 1.23× | 1.30× |

*n runs (hinted/unhinted): Base 12784/12784 · Harmlessness DPO 12784/12784 · Anti-sycophancy DPO 12784/12784*

## Unethical hint

| Metric | Base | Harmlessness DPO | Anti-sycophancy DPO | Harmlessness DPO / Base | Anti-sycophancy DPO / Base |
|---|---|---|---|---|---|
| Hinted median tokens (X=1, all) | 2024 [1003–4570] | 2730 [1402–5661] | 2916 [1541–5958] | 1.29× | 1.36× |
| Unhinted median tokens (X=0, all) | 1583 [808–3527] | 2068 [1016–4347] | 2242 [1047–4644] | 1.23× | 1.30× |

*n runs (hinted/unhinted): Base 12784/12784 · Harmlessness DPO 12784/12784 · Anti-sycophancy DPO 12784/12784*
