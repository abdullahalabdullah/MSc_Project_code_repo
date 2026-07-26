# Response length model comparison — per dataset (hint types pooled)

- **Base**: `Qwen/Qwen3-4B`
- **Harmlessness DPO**: `er-abd/harmless-dpo-qwen3-4B-step-735`
- **Anti-sycophancy DPO**: `er-abd/anti-syc-dpo-qwen3-4B-step-520`

Columns: three models' **Median [Q1–Q3]** (skew-robust central tendency + spread, in tokens), then the two trained-vs-base **paired GMR** contrasts — the per-question geometric-mean ratio: each question's runs are collapsed to their per-question MEDIAN length per model, then GMR = exp(mean over questions of [ln median(trained_q) − ln median(base_q)]) (>1× = the trained model is typically longer on the same question; point estimates only — descriptive summaries of observed runs, no confidence intervals). The pairing is exact at every level: all three models ran the identical instances, and questions are keyed by (dataset, instance_id) so pooled levels pair correctly. Note the GMR is NOT the ratio of the displayed pooled medians (that is the marginal contrast; the paired GMR weights every question equally). Length stats INCLUDE truncated runs at the 16384-token cap (truncation is caused by length, so dropping them would bias length down; the cap understates the true tail, so estimates are conservative). Hint-effect (Hinted − Unhinted) and followed/not-followed rows are omitted; per-model hint effects, p90 and truncation live in the step-44 `length_*_level_tables` files.

*Unhinted is hint-independent, so it is taken from one hint arm (not tripled); the hinted model cells pool the 3 hint types. The hinted CONTRAST cells are the geometric mean of the dataset's three per-hint cell GMRs (equal weight per hint type — verifiable from the dataset×hint file; with this complete balanced grid it equals averaging per-(question, hint) log-ratios over hints within each question, then over questions), NOT a GMR over 3-hint-pooled per-question medians — a mixture median would track whichever runs straddle the middle of the mixed distribution instead of averaging the per-hint effects.*

## mmlu_pro

| Metric | Base | Harmlessness DPO | Anti-sycophancy DPO | Harmlessness DPO / Base | Anti-sycophancy DPO / Base |
|---|---|---|---|---|---|
| Hinted median tokens (X=1, all) | 1563 [817–3510] | 2124 [1102–4538] | 2315 [1222–5000] | 1.29× | 1.39× |
| Unhinted median tokens (X=0, all) | 1051 [684–2183] | 1363 [794–2786] | 1448 [804–2949] | 1.23× | 1.29× |

*n runs (hinted/unhinted): Base 4800/1600 · Harmlessness DPO 4800/1600 · Anti-sycophancy DPO 4800/1600*

## gpqa_diamond

| Metric | Base | Harmlessness DPO | Anti-sycophancy DPO | Harmlessness DPO / Base | Anti-sycophancy DPO / Base |
|---|---|---|---|---|---|
| Hinted median tokens (X=1, all) | 4751 [2705–7032] | 5754 [3520–8156] | 6066 [3699–8569] | 1.21× | 1.29× |
| Unhinted median tokens (X=0, all) | 4372 [2182–6784] | 5291 [2893–7627] | 5608 [3126–8054] | 1.21× | 1.30× |

*n runs (hinted/unhinted): Base 4752/1584 · Harmlessness DPO 4752/1584 · Anti-sycophancy DPO 4752/1584*

## lsat_ar

| Metric | Base | Harmlessness DPO | Anti-sycophancy DPO | Harmlessness DPO / Base | Anti-sycophancy DPO / Base |
|---|---|---|---|---|---|
| Hinted median tokens (X=1, all) | 6280 [4088–8932] | 7502 [4996–10399] | 7854 [5296–11008] | 1.18× | 1.24× |
| Unhinted median tokens (X=0, all) | 4717 [2838–7493] | 5448 [3415–8551] | 5494 [3406–8641] | 1.16× | 1.19× |

*n runs (hinted/unhinted): Base 4800/1600 · Harmlessness DPO 4800/1600 · Anti-sycophancy DPO 4800/1600*

## lsat_lr

| Metric | Base | Harmlessness DPO | Anti-sycophancy DPO | Harmlessness DPO / Base | Anti-sycophancy DPO / Base |
|---|---|---|---|---|---|
| Hinted median tokens (X=1, all) | 1174 [706–2747] | 1759 [855–3599] | 2094 [856–3990] | 1.32× | 1.42× |
| Unhinted median tokens (X=0, all) | 959 [647–1800] | 1180 [763–2354] | 1180 [741–2601] | 1.21× | 1.24× |

*n runs (hinted/unhinted): Base 4800/1600 · Harmlessness DPO 4800/1600 · Anti-sycophancy DPO 4800/1600*

## medxpertqa

| Metric | Base | Harmlessness DPO | Anti-sycophancy DPO | Harmlessness DPO / Base | Anti-sycophancy DPO / Base |
|---|---|---|---|---|---|
| Hinted median tokens (X=1, all) | 1442 [996–2043] | 1900 [1308–2707] | 2106 [1445–2992] | 1.31× | 1.42× |
| Unhinted median tokens (X=0, all) | 1408 [1002–1980] | 1907 [1317–2663] | 2068 [1416–2903] | 1.32× | 1.41× |

*n runs (hinted/unhinted): Base 4800/1600 · Harmlessness DPO 4800/1600 · Anti-sycophancy DPO 4800/1600*

## lexam

| Metric | Base | Harmlessness DPO | Anti-sycophancy DPO | Harmlessness DPO / Base | Anti-sycophancy DPO / Base |
|---|---|---|---|---|---|
| Hinted median tokens (X=1, all) | 1940 [1162–2887] | 2532 [1593–3560] | 2726 [1791–3806] | 1.29× | 1.40× |
| Unhinted median tokens (X=0, all) | 1676 [876–3050] | 2247 [1151–3702] | 2376 [1222–3939] | 1.24× | 1.30× |

*n runs (hinted/unhinted): Base 4800/1600 · Harmlessness DPO 4800/1600 · Anti-sycophancy DPO 4800/1600*

## codemmlu

| Metric | Base | Harmlessness DPO | Anti-sycophancy DPO | Harmlessness DPO / Base | Anti-sycophancy DPO / Base |
|---|---|---|---|---|---|
| Hinted median tokens (X=1, all) | 1744 [703–3900] | 2370 [952–5169] | 2632 [1042–5708] | 1.30× | 1.39× |
| Unhinted median tokens (X=0, all) | 1164 [541–2780] | 1499 [632–3897] | 1672 [662–4264] | 1.27× | 1.36× |

*n runs (hinted/unhinted): Base 4800/1600 · Harmlessness DPO 4800/1600 · Anti-sycophancy DPO 4800/1600*

## wmdp

| Metric | Base | Harmlessness DPO | Anti-sycophancy DPO | Harmlessness DPO / Base | Anti-sycophancy DPO / Base |
|---|---|---|---|---|---|
| Hinted median tokens (X=1, all) | 942 [632–1668] | 1298 [805–2261] | 1456 [855–2522] | 1.27× | 1.38× |
| Unhinted median tokens (X=0, all) | 838 [601–1405] | 1071 [706–1841] | 1154 [722–2134] | 1.23× | 1.30× |

*n runs (hinted/unhinted): Base 4800/1600 · Harmlessness DPO 4800/1600 · Anti-sycophancy DPO 4800/1600*
