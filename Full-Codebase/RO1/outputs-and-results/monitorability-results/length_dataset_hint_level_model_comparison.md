# Response length model comparison — per (dataset, hint type), no pooling

- **Base**: `Qwen/Qwen3-4B`
- **Harmlessness DPO**: `er-abd/harmless-dpo-qwen3-4B-step-735`
- **Anti-sycophancy DPO**: `er-abd/anti-syc-dpo-qwen3-4B-step-520`

Columns: three models' **Median [Q1–Q3]** (skew-robust central tendency + spread, in tokens), then the two trained-vs-base **paired GMR** contrasts — the per-question geometric-mean ratio: each question's runs are collapsed to their per-question MEDIAN length per model, then GMR = exp(mean over questions of [ln median(trained_q) − ln median(base_q)]) (>1× = the trained model is typically longer on the same question; point estimates only — descriptive summaries of observed runs, no confidence intervals). The pairing is exact at every level: all three models ran the identical instances, and questions are keyed by (dataset, instance_id) so pooled levels pair correctly. Note the GMR is NOT the ratio of the displayed pooled medians (that is the marginal contrast; the paired GMR weights every question equally). Length stats INCLUDE truncated runs at the 16384-token cap (truncation is caused by length, so dropping them would bias length down; the cap understates the true tail, so estimates are conservative). Hint-effect (Hinted − Unhinted) and followed/not-followed rows are omitted; per-model hint effects, p90 and truncation live in the step-44 `length_*_level_tables` files.

## mmlu_pro

### mmlu_pro — Professor hint

| Metric | Base | Harmlessness DPO | Anti-sycophancy DPO | Harmlessness DPO / Base | Anti-sycophancy DPO / Base |
|---|---|---|---|---|---|
| Hinted median tokens (X=1, all) | 1402 [766–2799] | 1870 [997–3612] | 1974 [1039–3824] | 1.28× | 1.35× |
| Unhinted median tokens (X=0, all) | 1051 [684–2183] | 1363 [794–2786] | 1448 [804–2949] | 1.23× | 1.29× |

*n runs (hinted/unhinted): Base 1600/1600 · Harmlessness DPO 1600/1600 · Anti-sycophancy DPO 1600/1600*

### mmlu_pro — Grader-hacking hint

| Metric | Base | Harmlessness DPO | Anti-sycophancy DPO | Harmlessness DPO / Base | Anti-sycophancy DPO / Base |
|---|---|---|---|---|---|
| Hinted median tokens (X=1, all) | 1602 [740–3532] | 2133 [986–4454] | 2475 [1091–5290] | 1.26× | 1.41× |
| Unhinted median tokens (X=0, all) | 1051 [684–2183] | 1363 [794–2786] | 1448 [804–2949] | 1.23× | 1.29× |

*n runs (hinted/unhinted): Base 1600/1600 · Harmlessness DPO 1600/1600 · Anti-sycophancy DPO 1600/1600*

### mmlu_pro — Unethical hint

| Metric | Base | Harmlessness DPO | Anti-sycophancy DPO | Harmlessness DPO / Base | Anti-sycophancy DPO / Base |
|---|---|---|---|---|---|
| Hinted median tokens (X=1, all) | 1709 [955–4352] | 2440 [1348–5570] | 2554 [1498–5907] | 1.34× | 1.41× |
| Unhinted median tokens (X=0, all) | 1051 [684–2183] | 1363 [794–2786] | 1448 [804–2949] | 1.23× | 1.29× |

*n runs (hinted/unhinted): Base 1600/1600 · Harmlessness DPO 1600/1600 · Anti-sycophancy DPO 1600/1600*

## gpqa_diamond

### gpqa_diamond — Professor hint

| Metric | Base | Harmlessness DPO | Anti-sycophancy DPO | Harmlessness DPO / Base | Anti-sycophancy DPO / Base |
|---|---|---|---|---|---|
| Hinted median tokens (X=1, all) | 4712 [2694–6792] | 5698 [3538–7842] | 6091 [3689–8288] | 1.21× | 1.30× |
| Unhinted median tokens (X=0, all) | 4372 [2182–6784] | 5291 [2893–7627] | 5608 [3126–8054] | 1.21× | 1.30× |

*n runs (hinted/unhinted): Base 1584/1584 · Harmlessness DPO 1584/1584 · Anti-sycophancy DPO 1584/1584*

### gpqa_diamond — Grader-hacking hint

| Metric | Base | Harmlessness DPO | Anti-sycophancy DPO | Harmlessness DPO / Base | Anti-sycophancy DPO / Base |
|---|---|---|---|---|---|
| Hinted median tokens (X=1, all) | 4848 [2733–7182] | 5871 [3476–8309] | 6246 [3811–8703] | 1.19× | 1.28× |
| Unhinted median tokens (X=0, all) | 4372 [2182–6784] | 5291 [2893–7627] | 5608 [3126–8054] | 1.21× | 1.30× |

*n runs (hinted/unhinted): Base 1584/1584 · Harmlessness DPO 1584/1584 · Anti-sycophancy DPO 1584/1584*

### gpqa_diamond — Unethical hint

| Metric | Base | Harmlessness DPO | Anti-sycophancy DPO | Harmlessness DPO / Base | Anti-sycophancy DPO / Base |
|---|---|---|---|---|---|
| Hinted median tokens (X=1, all) | 4728 [2676–7181] | 5716 [3586–8428] | 5849 [3617–8720] | 1.22× | 1.28× |
| Unhinted median tokens (X=0, all) | 4372 [2182–6784] | 5291 [2893–7627] | 5608 [3126–8054] | 1.21× | 1.30× |

*n runs (hinted/unhinted): Base 1584/1584 · Harmlessness DPO 1584/1584 · Anti-sycophancy DPO 1584/1584*

## lsat_ar

### lsat_ar — Professor hint

| Metric | Base | Harmlessness DPO | Anti-sycophancy DPO | Harmlessness DPO / Base | Anti-sycophancy DPO / Base |
|---|---|---|---|---|---|
| Hinted median tokens (X=1, all) | 5463 [3417–8444] | 6634 [4049–9693] | 6898 [4327–10025] | 1.19× | 1.24× |
| Unhinted median tokens (X=0, all) | 4717 [2838–7493] | 5448 [3415–8551] | 5494 [3406–8641] | 1.16× | 1.19× |

*n runs (hinted/unhinted): Base 1600/1600 · Harmlessness DPO 1600/1600 · Anti-sycophancy DPO 1600/1600*

### lsat_ar — Grader-hacking hint

| Metric | Base | Harmlessness DPO | Anti-sycophancy DPO | Harmlessness DPO / Base | Anti-sycophancy DPO / Base |
|---|---|---|---|---|---|
| Hinted median tokens (X=1, all) | 6137 [3982–9026] | 7304 [4784–10508] | 8018 [5386–11596] | 1.19× | 1.28× |
| Unhinted median tokens (X=0, all) | 4717 [2838–7493] | 5448 [3415–8551] | 5494 [3406–8641] | 1.16× | 1.19× |

*n runs (hinted/unhinted): Base 1600/1600 · Harmlessness DPO 1600/1600 · Anti-sycophancy DPO 1600/1600*

### lsat_ar — Unethical hint

| Metric | Base | Harmlessness DPO | Anti-sycophancy DPO | Harmlessness DPO / Base | Anti-sycophancy DPO / Base |
|---|---|---|---|---|---|
| Hinted median tokens (X=1, all) | 6986 [4881–9262] | 8441 [5986–10832] | 8626 [6154–11244] | 1.17× | 1.21× |
| Unhinted median tokens (X=0, all) | 4717 [2838–7493] | 5448 [3415–8551] | 5494 [3406–8641] | 1.16× | 1.19× |

*n runs (hinted/unhinted): Base 1600/1600 · Harmlessness DPO 1600/1600 · Anti-sycophancy DPO 1600/1600*

## lsat_lr

### lsat_lr — Professor hint

| Metric | Base | Harmlessness DPO | Anti-sycophancy DPO | Harmlessness DPO / Base | Anti-sycophancy DPO / Base |
|---|---|---|---|---|---|
| Hinted median tokens (X=1, all) | 1055 [700–2366] | 1420 [832–3018] | 1481 [797–3438] | 1.26× | 1.33× |
| Unhinted median tokens (X=0, all) | 959 [647–1800] | 1180 [763–2354] | 1180 [741–2601] | 1.21× | 1.24× |

*n runs (hinted/unhinted): Base 1600/1600 · Harmlessness DPO 1600/1600 · Anti-sycophancy DPO 1600/1600*

### lsat_lr — Grader-hacking hint

| Metric | Base | Harmlessness DPO | Anti-sycophancy DPO | Harmlessness DPO / Base | Anti-sycophancy DPO / Base |
|---|---|---|---|---|---|
| Hinted median tokens (X=1, all) | 1101 [678–2692] | 1739 [806–3545] | 2119 [840–4182] | 1.33× | 1.46× |
| Unhinted median tokens (X=0, all) | 959 [647–1800] | 1180 [763–2354] | 1180 [741–2601] | 1.21× | 1.24× |

*n runs (hinted/unhinted): Base 1600/1600 · Harmlessness DPO 1600/1600 · Anti-sycophancy DPO 1600/1600*

### lsat_lr — Unethical hint

| Metric | Base | Harmlessness DPO | Anti-sycophancy DPO | Harmlessness DPO / Base | Anti-sycophancy DPO / Base |
|---|---|---|---|---|---|
| Hinted median tokens (X=1, all) | 1475 [740–3063] | 2442 [957–3969] | 2788 [970–4219] | 1.37× | 1.47× |
| Unhinted median tokens (X=0, all) | 959 [647–1800] | 1180 [763–2354] | 1180 [741–2601] | 1.21× | 1.24× |

*n runs (hinted/unhinted): Base 1600/1600 · Harmlessness DPO 1600/1600 · Anti-sycophancy DPO 1600/1600*

## medxpertqa

### medxpertqa — Professor hint

| Metric | Base | Harmlessness DPO | Anti-sycophancy DPO | Harmlessness DPO / Base | Anti-sycophancy DPO / Base |
|---|---|---|---|---|---|
| Hinted median tokens (X=1, all) | 1468 [1022–2090] | 1978 [1376–2777] | 2182 [1531–3057] | 1.33× | 1.47× |
| Unhinted median tokens (X=0, all) | 1408 [1002–1980] | 1907 [1317–2663] | 2068 [1416–2903] | 1.32× | 1.41× |

*n runs (hinted/unhinted): Base 1600/1600 · Harmlessness DPO 1600/1600 · Anti-sycophancy DPO 1600/1600*

### medxpertqa — Grader-hacking hint

| Metric | Base | Harmlessness DPO | Anti-sycophancy DPO | Harmlessness DPO / Base | Anti-sycophancy DPO / Base |
|---|---|---|---|---|---|
| Hinted median tokens (X=1, all) | 1562 [1038–2169] | 2094 [1378–3029] | 2327 [1584–3265] | 1.32× | 1.46× |
| Unhinted median tokens (X=0, all) | 1408 [1002–1980] | 1907 [1317–2663] | 2068 [1416–2903] | 1.32× | 1.41× |

*n runs (hinted/unhinted): Base 1600/1600 · Harmlessness DPO 1600/1600 · Anti-sycophancy DPO 1600/1600*

### medxpertqa — Unethical hint

| Metric | Base | Harmlessness DPO | Anti-sycophancy DPO | Harmlessness DPO / Base | Anti-sycophancy DPO / Base |
|---|---|---|---|---|---|
| Hinted median tokens (X=1, all) | 1347 [933–1834] | 1692 [1214–2395] | 1814 [1244–2650] | 1.27× | 1.35× |
| Unhinted median tokens (X=0, all) | 1408 [1002–1980] | 1907 [1317–2663] | 2068 [1416–2903] | 1.32× | 1.41× |

*n runs (hinted/unhinted): Base 1600/1600 · Harmlessness DPO 1600/1600 · Anti-sycophancy DPO 1600/1600*

## lexam

### lexam — Professor hint

| Metric | Base | Harmlessness DPO | Anti-sycophancy DPO | Harmlessness DPO / Base | Anti-sycophancy DPO / Base |
|---|---|---|---|---|---|
| Hinted median tokens (X=1, all) | 1880 [1132–2894] | 2536 [1543–3581] | 2642 [1728–3836] | 1.29× | 1.38× |
| Unhinted median tokens (X=0, all) | 1676 [876–3050] | 2247 [1151–3702] | 2376 [1222–3939] | 1.24× | 1.30× |

*n runs (hinted/unhinted): Base 1600/1600 · Harmlessness DPO 1600/1600 · Anti-sycophancy DPO 1600/1600*

### lexam — Grader-hacking hint

| Metric | Base | Harmlessness DPO | Anti-sycophancy DPO | Harmlessness DPO / Base | Anti-sycophancy DPO / Base |
|---|---|---|---|---|---|
| Hinted median tokens (X=1, all) | 2238 [1148–3359] | 2771 [1574–3943] | 3196 [1896–4290] | 1.27× | 1.40× |
| Unhinted median tokens (X=0, all) | 1676 [876–3050] | 2247 [1151–3702] | 2376 [1222–3939] | 1.24× | 1.30× |

*n runs (hinted/unhinted): Base 1600/1600 · Harmlessness DPO 1600/1600 · Anti-sycophancy DPO 1600/1600*

### lexam — Unethical hint

| Metric | Base | Harmlessness DPO | Anti-sycophancy DPO | Harmlessness DPO / Base | Anti-sycophancy DPO / Base |
|---|---|---|---|---|---|
| Hinted median tokens (X=1, all) | 1832 [1194–2457] | 2392 [1653–3185] | 2558 [1801–3316] | 1.32× | 1.41× |
| Unhinted median tokens (X=0, all) | 1676 [876–3050] | 2247 [1151–3702] | 2376 [1222–3939] | 1.24× | 1.30× |

*n runs (hinted/unhinted): Base 1600/1600 · Harmlessness DPO 1600/1600 · Anti-sycophancy DPO 1600/1600*

## codemmlu

### codemmlu — Professor hint

| Metric | Base | Harmlessness DPO | Anti-sycophancy DPO | Harmlessness DPO / Base | Anti-sycophancy DPO / Base |
|---|---|---|---|---|---|
| Hinted median tokens (X=1, all) | 1456 [614–3562] | 1960 [786–4688] | 2216 [843–5347] | 1.28× | 1.39× |
| Unhinted median tokens (X=0, all) | 1164 [541–2780] | 1499 [632–3897] | 1672 [662–4264] | 1.27× | 1.36× |

*n runs (hinted/unhinted): Base 1600/1600 · Harmlessness DPO 1600/1600 · Anti-sycophancy DPO 1600/1600*

### codemmlu — Grader-hacking hint

| Metric | Base | Harmlessness DPO | Anti-sycophancy DPO | Harmlessness DPO / Base | Anti-sycophancy DPO / Base |
|---|---|---|---|---|---|
| Hinted median tokens (X=1, all) | 1616 [630–3686] | 2166 [792–5032] | 2403 [878–5694] | 1.29× | 1.39× |
| Unhinted median tokens (X=0, all) | 1164 [541–2780] | 1499 [632–3897] | 1672 [662–4264] | 1.27× | 1.36× |

*n runs (hinted/unhinted): Base 1600/1600 · Harmlessness DPO 1600/1600 · Anti-sycophancy DPO 1600/1600*

### codemmlu — Unethical hint

| Metric | Base | Harmlessness DPO | Anti-sycophancy DPO | Harmlessness DPO / Base | Anti-sycophancy DPO / Base |
|---|---|---|---|---|---|
| Hinted median tokens (X=1, all) | 2176 [945–4501] | 3057 [1396–5620] | 3081 [1472–6098] | 1.32× | 1.39× |
| Unhinted median tokens (X=0, all) | 1164 [541–2780] | 1499 [632–3897] | 1672 [662–4264] | 1.27× | 1.36× |

*n runs (hinted/unhinted): Base 1600/1600 · Harmlessness DPO 1600/1600 · Anti-sycophancy DPO 1600/1600*

## wmdp

### wmdp — Professor hint

| Metric | Base | Harmlessness DPO | Anti-sycophancy DPO | Harmlessness DPO / Base | Anti-sycophancy DPO / Base |
|---|---|---|---|---|---|
| Hinted median tokens (X=1, all) | 938 [633–1643] | 1246 [804–2101] | 1361 [834–2304] | 1.24× | 1.31× |
| Unhinted median tokens (X=0, all) | 838 [601–1405] | 1071 [706–1841] | 1154 [722–2134] | 1.23× | 1.30× |

*n runs (hinted/unhinted): Base 1600/1600 · Harmlessness DPO 1600/1600 · Anti-sycophancy DPO 1600/1600*

### wmdp — Grader-hacking hint

| Metric | Base | Harmlessness DPO | Anti-sycophancy DPO | Harmlessness DPO / Base | Anti-sycophancy DPO / Base |
|---|---|---|---|---|---|
| Hinted median tokens (X=1, all) | 962 [621–1852] | 1310 [794–2404] | 1579 [860–2747] | 1.26× | 1.43× |
| Unhinted median tokens (X=0, all) | 838 [601–1405] | 1071 [706–1841] | 1154 [722–2134] | 1.23× | 1.30× |

*n runs (hinted/unhinted): Base 1600/1600 · Harmlessness DPO 1600/1600 · Anti-sycophancy DPO 1600/1600*

### wmdp — Unethical hint

| Metric | Base | Harmlessness DPO | Anti-sycophancy DPO | Harmlessness DPO / Base | Anti-sycophancy DPO / Base |
|---|---|---|---|---|---|
| Hinted median tokens (X=1, all) | 933 [643–1547] | 1361 [813–2251] | 1467 [867–2434] | 1.31× | 1.40× |
| Unhinted median tokens (X=0, all) | 838 [601–1405] | 1071 [706–1841] | 1154 [722–2134] | 1.23× | 1.30× |

*n runs (hinted/unhinted): Base 1600/1600 · Harmlessness DPO 1600/1600 · Anti-sycophancy DPO 1600/1600*
