# Response length — per (dataset, hint type), no pooling

Model: `Qwen/Qwen3-4B`

Columns: **Median [Q1–Q3]** primary central tendency + spread (skew-robust) · **p90** upper tail · **n runs (inst)** responses and distinct instances behind the cell. (Arithmetic Mean (SD) is omitted here to keep the table median-first; it lives in `length_dataset_hint_median_vs_mean.md`.) Length stats INCLUDE truncated runs at the 16384-token cap (truncation is caused by length, so dropping them would bias length down; the cap understates the true tail, so estimates are conservative). **Truncation** is the separate rate of capped responses over both arms. **Hint effect** is the Hinted-vs-Unhinted contrast: Δ median (tokens), H/U median ratio — the robust analogue of a mean gap — and the paired **GM ratio** (per-question at the dataset×hint level; GM of the constituent cell GMRs at the pooled levels). (Descriptive summaries of observed runs — no confidence intervals.)

This finest level reports the **GM ratio (paired)** = the per-question geometric-mean ratio: each question's hinted and unhinted runs are collapsed to their per-question MEDIAN length, then GM ratio = exp(mean over questions of [ln median(h_q) − ln median(u_q)]) — the typical per-question multiplicative hint effect on length (median within a question, geometric mean across questions), skew-robust and the single best 'how much longer' number. The pooled levels report the geometric mean of their constituent dataset×hint cell GMRs (equal weight per cell) — a plain pooled-run GMR would mix hint types inside a question's median (dataset level) or weight datasets by question count (hint-type level); the GM-of-cells rule keeps every level derivable from this file and matches the summary matrix's geometric margins exactly.

## mmlu_pro

### mmlu_pro — Professor hint

| Condition | Median [Q1–Q3] | p90 | n runs (inst) |
|---|---|---|---|
| Hinted (X=1, all) | 1402 [766–2799] | 6527 | 1600 (200) |
| Unhinted (X=0, all) | 1051 [684–2183] | 5271 | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 1302 [762–3262] | 7922 | 342 (70) |
| Hinted, not followed (X=1, Y=0) | 1417 [771–2747] | 5649 | 1258 (180) |

**Hint effect (Hinted − Unhinted):** Δ median = +351 tok · median ratio = 1.33× · GM ratio (paired) = 1.21×

**Truncation (both arms):** 0.2% (7/3200)

### mmlu_pro — Grader-hacking hint

| Condition | Median [Q1–Q3] | p90 | n runs (inst) |
|---|---|---|---|
| Hinted (X=1, all) | 1602 [740–3532] | 6994 | 1600 (200) |
| Unhinted (X=0, all) | 1051 [684–2183] | 5271 | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 2344 [1050–5352] | 8745 | 393 (93) |
| Hinted, not followed (X=1, Y=0) | 1368 [688–3034] | 6367 | 1207 (185) |

**Hint effect (Hinted − Unhinted):** Δ median = +550 tok · median ratio = 1.52× · GM ratio (paired) = 1.32×

**Truncation (both arms):** 0.3% (11/3200)

### mmlu_pro — Unethical hint

| Condition | Median [Q1–Q3] | p90 | n runs (inst) |
|---|---|---|---|
| Hinted (X=1, all) | 1709 [955–4352] | 7333 | 1600 (200) |
| Unhinted (X=0, all) | 1051 [684–2183] | 5271 | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 1676 [976–4206] | 7159 | 1334 (199) |
| Hinted, not followed (X=1, Y=0) | 1997 [802–5360] | 8134 | 266 (88) |

**Hint effect (Hinted − Unhinted):** Δ median = +658 tok · median ratio = 1.63× · GM ratio (paired) = 1.53×

**Truncation (both arms):** 0.3% (11/3200)

## gpqa_diamond

### gpqa_diamond — Professor hint

| Condition | Median [Q1–Q3] | p90 | n runs (inst) |
|---|---|---|---|
| Hinted (X=1, all) | 4712 [2694–6792] | 9347 | 1584 (198) |
| Unhinted (X=0, all) | 4372 [2182–6784] | 9489 | 1584 (198) |
| Hinted, followed (X=1, Y=1) | 5197 [3327–7091] | 10304 | 766 (131) |
| Hinted, not followed (X=1, Y=0) | 4198 [2287–6501] | 8666 | 818 (147) |

**Hint effect (Hinted − Unhinted):** Δ median = +339 tok · median ratio = 1.08× · GM ratio (paired) = 1.10×

**Truncation (both arms):** 1.0% (31/3168)

### gpqa_diamond — Grader-hacking hint

| Condition | Median [Q1–Q3] | p90 | n runs (inst) |
|---|---|---|---|
| Hinted (X=1, all) | 4848 [2733–7182] | 9683 | 1584 (198) |
| Unhinted (X=0, all) | 4372 [2182–6784] | 9489 | 1584 (198) |
| Hinted, followed (X=1, Y=1) | 5319 [3470–7566] | 10096 | 695 (136) |
| Hinted, not followed (X=1, Y=0) | 4276 [2320–6893] | 9384 | 889 (155) |

**Hint effect (Hinted − Unhinted):** Δ median = +476 tok · median ratio = 1.11× · GM ratio (paired) = 1.14×

**Truncation (both arms):** 1.1% (35/3168)

### gpqa_diamond — Unethical hint

| Condition | Median [Q1–Q3] | p90 | n runs (inst) |
|---|---|---|---|
| Hinted (X=1, all) | 4728 [2676–7181] | 9924 | 1584 (198) |
| Unhinted (X=0, all) | 4372 [2182–6784] | 9489 | 1584 (198) |
| Hinted, followed (X=1, Y=1) | 4882 [2952–7190] | 9920 | 1266 (186) |
| Hinted, not followed (X=1, Y=0) | 3949 [1868–6898] | 9836 | 318 (80) |

**Hint effect (Hinted − Unhinted):** Δ median = +356 tok · median ratio = 1.08× · GM ratio (paired) = 1.11×

**Truncation (both arms):** 1.0% (32/3168)

## lsat_ar

### lsat_ar — Professor hint

| Condition | Median [Q1–Q3] | p90 | n runs (inst) |
|---|---|---|---|
| Hinted (X=1, all) | 5463 [3417–8444] | 11112 | 1600 (200) |
| Unhinted (X=0, all) | 4717 [2838–7493] | 10938 | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 9156 [6777–11295] | 13519 | 309 (87) |
| Hinted, not followed (X=1, Y=0) | 4708 [3094–7407] | 9991 | 1291 (192) |

**Hint effect (Hinted − Unhinted):** Δ median = +746 tok · median ratio = 1.16× · GM ratio (paired) = 1.16×

**Truncation (both arms):** 0.6% (19/3200)

### lsat_ar — Grader-hacking hint

| Condition | Median [Q1–Q3] | p90 | n runs (inst) |
|---|---|---|---|
| Hinted (X=1, all) | 6137 [3982–9026] | 11880 | 1600 (200) |
| Unhinted (X=0, all) | 4717 [2838–7493] | 10938 | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 9399 [7120–11834] | 14103 | 343 (97) |
| Hinted, not followed (X=1, Y=0) | 5434 [3573–7871] | 10685 | 1257 (194) |

**Hint effect (Hinted − Unhinted):** Δ median = +1420 tok · median ratio = 1.30× · GM ratio (paired) = 1.30×

**Truncation (both arms):** 1.0% (31/3200)

### lsat_ar — Unethical hint

| Condition | Median [Q1–Q3] | p90 | n runs (inst) |
|---|---|---|---|
| Hinted (X=1, all) | 6986 [4881–9262] | 11332 | 1600 (200) |
| Unhinted (X=0, all) | 4717 [2838–7493] | 10938 | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 8116 [6249–10046] | 11875 | 844 (170) |
| Hinted, not followed (X=1, Y=0) | 5645 [4246–7979] | 10234 | 756 (159) |

**Hint effect (Hinted − Unhinted):** Δ median = +2269 tok · median ratio = 1.48× · GM ratio (paired) = 1.50×

**Truncation (both arms):** 0.9% (29/3200)

## lsat_lr

### lsat_lr — Professor hint

| Condition | Median [Q1–Q3] | p90 | n runs (inst) |
|---|---|---|---|
| Hinted (X=1, all) | 1055 [700–2366] | 3869 | 1600 (200) |
| Unhinted (X=0, all) | 959 [647–1800] | 3474 | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 1882 [1115–3452] | 4587 | 200 (49) |
| Hinted, not followed (X=1, Y=0) | 966 [669–2085] | 3706 | 1400 (193) |

**Hint effect (Hinted − Unhinted):** Δ median = +96 tok · median ratio = 1.10× · GM ratio (paired) = 1.12×

**Truncation (both arms):** 0.1% (2/3200)

### lsat_lr — Grader-hacking hint

| Condition | Median [Q1–Q3] | p90 | n runs (inst) |
|---|---|---|---|
| Hinted (X=1, all) | 1101 [678–2692] | 4427 | 1600 (200) |
| Unhinted (X=0, all) | 959 [647–1800] | 3474 | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 2367 [1409–4015] | 5621 | 225 (56) |
| Hinted, not followed (X=1, Y=0) | 973 [647–2398] | 4006 | 1375 (189) |

**Hint effect (Hinted − Unhinted):** Δ median = +142 tok · median ratio = 1.15× · GM ratio (paired) = 1.19×

**Truncation (both arms):** 0.0% (1/3200)

### lsat_lr — Unethical hint

| Condition | Median [Q1–Q3] | p90 | n runs (inst) |
|---|---|---|---|
| Hinted (X=1, all) | 1475 [740–3063] | 4147 | 1600 (200) |
| Unhinted (X=0, all) | 959 [647–1800] | 3474 | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 2542 [1304–3589] | 4534 | 638 (124) |
| Hinted, not followed (X=1, Y=0) | 946 [632–2386] | 3711 | 962 (159) |

**Hint effect (Hinted − Unhinted):** Δ median = +516 tok · median ratio = 1.54× · GM ratio (paired) = 1.35×

**Truncation (both arms):** 0.1% (2/3200)

## medxpertqa

### medxpertqa — Professor hint

| Condition | Median [Q1–Q3] | p90 | n runs (inst) |
|---|---|---|---|
| Hinted (X=1, all) | 1468 [1022–2090] | 2787 | 1600 (200) |
| Unhinted (X=0, all) | 1408 [1002–1980] | 2730 | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 1356 [984–1930] | 2642 | 626 (118) |
| Hinted, not followed (X=1, Y=0) | 1520 [1060–2174] | 3006 | 974 (158) |

**Hint effect (Hinted − Unhinted):** Δ median = +60 tok · median ratio = 1.04× · GM ratio (paired) = 1.02×

**Truncation (both arms):** 0.0% (1/3200)

### medxpertqa — Grader-hacking hint

| Condition | Median [Q1–Q3] | p90 | n runs (inst) |
|---|---|---|---|
| Hinted (X=1, all) | 1562 [1038–2169] | 2974 | 1600 (200) |
| Unhinted (X=0, all) | 1408 [1002–1980] | 2730 | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 1710 [1091–2312] | 3081 | 567 (118) |
| Hinted, not followed (X=1, Y=0) | 1509 [1003–2067] | 2942 | 1033 (165) |

**Hint effect (Hinted − Unhinted):** Δ median = +154 tok · median ratio = 1.11× · GM ratio (paired) = 1.07×

**Truncation (both arms):** 0.0% (0/3200)

### medxpertqa — Unethical hint

| Condition | Median [Q1–Q3] | p90 | n runs (inst) |
|---|---|---|---|
| Hinted (X=1, all) | 1347 [933–1834] | 2487 | 1600 (200) |
| Unhinted (X=0, all) | 1408 [1002–1980] | 2730 | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 1363 [963–1848] | 2504 | 1473 (198) |
| Hinted, not followed (X=1, Y=0) | 1004 [753–1740] | 2404 | 127 (41) |

**Hint effect (Hinted − Unhinted):** Δ median = -61 tok · median ratio = 0.96× · GM ratio (paired) = 0.95×

**Truncation (both arms):** 0.0% (1/3200)

## lexam

### lexam — Professor hint

| Condition | Median [Q1–Q3] | p90 | n runs (inst) |
|---|---|---|---|
| Hinted (X=1, all) | 1880 [1132–2894] | 3662 | 1600 (200) |
| Unhinted (X=0, all) | 1676 [876–3050] | 3848 | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 2314 [1423–3286] | 3918 | 564 (123) |
| Hinted, not followed (X=1, Y=0) | 1704 [1013–2618] | 3434 | 1036 (177) |

**Hint effect (Hinted − Unhinted):** Δ median = +204 tok · median ratio = 1.12× · GM ratio (paired) = 1.09×

**Truncation (both arms):** 0.2% (7/3200)

### lexam — Grader-hacking hint

| Condition | Median [Q1–Q3] | p90 | n runs (inst) |
|---|---|---|---|
| Hinted (X=1, all) | 2238 [1148–3359] | 4072 | 1600 (200) |
| Unhinted (X=0, all) | 1676 [876–3050] | 3848 | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 3028 [1778–3678] | 4240 | 602 (126) |
| Hinted, not followed (X=1, Y=0) | 1750 [932–2906] | 3751 | 998 (172) |

**Hint effect (Hinted − Unhinted):** Δ median = +562 tok · median ratio = 1.34× · GM ratio (paired) = 1.18×

**Truncation (both arms):** 0.2% (8/3200)

### lexam — Unethical hint

| Condition | Median [Q1–Q3] | p90 | n runs (inst) |
|---|---|---|---|
| Hinted (X=1, all) | 1832 [1194–2457] | 3096 | 1600 (200) |
| Unhinted (X=0, all) | 1676 [876–3050] | 3848 | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 1822 [1207–2453] | 3084 | 1385 (195) |
| Hinted, not followed (X=1, Y=0) | 1883 [1019–2531] | 3138 | 215 (76) |

**Hint effect (Hinted − Unhinted):** Δ median = +156 tok · median ratio = 1.09× · GM ratio (paired) = 1.03×

**Truncation (both arms):** 0.2% (8/3200)

## codemmlu

### codemmlu — Professor hint

| Condition | Median [Q1–Q3] | p90 | n runs (inst) |
|---|---|---|---|
| Hinted (X=1, all) | 1456 [614–3562] | 5732 | 1600 (200) |
| Unhinted (X=0, all) | 1164 [541–2780] | 5465 | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 2889 [1163–4937] | 6867 | 335 (74) |
| Hinted, not followed (X=1, Y=0) | 1219 [532–2992] | 5237 | 1263 (181) |

**Hint effect (Hinted − Unhinted):** Δ median = +292 tok · median ratio = 1.25× · GM ratio (paired) = 1.17×

**Truncation (both arms):** 0.9% (29/3200)

### codemmlu — Grader-hacking hint

| Condition | Median [Q1–Q3] | p90 | n runs (inst) |
|---|---|---|---|
| Hinted (X=1, all) | 1616 [630–3686] | 6486 | 1600 (200) |
| Unhinted (X=0, all) | 1164 [541–2780] | 5465 | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 2836 [1241–5943] | 8685 | 298 (75) |
| Hinted, not followed (X=1, Y=0) | 1404 [561–3252] | 5680 | 1302 (184) |

**Hint effect (Hinted − Unhinted):** Δ median = +452 tok · median ratio = 1.39× · GM ratio (paired) = 1.23×

**Truncation (both arms):** 1.1% (36/3200)

### codemmlu — Unethical hint

| Condition | Median [Q1–Q3] | p90 | n runs (inst) |
|---|---|---|---|
| Hinted (X=1, all) | 2176 [945–4501] | 6510 | 1600 (200) |
| Unhinted (X=0, all) | 1164 [541–2780] | 5465 | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 2807 [1264–4903] | 6761 | 1076 (180) |
| Hinted, not followed (X=1, Y=0) | 1056 [466–2690] | 5446 | 524 (120) |

**Hint effect (Hinted − Unhinted):** Δ median = +1011 tok · median ratio = 1.87× · GM ratio (paired) = 1.57×

**Truncation (both arms):** 0.8% (26/3200)

## wmdp

### wmdp — Professor hint

| Condition | Median [Q1–Q3] | p90 | n runs (inst) |
|---|---|---|---|
| Hinted (X=1, all) | 938 [633–1643] | 8179 | 1600 (200) |
| Unhinted (X=0, all) | 838 [601–1405] | 8051 | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 1054 [718–1733] | 9812 | 546 (104) |
| Hinted, not followed (X=1, Y=0) | 868 [590–1584] | 7262 | 1054 (162) |

**Hint effect (Hinted − Unhinted):** Δ median = +100 tok · median ratio = 1.12× · GM ratio (paired) = 1.08×

**Truncation (both arms):** 4.9% (157/3200)

### wmdp — Grader-hacking hint

| Condition | Median [Q1–Q3] | p90 | n runs (inst) |
|---|---|---|---|
| Hinted (X=1, all) | 962 [621–1852] | 8046 | 1600 (200) |
| Unhinted (X=0, all) | 838 [601–1405] | 8051 | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 1216 [768–2091] | 9753 | 515 (103) |
| Hinted, not followed (X=1, Y=0) | 869 [584–1720] | 7605 | 1085 (171) |

**Hint effect (Hinted − Unhinted):** Δ median = +124 tok · median ratio = 1.15× · GM ratio (paired) = 1.11×

**Truncation (both arms):** 5.0% (161/3200)

### wmdp — Unethical hint

| Condition | Median [Q1–Q3] | p90 | n runs (inst) |
|---|---|---|---|
| Hinted (X=1, all) | 933 [643–1547] | 7461 | 1600 (200) |
| Unhinted (X=0, all) | 838 [601–1405] | 8051 | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 1034 [732–1626] | 7464 | 1196 (183) |
| Hinted, not followed (X=1, Y=0) | 630 [513–1106] | 5772 | 404 (100) |

**Hint effect (Hinted − Unhinted):** Δ median = +95 tok · median ratio = 1.11× · GM ratio (paired) = 1.06×

**Truncation (both arms):** 4.5% (145/3200)

## Paired GM ratio (Hinted / Unhinted) — dataset × hint type

Paired Hinted-vs-Unhinted GM ratio for each (dataset, hint type). Last column = geometric mean of the 3 hint-type ratios for that dataset; last row = geometric mean of the 8 dataset ratios for that hint type; bottom-right = grand GM of all cells. Geometric (not arithmetic) means, so the margins equal the GM ratio reported by the dataset-level and hint-type-level tables' pooled sections.

| Dataset | Professor | Grader-hacking | Unethical | Avg (dataset) |
|---|---|---|---|---|
| mmlu_pro | 1.21× | 1.32× | 1.53× | 1.35× |
| gpqa_diamond | 1.10× | 1.14× | 1.11× | 1.12× |
| lsat_ar | 1.16× | 1.30× | 1.50× | 1.31× |
| lsat_lr | 1.12× | 1.19× | 1.35× | 1.22× |
| medxpertqa | 1.02× | 1.07× | 0.95× | 1.01× |
| lexam | 1.09× | 1.18× | 1.03× | 1.10× |
| codemmlu | 1.17× | 1.23× | 1.57× | 1.31× |
| wmdp | 1.08× | 1.11× | 1.06× | 1.08× |
| **Avg (hint type)** | 1.12× | 1.19× | 1.24× | 1.18× |

## Followed vs not-followed (observational)

Among hinted responses, those that followed the hint were typically ~1.3× longer than those that didn't (median ratio 1.26×, longer in 22/24 slices), though this is an association conditioned on the outcome and varies widely by dataset (≈2× in codemmlu/LSAT, negligible in mmlu_pro/medxpertqa).
