# Response length — per dataset (hint types pooled)

Model: `Qwen/Qwen3-4B`

Columns: **Median [Q1–Q3]** primary central tendency + spread (skew-robust) · **p90** upper tail · **n runs (inst)** responses and distinct instances behind the cell. (Arithmetic Mean (SD) is omitted here to keep the table median-first; it lives in `length_dataset_hint_median_vs_mean.md`.) Length stats INCLUDE truncated runs at the 16384-token cap (truncation is caused by length, so dropping them would bias length down; the cap understates the true tail, so estimates are conservative). **Truncation** is the separate rate of capped responses over both arms. **Hint effect** is the Hinted-vs-Unhinted contrast: Δ median (tokens), H/U median ratio — the robust analogue of a mean gap — and the paired **GM ratio** (per-question at the dataset×hint level; GM of the constituent cell GMRs at the pooled levels). (Descriptive summaries of observed runs — no confidence intervals.)

*Unhinted is hint-independent, so it is taken from one hint arm (not tripled); hinted/followed/not-followed pool the 3 hint types.*

## mmlu_pro

| Condition | Median [Q1–Q3] | p90 | n runs (inst) |
|---|---|---|---|
| Hinted (X=1, all) | 1563 [817–3510] | 7052 | 4800 (200) |
| Unhinted (X=0, all) | 1051 [684–2183] | 5271 | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 1717 [949–4398] | 7567 | 2069 (199) |
| Hinted, not followed (X=1, Y=0) | 1435 [730–3012] | 6393 | 2731 (187) |

**Hint effect (Hinted − Unhinted):** Δ median = +512 tok · median ratio = 1.49× · GM ratio (paired, GM of the 3 per-hint cell GMRs) = 1.35×

**Truncation (both arms):** 0.3% (17/6400)

## gpqa_diamond

| Condition | Median [Q1–Q3] | p90 | n runs (inst) |
|---|---|---|---|
| Hinted (X=1, all) | 4751 [2705–7032] | 9684 | 4752 (198) |
| Unhinted (X=0, all) | 4372 [2182–6784] | 9489 | 1584 (198) |
| Hinted, followed (X=1, Y=1) | 5058 [3162–7254] | 10041 | 2727 (187) |
| Hinted, not followed (X=1, Y=0) | 4197 [2252–6742] | 9301 | 2025 (166) |

**Hint effect (Hinted − Unhinted):** Δ median = +378 tok · median ratio = 1.09× · GM ratio (paired, GM of the 3 per-hint cell GMRs) = 1.12×

**Truncation (both arms):** 1.0% (62/6336)

## lsat_ar

| Condition | Median [Q1–Q3] | p90 | n runs (inst) |
|---|---|---|---|
| Hinted (X=1, all) | 6280 [4088–8932] | 11433 | 4800 (200) |
| Unhinted (X=0, all) | 4717 [2838–7493] | 10938 | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 8568 [6450–10654] | 12871 | 1496 (176) |
| Hinted, not followed (X=1, Y=0) | 5200 [3558–7730] | 10270 | 3304 (194) |

**Hint effect (Hinted − Unhinted):** Δ median = +1564 tok · median ratio = 1.33× · GM ratio (paired, GM of the 3 per-hint cell GMRs) = 1.31×

**Truncation (both arms):** 1.0% (63/6400)

## lsat_lr

| Condition | Median [Q1–Q3] | p90 | n runs (inst) |
|---|---|---|---|
| Hinted (X=1, all) | 1174 [706–2747] | 4146 | 4800 (200) |
| Unhinted (X=0, all) | 959 [647–1800] | 3474 | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 2374 [1247–3606] | 4729 | 1063 (124) |
| Hinted, not followed (X=1, Y=0) | 966 [651–2298] | 3843 | 3737 (194) |

**Hint effect (Hinted − Unhinted):** Δ median = +215 tok · median ratio = 1.22× · GM ratio (paired, GM of the 3 per-hint cell GMRs) = 1.22×

**Truncation (both arms):** 0.1% (5/6400)

## medxpertqa

| Condition | Median [Q1–Q3] | p90 | n runs (inst) |
|---|---|---|---|
| Hinted (X=1, all) | 1442 [996–2043] | 2780 | 4800 (200) |
| Unhinted (X=0, all) | 1408 [1002–1980] | 2730 | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 1412 [988–1992] | 2680 | 2666 (198) |
| Hinted, not followed (X=1, Y=0) | 1498 [1003–2100] | 2912 | 2134 (171) |

**Hint effect (Hinted − Unhinted):** Δ median = +34 tok · median ratio = 1.02× · GM ratio (paired, GM of the 3 per-hint cell GMRs) = 1.01×

**Truncation (both arms):** 0.0% (2/6400)

## lexam

| Condition | Median [Q1–Q3] | p90 | n runs (inst) |
|---|---|---|---|
| Hinted (X=1, all) | 1940 [1162–2887] | 3670 | 4800 (200) |
| Unhinted (X=0, all) | 1676 [876–3050] | 3848 | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 2098 [1330–3010] | 3718 | 2551 (195) |
| Hinted, not followed (X=1, Y=0) | 1752 [979–2739] | 3576 | 2249 (181) |

**Hint effect (Hinted − Unhinted):** Δ median = +264 tok · median ratio = 1.16× · GM ratio (paired, GM of the 3 per-hint cell GMRs) = 1.10×

**Truncation (both arms):** 0.1% (9/6400)

## codemmlu

| Condition | Median [Q1–Q3] | p90 | n runs (inst) |
|---|---|---|---|
| Hinted (X=1, all) | 1744 [703–3900] | 6318 | 4800 (200) |
| Unhinted (X=0, all) | 1164 [541–2780] | 5465 | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 2844 [1233–5013] | 7083 | 1709 (180) |
| Hinted, not followed (X=1, Y=0) | 1248 [530–3075] | 5471 | 3089 (186) |

**Hint effect (Hinted − Unhinted):** Δ median = +579 tok · median ratio = 1.50× · GM ratio (paired, GM of the 3 per-hint cell GMRs) = 1.31×

**Truncation (both arms):** 1.0% (63/6400)

## wmdp

| Condition | Median [Q1–Q3] | p90 | n runs (inst) |
|---|---|---|---|
| Hinted (X=1, all) | 942 [632–1668] | 7989 | 4800 (200) |
| Unhinted (X=0, all) | 838 [601–1405] | 8051 | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 1062 [735–1746] | 8542 | 2257 (183) |
| Hinted, not followed (X=1, Y=0) | 820 [572–1579] | 7449 | 2543 (176) |

**Hint effect (Hinted − Unhinted):** Δ median = +104 tok · median ratio = 1.12× · GM ratio (paired, GM of the 3 per-hint cell GMRs) = 1.08×

**Truncation (both arms):** 4.6% (295/6400)
