# Response length — per dataset (hint types pooled)

Model: `Qwen/Qwen3-4B`

Columns: **Median [Q1–Q3]** primary central tendency + spread (skew-robust) · **p90** upper tail · **Mean (SD)** secondary (comparability/cost; the mean−median gap flags skew) · **n runs** valid runs behind the cell. Distributional stats use VALID runs only (not truncated, answer parsed); **Truncation** is reported separately as one rate over every response in both arms. **Hint effect** is the Hinted-vs-Unhinted median contrast (Δ median tokens and H/U median ratio) — the robust analogue of a mean gap.

*Unhinted is hint-independent, so it is taken from one hint arm (not tripled); hinted/followed/not-followed pool the 3 hint types.*

## mmlu_pro

| Condition | Median [Q1–Q3] | p90 | Mean (SD) | n runs |
|---|---|---|---|---|
| Hinted (X=1, all) | 1558 [816–3491] | 6988 | 2727 (2831) | 4789 |
| Unhinted (X=0, all) | 1049 [682–2166] | 5059 | 2034 (2444) | 1594 |
| Hinted, followed (X=1, Y=1) | 1711 [943–4351] | 7509 | 3041 (2980) | 2061 |
| Hinted, not followed (X=1, Y=0) | 1434 [730–3006] | 6368 | 2490 (2689) | 2728 |

**Hint effect (Hinted − Unhinted):** Δ median = +509 tok · ratio = 1.49×

**Truncation (both arms; unhinted counted per-hint):** 0.3% (29/9600)

## gpqa_diamond

| Condition | Median [Q1–Q3] | p90 | Mean (SD) | n runs |
|---|---|---|---|---|
| Hinted (X=1, all) | 4716 [2686–6958] | 9494 | 5147 (3145) | 4708 |
| Unhinted (X=0, all) | 4299 [2164–6686] | 9174 | 4806 (3174) | 1566 |
| Hinted, followed (X=1, Y=1) | 5044 [3146–7194] | 9853 | 5490 (3187) | 2708 |
| Hinted, not followed (X=1, Y=0) | 4136 [2218–6584] | 8946 | 4683 (3027) | 2000 |

**Hint effect (Hinted − Unhinted):** Δ median = +418 tok · ratio = 1.10×

**Truncation (both arms; unhinted counted per-hint):** 1.0% (98/9504)

## lsat_ar

| Condition | Median [Q1–Q3] | p90 | Mean (SD) | n runs |
|---|---|---|---|---|
| Hinted (X=1, all) | 6250 [4055–8842] | 11223 | 6651 (3216) | 4745 |
| Unhinted (X=0, all) | 4704 [2836–7421] | 10816 | 5540 (3419) | 1592 |
| Hinted, followed (X=1, Y=1) | 8506 [6423–10470] | 12606 | 8549 (2968) | 1471 |
| Hinted, not followed (X=1, Y=0) | 5180 [3536–7620] | 10056 | 5798 (2947) | 3274 |

**Hint effect (Hinted − Unhinted):** Δ median = +1546 tok · ratio = 1.33×

**Truncation (both arms; unhinted counted per-hint):** 0.8% (79/9600)

## lsat_lr

| Condition | Median [Q1–Q3] | p90 | Mean (SD) | n runs |
|---|---|---|---|---|
| Hinted (X=1, all) | 1172 [706–2740] | 4133 | 1869 (1547) | 4795 |
| Unhinted (X=0, all) | 959 [647–1800] | 3474 | 1481 (1225) | 1600 |
| Hinted, followed (X=1, Y=1) | 2372 [1246–3606] | 4725 | 2644 (1628) | 1062 |
| Hinted, not followed (X=1, Y=0) | 965 [651–2297] | 3834 | 1648 (1449) | 3733 |

**Hint effect (Hinted − Unhinted):** Δ median = +213 tok · ratio = 1.22×

**Truncation (both arms; unhinted counted per-hint):** 0.1% (5/9600)

## medxpertqa

| Condition | Median [Q1–Q3] | p90 | Mean (SD) | n runs |
|---|---|---|---|---|
| Hinted (X=1, all) | 1442 [995–2043] | 2778 | 1666 (999) | 4798 |
| Unhinted (X=0, all) | 1408 [1002–1980] | 2730 | 1627 (911) | 1600 |
| Hinted, followed (X=1, Y=1) | 1411 [988–1988] | 2676 | 1628 (965) | 2665 |
| Hinted, not followed (X=1, Y=0) | 1498 [1003–2098] | 2910 | 1714 (1039) | 2133 |

**Hint effect (Hinted − Unhinted):** Δ median = +34 tok · ratio = 1.02×

**Truncation (both arms; unhinted counted per-hint):** 0.0% (2/9600)

## lexam

| Condition | Median [Q1–Q3] | p90 | Mean (SD) | n runs |
|---|---|---|---|---|
| Hinted (X=1, all) | 1939 [1161–2886] | 3666 | 2086 (1107) | 4798 |
| Unhinted (X=0, all) | 1671 [872–3040] | 3820 | 1995 (1265) | 1593 |
| Hinted, followed (X=1, Y=1) | 2098 [1330–3010] | 3718 | 2216 (1078) | 2551 |
| Hinted, not followed (X=1, Y=0) | 1751 [978–2736] | 3567 | 1938 (1121) | 2247 |

**Hint effect (Hinted − Unhinted):** Δ median = +268 tok · ratio = 1.16×

**Truncation (both arms; unhinted counted per-hint):** 0.2% (23/9600)

## codemmlu

| Condition | Median [Q1–Q3] | p90 | Mean (SD) | n runs |
|---|---|---|---|---|
| Hinted (X=1, all) | 1706 [697–3828] | 6062 | 2567 (2361) | 4751 |
| Unhinted (X=0, all) | 1150 [538–2681] | 5072 | 2047 (2235) | 1586 |
| Hinted, followed (X=1, Y=1) | 2788 [1224–4966] | 6870 | 3373 (2491) | 1694 |
| Hinted, not followed (X=1, Y=0) | 1223 [526–2994] | 5266 | 2121 (2161) | 3057 |

**Hint effect (Hinted − Unhinted):** Δ median = +556 tok · ratio = 1.48×

**Truncation (both arms; unhinted counted per-hint):** 0.9% (91/9600)

## wmdp

| Condition | Median [Q1–Q3] | p90 | Mean (SD) | n runs |
|---|---|---|---|---|
| Hinted (X=1, all) | 903 [621–1487] | 3588 | 1793 (2630) | 4574 |
| Unhinted (X=0, all) | 810 [594–1248] | 3124 | 1640 (2571) | 1516 |
| Hinted, followed (X=1, Y=1) | 1040 [727–1644] | 5115 | 2094 (3008) | 2182 |
| Hinted, not followed (X=1, Y=0) | 773 [563–1357] | 2675 | 1518 (2195) | 2392 |

**Hint effect (Hinted − Unhinted):** Δ median = +92 tok · ratio = 1.11×

**Truncation (both arms; unhinted counted per-hint):** 4.8% (463/9600)
