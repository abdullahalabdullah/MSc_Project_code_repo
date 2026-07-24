# Response length — per (dataset, hint type), no pooling

Model: `Qwen/Qwen3-4B`

Columns: **Median [Q1–Q3]** primary central tendency + spread (skew-robust) · **p90** upper tail · **Mean (SD)** secondary (comparability/cost; the mean−median gap flags skew) · **n runs** valid runs behind the cell. Distributional stats use VALID runs only (not truncated, answer parsed); **Truncation** is reported separately as one rate over every response in both arms. **Hint effect** is the Hinted-vs-Unhinted median contrast (Δ median tokens and H/U median ratio) — the robust analogue of a mean gap.

## mmlu_pro

### mmlu_pro — Professor hint

| Condition | Median [Q1–Q3] | p90 | Mean (SD) | n runs (inst) |
|---|---|---|---|---|
| Hinted (X=1, all) | 1401 [766–2796] | 6396 | 2425 (2633) | 1599 (200) |
| Unhinted (X=0, all) | 1049 [682–2166] | 5059 | 2034 (2444) | 1594 (200) |
| Hinted, followed (X=1, Y=1) | 1296 [761–3243] | 7854 | 2757 (3118) | 341 (70) |
| Hinted, not followed (X=1, Y=0) | 1417 [771–2747] | 5649 | 2335 (2478) | 1258 (180) |

**Hint effect (Hinted − Unhinted):** Δ median = +352 tok · ratio = 1.34×

**Truncation (both arms):** 0.2% (7/3200)

### mmlu_pro — Grader-hacking hint

| Condition | Median [Q1–Q3] | p90 | Mean (SD) | n runs (inst) |
|---|---|---|---|---|
| Hinted (X=1, all) | 1596 [738–3508] | 6898 | 2729 (2869) | 1595 (200) |
| Unhinted (X=0, all) | 1049 [682–2166] | 5059 | 2034 (2444) | 1594 (200) |
| Hinted, followed (X=1, Y=1) | 2314 [1049–5332] | 8419 | 3590 (3212) | 390 (93) |
| Hinted, not followed (X=1, Y=0) | 1368 [688–3007] | 6310 | 2450 (2691) | 1205 (185) |

**Hint effect (Hinted − Unhinted):** Δ median = +547 tok · ratio = 1.52×

**Truncation (both arms):** 0.3% (11/3200)

### mmlu_pro — Unethical hint

| Condition | Median [Q1–Q3] | p90 | Mean (SD) | n runs (inst) |
|---|---|---|---|---|
| Hinted (X=1, all) | 1702 [951–4322] | 7287 | 3028 (2951) | 1595 (200) |
| Unhinted (X=0, all) | 1049 [682–2166] | 5059 | 2034 (2444) | 1594 (200) |
| Hinted, followed (X=1, Y=1) | 1674 [975–4179] | 7096 | 2953 (2853) | 1330 (199) |
| Hinted, not followed (X=1, Y=0) | 1987 [800–5343] | 8057 | 3408 (3384) | 265 (88) |

**Hint effect (Hinted − Unhinted):** Δ median = +653 tok · ratio = 1.62×

**Truncation (both arms):** 0.3% (11/3200)

## gpqa_diamond

### gpqa_diamond — Professor hint

| Condition | Median [Q1–Q3] | p90 | Mean (SD) | n runs (inst) |
|---|---|---|---|---|
| Hinted (X=1, all) | 4686 [2682–6746] | 9059 | 5058 (3084) | 1571 (198) |
| Unhinted (X=0, all) | 4299 [2164–6686] | 9174 | 4806 (3174) | 1566 (198) |
| Hinted, followed (X=1, Y=1) | 5161 [3325–7048] | 10045 | 5569 (3220) | 762 (131) |
| Hinted, not followed (X=1, Y=0) | 4115 [2271–6400] | 8485 | 4577 (2869) | 809 (145) |

**Hint effect (Hinted − Unhinted):** Δ median = +387 tok · ratio = 1.09×

**Truncation (both arms):** 1.0% (31/3168)

### gpqa_diamond — Grader-hacking hint

| Condition | Median [Q1–Q3] | p90 | Mean (SD) | n runs (inst) |
|---|---|---|---|---|
| Hinted (X=1, all) | 4809 [2706–7064] | 9537 | 5215 (3157) | 1567 (198) |
| Unhinted (X=0, all) | 4299 [2164–6686] | 9174 | 4806 (3174) | 1566 (198) |
| Hinted, followed (X=1, Y=1) | 5298 [3433–7458] | 9965 | 5758 (3191) | 690 (135) |
| Hinted, not followed (X=1, Y=0) | 4189 [2306–6758] | 9079 | 4788 (3064) | 877 (153) |

**Hint effect (Hinted − Unhinted):** Δ median = +510 tok · ratio = 1.12×

**Truncation (both arms):** 1.1% (35/3168)

### gpqa_diamond — Unethical hint

| Condition | Median [Q1–Q3] | p90 | Mean (SD) | n runs (inst) |
|---|---|---|---|---|
| Hinted (X=1, all) | 4702 [2647–7068] | 9708 | 5168 (3194) | 1570 (198) |
| Unhinted (X=0, all) | 4299 [2164–6686] | 9174 | 4806 (3174) | 1566 (198) |
| Hinted, followed (X=1, Y=1) | 4846 [2928–7143] | 9718 | 5295 (3154) | 1256 (186) |
| Hinted, not followed (X=1, Y=0) | 3863 [1845–6763] | 9702 | 4663 (3305) | 314 (79) |

**Hint effect (Hinted − Unhinted):** Δ median = +404 tok · ratio = 1.09×

**Truncation (both arms):** 1.0% (32/3168)

## lsat_ar

### lsat_ar — Professor hint

| Condition | Median [Q1–Q3] | p90 | Mean (SD) | n runs (inst) |
|---|---|---|---|---|
| Hinted (X=1, all) | 5438 [3399–8328] | 11004 | 6106 (3341) | 1589 (200) |
| Unhinted (X=0, all) | 4704 [2836–7421] | 10816 | 5540 (3419) | 1592 (200) |
| Hinted, followed (X=1, Y=1) | 9078 [6765–11268] | 13376 | 9030 (3245) | 306 (85) |
| Hinted, not followed (X=1, Y=0) | 4661 [3070–7320] | 9849 | 5409 (2966) | 1283 (190) |

**Hint effect (Hinted − Unhinted):** Δ median = +734 tok · ratio = 1.16×

**Truncation (both arms):** 0.6% (19/3200)

### lsat_ar — Grader-hacking hint

| Condition | Median [Q1–Q3] | p90 | Mean (SD) | n runs (inst) |
|---|---|---|---|---|
| Hinted (X=1, all) | 6060 [3968–8838] | 11599 | 6654 (3353) | 1577 (200) |
| Unhinted (X=0, all) | 4704 [2836–7421] | 10816 | 5540 (3419) | 1592 (200) |
| Hinted, followed (X=1, Y=1) | 9163 [7046–11576] | 13446 | 9248 (3145) | 329 (94) |
| Hinted, not followed (X=1, Y=0) | 5412 [3564–7837] | 10414 | 5971 (3061) | 1248 (194) |

**Hint effect (Hinted − Unhinted):** Δ median = +1356 tok · ratio = 1.29×

**Truncation (both arms):** 1.0% (31/3200)

### lsat_ar — Unethical hint

| Condition | Median [Q1–Q3] | p90 | Mean (SD) | n runs (inst) |
|---|---|---|---|---|
| Hinted (X=1, all) | 6948 [4854–9172] | 11181 | 7195 (2833) | 1579 (200) |
| Unhinted (X=0, all) | 4704 [2836–7421] | 10816 | 5540 (3419) | 1592 (200) |
| Hinted, followed (X=1, Y=1) | 8050 [6236–9980] | 11745 | 8097 (2700) | 836 (170) |
| Hinted, not followed (X=1, Y=0) | 5574 [4234–7804] | 9906 | 6181 (2629) | 743 (158) |

**Hint effect (Hinted − Unhinted):** Δ median = +2244 tok · ratio = 1.48×

**Truncation (both arms):** 0.9% (29/3200)

## lsat_lr

### lsat_lr — Professor hint

| Condition | Median [Q1–Q3] | p90 | Mean (SD) | n runs (inst) |
|---|---|---|---|---|
| Hinted (X=1, all) | 1054 [700–2357] | 3862 | 1702 (1448) | 1598 (200) |
| Unhinted (X=0, all) | 959 [647–1800] | 3474 | 1481 (1225) | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 1882 [1115–3452] | 4587 | 2416 (1576) | 200 (49) |
| Hinted, not followed (X=1, Y=0) | 966 [669–2078] | 3699 | 1600 (1400) | 1398 (193) |

**Hint effect (Hinted − Unhinted):** Δ median = +94 tok · ratio = 1.10×

**Truncation (both arms):** 0.1% (2/3200)

### lsat_lr — Grader-hacking hint

| Condition | Median [Q1–Q3] | p90 | Mean (SD) | n runs (inst) |
|---|---|---|---|---|
| Hinted (X=1, all) | 1100 [678–2690] | 4377 | 1869 (1630) | 1599 (200) |
| Unhinted (X=0, all) | 959 [647–1800] | 3474 | 1481 (1225) | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 2367 [1409–4015] | 5621 | 2823 (1812) | 225 (56) |
| Hinted, not followed (X=1, Y=0) | 972 [647–2393] | 3982 | 1713 (1544) | 1374 (189) |

**Hint effect (Hinted − Unhinted):** Δ median = +141 tok · ratio = 1.15×

**Truncation (both arms):** 0.0% (1/3200)

### lsat_lr — Unethical hint

| Condition | Median [Q1–Q3] | p90 | Mean (SD) | n runs (inst) |
|---|---|---|---|---|
| Hinted (X=1, all) | 1465 [740–3056] | 4132 | 2035 (1539) | 1598 (200) |
| Unhinted (X=0, all) | 959 [647–1800] | 3474 | 1481 (1225) | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 2539 [1303–3588] | 4507 | 2653 (1568) | 637 (124) |
| Hinted, not followed (X=1, Y=0) | 946 [632–2382] | 3708 | 1626 (1376) | 961 (158) |

**Hint effect (Hinted − Unhinted):** Δ median = +506 tok · ratio = 1.53×

**Truncation (both arms):** 0.1% (2/3200)

## medxpertqa

### medxpertqa — Professor hint

| Condition | Median [Q1–Q3] | p90 | Mean (SD) | n runs (inst) |
|---|---|---|---|---|
| Hinted (X=1, all) | 1467 [1022–2088] | 2785 | 1690 (985) | 1599 (200) |
| Unhinted (X=0, all) | 1408 [1002–1980] | 2730 | 1627 (911) | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 1356 [984–1930] | 2642 | 1557 (844) | 626 (118) |
| Hinted, not followed (X=1, Y=0) | 1519 [1060–2172] | 3003 | 1776 (1057) | 973 (158) |

**Hint effect (Hinted − Unhinted):** Δ median = +59 tok · ratio = 1.04×

**Truncation (both arms):** 0.0% (1/3200)

### medxpertqa — Grader-hacking hint

| Condition | Median [Q1–Q3] | p90 | Mean (SD) | n runs (inst) |
|---|---|---|---|---|
| Hinted (X=1, all) | 1562 [1038–2169] | 2974 | 1765 (1063) | 1600 (200) |
| Unhinted (X=0, all) | 1408 [1002–1980] | 2730 | 1627 (911) | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 1710 [1091–2312] | 3081 | 1869 (1089) | 567 (118) |
| Hinted, not followed (X=1, Y=0) | 1509 [1003–2067] | 2942 | 1709 (1044) | 1033 (165) |

**Hint effect (Hinted − Unhinted):** Δ median = +154 tok · ratio = 1.11×

**Truncation (both arms):** 0.0% (0/3200)

### medxpertqa — Unethical hint

| Condition | Median [Q1–Q3] | p90 | Mean (SD) | n runs (inst) |
|---|---|---|---|---|
| Hinted (X=1, all) | 1347 [932–1834] | 2485 | 1543 (934) | 1599 (200) |
| Unhinted (X=0, all) | 1408 [1002–1980] | 2730 | 1627 (911) | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 1363 [963–1848] | 2500 | 1565 (949) | 1472 (198) |
| Hinted, not followed (X=1, Y=0) | 1004 [753–1740] | 2404 | 1279 (697) | 127 (41) |

**Hint effect (Hinted − Unhinted):** Δ median = -61 tok · ratio = 0.96×

**Truncation (both arms):** 0.0% (1/3200)

## lexam

### lexam — Professor hint

| Condition | Median [Q1–Q3] | p90 | Mean (SD) | n runs (inst) |
|---|---|---|---|---|
| Hinted (X=1, all) | 1880 [1132–2894] | 3662 | 2065 (1105) | 1600 (200) |
| Unhinted (X=0, all) | 1671 [872–3040] | 3820 | 1995 (1265) | 1593 (200) |
| Hinted, followed (X=1, Y=1) | 2314 [1423–3286] | 3918 | 2377 (1128) | 564 (123) |
| Hinted, not followed (X=1, Y=0) | 1704 [1013–2618] | 3434 | 1895 (1055) | 1036 (177) |

**Hint effect (Hinted − Unhinted):** Δ median = +209 tok · ratio = 1.13×

**Truncation (both arms):** 0.2% (7/3200)

### lexam — Grader-hacking hint

| Condition | Median [Q1–Q3] | p90 | Mean (SD) | n runs (inst) |
|---|---|---|---|---|
| Hinted (X=1, all) | 2237 [1148–3357] | 4070 | 2305 (1266) | 1599 (200) |
| Unhinted (X=0, all) | 1671 [872–3040] | 3820 | 1995 (1265) | 1593 (200) |
| Hinted, followed (X=1, Y=1) | 3028 [1778–3678] | 4240 | 2806 (1187) | 602 (126) |
| Hinted, not followed (X=1, Y=0) | 1750 [932–2904] | 3746 | 2002 (1216) | 997 (172) |

**Hint effect (Hinted − Unhinted):** Δ median = +566 tok · ratio = 1.34×

**Truncation (both arms):** 0.2% (8/3200)

### lexam — Unethical hint

| Condition | Median [Q1–Q3] | p90 | Mean (SD) | n runs (inst) |
|---|---|---|---|---|
| Hinted (X=1, all) | 1831 [1194–2456] | 3094 | 1888 (874) | 1599 (200) |
| Unhinted (X=0, all) | 1671 [872–3040] | 3820 | 1995 (1265) | 1593 (200) |
| Hinted, followed (X=1, Y=1) | 1822 [1207–2453] | 3084 | 1894 (863) | 1385 (195) |
| Hinted, not followed (X=1, Y=0) | 1877 [1006–2502] | 3104 | 1848 (939) | 214 (76) |

**Hint effect (Hinted − Unhinted):** Δ median = +160 tok · ratio = 1.10×

**Truncation (both arms):** 0.2% (8/3200)

## codemmlu

### codemmlu — Professor hint

| Condition | Median [Q1–Q3] | p90 | Mean (SD) | n runs (inst) |
|---|---|---|---|---|
| Hinted (X=1, all) | 1431 [606–3512] | 5596 | 2329 (2217) | 1585 (200) |
| Unhinted (X=0, all) | 1150 [538–2681] | 5072 | 2047 (2235) | 1586 (200) |
| Hinted, followed (X=1, Y=1) | 2825 [1140–4841] | 6615 | 3281 (2446) | 330 (73) |
| Hinted, not followed (X=1, Y=0) | 1206 [530–2908] | 5134 | 2078 (2083) | 1255 (181) |

**Hint effect (Hinted − Unhinted):** Δ median = +280 tok · ratio = 1.24×

**Truncation (both arms):** 0.9% (29/3200)

### codemmlu — Grader-hacking hint

| Condition | Median [Q1–Q3] | p90 | Mean (SD) | n runs (inst) |
|---|---|---|---|---|
| Hinted (X=1, all) | 1590 [628–3596] | 6114 | 2492 (2457) | 1578 (200) |
| Unhinted (X=0, all) | 1150 [538–2681] | 5072 | 2047 (2235) | 1586 (200) |
| Hinted, followed (X=1, Y=1) | 2608 [1219–5556] | 7868 | 3598 (2903) | 292 (72) |
| Hinted, not followed (X=1, Y=0) | 1360 [557–3124] | 5454 | 2242 (2272) | 1286 (182) |

**Hint effect (Hinted − Unhinted):** Δ median = +439 tok · ratio = 1.38×

**Truncation (both arms):** 1.1% (36/3200)

### codemmlu — Unethical hint

| Condition | Median [Q1–Q3] | p90 | Mean (SD) | n runs (inst) |
|---|---|---|---|---|
| Hinted (X=1, all) | 2139 [937–4452] | 6393 | 2879 (2370) | 1588 (200) |
| Unhinted (X=0, all) | 1150 [538–2681] | 5072 | 2047 (2235) | 1586 (200) |
| Hinted, followed (X=1, Y=1) | 2794 [1263–4890] | 6723 | 3340 (2380) | 1072 (179) |
| Hinted, not followed (X=1, Y=0) | 1045 [461–2511] | 5249 | 1923 (2043) | 516 (116) |

**Hint effect (Hinted − Unhinted):** Δ median = +988 tok · ratio = 1.86×

**Truncation (both arms):** 0.8% (26/3200)

## wmdp

### wmdp — Professor hint

| Condition | Median [Q1–Q3] | p90 | Mean (SD) | n runs (inst) |
|---|---|---|---|---|
| Hinted (X=1, all) | 895 [621–1416] | 3629 | 1801 (2679) | 1524 (197) |
| Unhinted (X=0, all) | 810 [594–1248] | 3124 | 1640 (2571) | 1516 (197) |
| Hinted, followed (X=1, Y=1) | 1034 [705–1565] | 6091 | 2184 (3275) | 525 (101) |
| Hinted, not followed (X=1, Y=0) | 828 [582–1374] | 2848 | 1600 (2282) | 999 (157) |

**Hint effect (Hinted − Unhinted):** Δ median = +84 tok · ratio = 1.10×

**Truncation (both arms):** 4.9% (157/3200)

### wmdp — Grader-hacking hint

| Condition | Median [Q1–Q3] | p90 | Mean (SD) | n runs (inst) |
|---|---|---|---|---|
| Hinted (X=1, all) | 915 [609–1646] | 3636 | 1820 (2587) | 1519 (197) |
| Unhinted (X=0, all) | 810 [594–1248] | 3124 | 1640 (2571) | 1516 (197) |
| Hinted, followed (X=1, Y=1) | 1165 [760–1878] | 4851 | 2228 (3075) | 495 (98) |
| Hinted, not followed (X=1, Y=0) | 834 [572–1511] | 2987 | 1624 (2290) | 1024 (167) |

**Hint effect (Hinted − Unhinted):** Δ median = +104 tok · ratio = 1.13×

**Truncation (both arms):** 5.0% (161/3200)

### wmdp — Unethical hint

| Condition | Median [Q1–Q3] | p90 | Mean (SD) | n runs (inst) |
|---|---|---|---|---|
| Hinted (X=1, all) | 898 [634–1400] | 3507 | 1757 (2623) | 1531 (197) |
| Unhinted (X=0, all) | 810 [594–1248] | 3124 | 1640 (2571) | 1516 (197) |
| Hinted, followed (X=1, Y=1) | 1016 [725–1542] | 5020 | 1997 (2848) | 1162 (181) |
| Hinted, not followed (X=1, Y=0) | 615 [505–854] | 1649 | 1002 (1508) | 369 (89) |

**Hint effect (Hinted − Unhinted):** Δ median = +88 tok · ratio = 1.11×

**Truncation (both arms):** 4.5% (145/3200)
