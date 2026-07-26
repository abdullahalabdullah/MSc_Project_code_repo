# Response length — median vs mean (SD), per (dataset, hint type)

Model: `Qwen/Qwen3-4B`

This companion to `length_dataset_hint_level_tables.md` keeps the arithmetic **Mean (SD)** the main tables omit, beside the **Median [Q1–Q3]**, so the skew is explicit. **Mean÷Median** > 1 quantifies right-skew — the mean inflated by the heavy upper tail (and note SD ≈ mean, so a literal 'mean ± SD' would imply impossible negative lengths, which is exactly why the main tables lead with the median). Same truncated-at-cap pools as the main tables.

## mmlu_pro

### mmlu_pro — Professor hint

| Condition | Median [Q1–Q3] | Mean (SD) | Mean÷Median | n runs (inst) |
|---|---|---|---|---|
| Hinted (X=1, all) | 1402 [766–2799] | 2434 (2655) | 1.74× | 1600 (200) |
| Unhinted (X=0, all) | 1051 [684–2183] | 2087 (2592) | 1.99× | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 1302 [762–3262] | 2796 (3200) | 2.15× | 342 (70) |
| Hinted, not followed (X=1, Y=0) | 1417 [771–2747] | 2335 (2478) | 1.65× | 1258 (180) |

### mmlu_pro — Grader-hacking hint

| Condition | Median [Q1–Q3] | Mean (SD) | Mean÷Median | n runs (inst) |
|---|---|---|---|---|
| Hinted (X=1, all) | 1602 [740–3532] | 2771 (2964) | 1.73× | 1600 (200) |
| Unhinted (X=0, all) | 1051 [684–2183] | 2087 (2592) | 1.99× | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 2344 [1050–5352] | 3688 (3389) | 1.57× | 393 (93) |
| Hinted, not followed (X=1, Y=0) | 1368 [688–3034] | 2473 (2748) | 1.81× | 1207 (185) |

### mmlu_pro — Unethical hint

| Condition | Median [Q1–Q3] | Mean (SD) | Mean÷Median | n runs (inst) |
|---|---|---|---|---|
| Hinted (X=1, all) | 1709 [955–4352] | 3070 (3040) | 1.80× | 1600 (200) |
| Unhinted (X=0, all) | 1051 [684–2183] | 2087 (2592) | 1.99× | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 1676 [976–4206] | 2993 (2942) | 1.79× | 1334 (199) |
| Hinted, not followed (X=1, Y=0) | 1997 [802–5360] | 3457 (3470) | 1.73× | 266 (88) |

## gpqa_diamond

### gpqa_diamond — Professor hint

| Condition | Median [Q1–Q3] | Mean (SD) | Mean÷Median | n runs (inst) |
|---|---|---|---|---|
| Hinted (X=1, all) | 4712 [2694–6792] | 5151 (3237) | 1.09× | 1584 (198) |
| Unhinted (X=0, all) | 4372 [2182–6784] | 4938 (3387) | 1.13× | 1584 (198) |
| Hinted, followed (X=1, Y=1) | 5197 [3327–7091] | 5625 (3305) | 1.08× | 766 (131) |
| Hinted, not followed (X=1, Y=0) | 4198 [2287–6501] | 4707 (3108) | 1.12× | 818 (147) |

### gpqa_diamond — Grader-hacking hint

| Condition | Median [Q1–Q3] | Mean (SD) | Mean÷Median | n runs (inst) |
|---|---|---|---|---|
| Hinted (X=1, all) | 4848 [2733–7182] | 5335 (3344) | 1.10× | 1584 (198) |
| Unhinted (X=0, all) | 4372 [2182–6784] | 4938 (3387) | 1.13× | 1584 (198) |
| Hinted, followed (X=1, Y=1) | 5319 [3470–7566] | 5834 (3304) | 1.10× | 695 (136) |
| Hinted, not followed (X=1, Y=0) | 4276 [2320–6893] | 4945 (3325) | 1.16× | 889 (155) |

### gpqa_diamond — Unethical hint

| Condition | Median [Q1–Q3] | Mean (SD) | Mean÷Median | n runs (inst) |
|---|---|---|---|---|
| Hinted (X=1, all) | 4728 [2676–7181] | 5267 (3348) | 1.11× | 1584 (198) |
| Unhinted (X=0, all) | 4372 [2182–6784] | 4938 (3387) | 1.13× | 1584 (198) |
| Hinted, followed (X=1, Y=1) | 4882 [2952–7190] | 5382 (3292) | 1.10× | 1266 (186) |
| Hinted, not followed (X=1, Y=0) | 3949 [1868–6898] | 4810 (3535) | 1.22× | 318 (80) |

## lsat_ar

### lsat_ar — Professor hint

| Condition | Median [Q1–Q3] | Mean (SD) | Mean÷Median | n runs (inst) |
|---|---|---|---|---|
| Hinted (X=1, all) | 5463 [3417–8444] | 6177 (3436) | 1.13× | 1600 (200) |
| Unhinted (X=0, all) | 4717 [2838–7493] | 5595 (3495) | 1.19× | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 9156 [6777–11295] | 9102 (3309) | 0.99× | 309 (87) |
| Hinted, not followed (X=1, Y=0) | 4708 [3094–7407] | 5477 (3080) | 1.16× | 1291 (192) |

### lsat_ar — Grader-hacking hint

| Condition | Median [Q1–Q3] | Mean (SD) | Mean÷Median | n runs (inst) |
|---|---|---|---|---|
| Hinted (X=1, all) | 6137 [3982–9026] | 6794 (3525) | 1.11× | 1600 (200) |
| Unhinted (X=0, all) | 4717 [2838–7493] | 5595 (3495) | 1.19× | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 9399 [7120–11834] | 9539 (3389) | 1.01× | 343 (97) |
| Hinted, not followed (X=1, Y=0) | 5434 [3573–7871] | 6045 (3174) | 1.11× | 1257 (194) |

### lsat_ar — Unethical hint

| Condition | Median [Q1–Q3] | Mean (SD) | Mean÷Median | n runs (inst) |
|---|---|---|---|---|
| Hinted (X=1, all) | 6986 [4881–9262] | 7316 (3002) | 1.05× | 1600 (200) |
| Unhinted (X=0, all) | 4717 [2838–7493] | 5595 (3495) | 1.19× | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 8116 [6249–10046] | 8176 (2805) | 1.01× | 844 (170) |
| Hinted, not followed (X=1, Y=0) | 5645 [4246–7979] | 6356 (2925) | 1.13× | 756 (159) |

## lsat_lr

### lsat_lr — Professor hint

| Condition | Median [Q1–Q3] | Mean (SD) | Mean÷Median | n runs (inst) |
|---|---|---|---|---|
| Hinted (X=1, all) | 1055 [700–2366] | 1720 (1537) | 1.63× | 1600 (200) |
| Unhinted (X=0, all) | 959 [647–1800] | 1481 (1225) | 1.54× | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 1882 [1115–3452] | 2416 (1576) | 1.28× | 200 (49) |
| Hinted, not followed (X=1, Y=0) | 966 [669–2085] | 1621 (1506) | 1.68× | 1400 (193) |

### lsat_lr — Grader-hacking hint

| Condition | Median [Q1–Q3] | Mean (SD) | Mean÷Median | n runs (inst) |
|---|---|---|---|---|
| Hinted (X=1, all) | 1101 [678–2692] | 1878 (1670) | 1.71× | 1600 (200) |
| Unhinted (X=0, all) | 959 [647–1800] | 1481 (1225) | 1.54× | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 2367 [1409–4015] | 2823 (1812) | 1.19× | 225 (56) |
| Hinted, not followed (X=1, Y=0) | 973 [647–2398] | 1723 (1593) | 1.77× | 1375 (189) |

### lsat_lr — Unethical hint

| Condition | Median [Q1–Q3] | Mean (SD) | Mean÷Median | n runs (inst) |
|---|---|---|---|---|
| Hinted (X=1, all) | 1475 [740–3063] | 2053 (1620) | 1.39× | 1600 (200) |
| Unhinted (X=0, all) | 959 [647–1800] | 1481 (1225) | 1.54× | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 2542 [1304–3589] | 2674 (1658) | 1.05× | 638 (124) |
| Hinted, not followed (X=1, Y=0) | 946 [632–2386] | 1641 (1455) | 1.73× | 962 (159) |

## medxpertqa

### medxpertqa — Professor hint

| Condition | Median [Q1–Q3] | Mean (SD) | Mean÷Median | n runs (inst) |
|---|---|---|---|---|
| Hinted (X=1, all) | 1468 [1022–2090] | 1699 (1051) | 1.16× | 1600 (200) |
| Unhinted (X=0, all) | 1408 [1002–1980] | 1627 (911) | 1.16× | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 1356 [984–1930] | 1557 (844) | 1.15× | 626 (118) |
| Hinted, not followed (X=1, Y=0) | 1520 [1060–2174] | 1791 (1156) | 1.18× | 974 (158) |

### medxpertqa — Grader-hacking hint

| Condition | Median [Q1–Q3] | Mean (SD) | Mean÷Median | n runs (inst) |
|---|---|---|---|---|
| Hinted (X=1, all) | 1562 [1038–2169] | 1765 (1063) | 1.13× | 1600 (200) |
| Unhinted (X=0, all) | 1408 [1002–1980] | 1627 (911) | 1.16× | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 1710 [1091–2312] | 1869 (1089) | 1.09× | 567 (118) |
| Hinted, not followed (X=1, Y=0) | 1509 [1003–2067] | 1709 (1044) | 1.13× | 1033 (165) |

### medxpertqa — Unethical hint

| Condition | Median [Q1–Q3] | Mean (SD) | Mean÷Median | n runs (inst) |
|---|---|---|---|---|
| Hinted (X=1, all) | 1347 [933–1834] | 1552 (1005) | 1.15× | 1600 (200) |
| Unhinted (X=0, all) | 1408 [1002–1980] | 1627 (911) | 1.16× | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 1363 [963–1848] | 1576 (1024) | 1.16× | 1473 (198) |
| Hinted, not followed (X=1, Y=0) | 1004 [753–1740] | 1279 (697) | 1.27× | 127 (41) |

## lexam

### lexam — Professor hint

| Condition | Median [Q1–Q3] | Mean (SD) | Mean÷Median | n runs (inst) |
|---|---|---|---|---|
| Hinted (X=1, all) | 1880 [1132–2894] | 2065 (1105) | 1.10× | 1600 (200) |
| Unhinted (X=0, all) | 1676 [876–3050] | 2058 (1580) | 1.23× | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 2314 [1423–3286] | 2377 (1128) | 1.03× | 564 (123) |
| Hinted, not followed (X=1, Y=0) | 1704 [1013–2618] | 1895 (1055) | 1.11× | 1036 (177) |

### lexam — Grader-hacking hint

| Condition | Median [Q1–Q3] | Mean (SD) | Mean÷Median | n runs (inst) |
|---|---|---|---|---|
| Hinted (X=1, all) | 2238 [1148–3359] | 2314 (1314) | 1.03× | 1600 (200) |
| Unhinted (X=0, all) | 1676 [876–3050] | 2058 (1580) | 1.23× | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 3028 [1778–3678] | 2806 (1187) | 0.93× | 602 (126) |
| Hinted, not followed (X=1, Y=0) | 1750 [932–2906] | 2017 (1298) | 1.15× | 998 (172) |

### lexam — Unethical hint

| Condition | Median [Q1–Q3] | Mean (SD) | Mean÷Median | n runs (inst) |
|---|---|---|---|---|
| Hinted (X=1, all) | 1832 [1194–2457] | 1897 (946) | 1.04× | 1600 (200) |
| Unhinted (X=0, all) | 1676 [876–3050] | 2058 (1580) | 1.23× | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 1822 [1207–2453] | 1894 (863) | 1.04× | 1385 (195) |
| Hinted, not followed (X=1, Y=0) | 1883 [1019–2531] | 1916 (1364) | 1.02× | 215 (76) |

## codemmlu

### codemmlu — Professor hint

| Condition | Median [Q1–Q3] | Mean (SD) | Mean÷Median | n runs (inst) |
|---|---|---|---|---|
| Hinted (X=1, all) | 1456 [614–3562] | 2460 (2590) | 1.69× | 1600 (200) |
| Unhinted (X=0, all) | 1164 [541–2780] | 2173 (2595) | 1.87× | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 2889 [1163–4937] | 3477 (2903) | 1.20× | 335 (74) |
| Hinted, not followed (X=1, Y=0) | 1219 [532–2992] | 2169 (2367) | 1.78× | 1263 (181) |

### codemmlu — Grader-hacking hint

| Condition | Median [Q1–Q3] | Mean (SD) | Mean÷Median | n runs (inst) |
|---|---|---|---|---|
| Hinted (X=1, all) | 1616 [630–3686] | 2684 (2928) | 1.66× | 1600 (200) |
| Unhinted (X=0, all) | 1164 [541–2780] | 2173 (2595) | 1.87× | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 2836 [1241–5943] | 3855 (3390) | 1.36× | 298 (75) |
| Hinted, not followed (X=1, Y=0) | 1404 [561–3252] | 2415 (2744) | 1.72× | 1302 (184) |

### codemmlu — Unethical hint

| Condition | Median [Q1–Q3] | Mean (SD) | Mean÷Median | n runs (inst) |
|---|---|---|---|---|
| Hinted (X=1, all) | 2176 [945–4501] | 2981 (2633) | 1.37× | 1600 (200) |
| Unhinted (X=0, all) | 1164 [541–2780] | 2173 (2595) | 1.87× | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 2807 [1264–4903] | 3388 (2504) | 1.21× | 1076 (180) |
| Hinted, not followed (X=1, Y=0) | 1056 [466–2690] | 2144 (2695) | 2.03× | 524 (120) |

## wmdp

### wmdp — Professor hint

| Condition | Median [Q1–Q3] | Mean (SD) | Mean÷Median | n runs (inst) |
|---|---|---|---|---|
| Hinted (X=1, all) | 938 [633–1643] | 2490 (4048) | 2.65× | 1600 (200) |
| Unhinted (X=0, all) | 838 [601–1405] | 2414 (4133) | 2.88× | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 1054 [718–1733] | 2730 (4217) | 2.59× | 546 (104) |
| Hinted, not followed (X=1, Y=0) | 868 [590–1584] | 2367 (3954) | 2.72× | 1054 (162) |

### wmdp — Grader-hacking hint

| Condition | Median [Q1–Q3] | Mean (SD) | Mean÷Median | n runs (inst) |
|---|---|---|---|---|
| Hinted (X=1, all) | 962 [621–1852] | 2553 (4053) | 2.65× | 1600 (200) |
| Unhinted (X=0, all) | 838 [601–1405] | 2414 (4133) | 2.88× | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 1216 [768–2091] | 2777 (4072) | 2.28× | 515 (103) |
| Hinted, not followed (X=1, Y=0) | 869 [584–1720] | 2446 (4042) | 2.81× | 1085 (171) |

### wmdp — Unethical hint

| Condition | Median [Q1–Q3] | Mean (SD) | Mean÷Median | n runs (inst) |
|---|---|---|---|---|
| Hinted (X=1, all) | 933 [643–1547] | 2364 (3862) | 2.53× | 1600 (200) |
| Unhinted (X=0, all) | 838 [601–1405] | 2414 (4133) | 2.88× | 1600 (200) |
| Hinted, followed (X=1, Y=1) | 1034 [732–1626] | 2383 (3624) | 2.30× | 1196 (183) |
| Hinted, not followed (X=1, Y=0) | 630 [513–1106] | 2308 (4498) | 3.66× | 404 (100) |
