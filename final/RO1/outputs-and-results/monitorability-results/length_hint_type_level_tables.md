# Response length — per hint type (datasets pooled)

Model: `Qwen/Qwen3-4B`

Columns: **Median [Q1–Q3]** primary central tendency + spread (skew-robust) · **p90** upper tail · **Mean (SD)** secondary (comparability/cost; the mean−median gap flags skew) · **n runs** valid runs behind the cell. Distributional stats use VALID runs only (not truncated, answer parsed); **Truncation** is reported separately as one rate over every response in both arms. **Hint effect** is the Hinted-vs-Unhinted median contrast (Δ median tokens and H/U median ratio) — the robust analogue of a mean gap.

*Datasets pooled; each dataset's unhinted arm is distinct, so no double counting. Note the pool mixes very different length regimes (e.g. lsat_ar ~5.5k vs medxpertqa ~1.6k median tokens) — read as a summary, per-dataset tables are the meaningful unit.*

## Professor hint

| Condition | Median [Q1–Q3] | p90 | Mean (SD) | n runs |
|---|---|---|---|---|
| Hinted (X=1, all) | 1818 [906–3810] | 6971 | 2897 (2830) | 12665 |
| Unhinted (X=0, all) | 1562 [803–3436] | 6417 | 2647 (2754) | 12647 |
| Hinted, followed (X=1, Y=1) | 2232 [1098–4876] | 8498 | 3551 (3337) | 3654 |
| Hinted, not followed (X=1, Y=0) | 1691 [833–3448] | 6270 | 2632 (2549) | 9011 |

**Hint effect (Hinted − Unhinted):** Δ median = +256 tok · ratio = 1.16×

**Truncation (both arms):** 1.0% (253/25568)

## Grader-hacking hint

| Condition | Median [Q1–Q3] | p90 | Mean (SD) | n runs |
|---|---|---|---|---|
| Hinted (X=1, all) | 2025 [912–4172] | 7456 | 3104 (2968) | 12634 |
| Unhinted (X=0, all) | 1562 [803–3436] | 6417 | 2647 (2754) | 12647 |
| Hinted, followed (X=1, Y=1) | 2786 [1353–5305] | 9043 | 3887 (3347) | 3590 |
| Hinted, not followed (X=1, Y=0) | 1768 [807–3770] | 6731 | 2793 (2742) | 9044 |

**Hint effect (Hinted − Unhinted):** Δ median = +463 tok · ratio = 1.30×

**Truncation (both arms):** 1.1% (283/25568)

## Unethical hint

| Condition | Median [Q1–Q3] | p90 | Mean (SD) | n runs |
|---|---|---|---|---|
| Hinted (X=1, all) | 1996 [996–4462] | 7695 | 3184 (2980) | 12659 |
| Unhinted (X=0, all) | 1562 [803–3436] | 6417 | 2647 (2754) | 12647 |
| Hinted, followed (X=1, Y=1) | 2026 [1115–4478] | 7860 | 3264 (3004) | 9150 |
| Hinted, not followed (X=1, Y=0) | 1871 [695–4442] | 7248 | 2976 (2905) | 3509 |

**Hint effect (Hinted − Unhinted):** Δ median = +434 tok · ratio = 1.28×

**Truncation (both arms):** 1.0% (254/25568)
