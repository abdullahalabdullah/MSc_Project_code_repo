# Response length — per hint type (datasets pooled)

Model: `Qwen/Qwen3-4B`

Columns: **Median [Q1–Q3]** primary central tendency + spread (skew-robust) · **p90** upper tail · **n runs (inst)** responses and distinct instances behind the cell. (Arithmetic Mean (SD) is omitted here to keep the table median-first; it lives in `length_dataset_hint_median_vs_mean.md`.) Length stats INCLUDE truncated runs at the 16384-token cap (truncation is caused by length, so dropping them would bias length down; the cap understates the true tail, so estimates are conservative). **Truncation** is the separate rate of capped responses over both arms. **Hint effect** is the Hinted-vs-Unhinted contrast: Δ median (tokens), H/U median ratio — the robust analogue of a mean gap — and the paired **GM ratio** (per-question at the dataset×hint level; GM of the constituent cell GMRs at the pooled levels). (Descriptive summaries of observed runs — no confidence intervals.)

*Datasets pooled; each dataset's unhinted arm is distinct, so no double counting. Note the pool mixes very different length regimes (e.g. lsat_ar ~5.5k vs medxpertqa ~1.6k median tokens) — read as a summary, per-dataset tables are the meaningful unit.*

## Professor hint

| Condition | Median [Q1–Q3] | p90 | n runs (inst) |
|---|---|---|---|
| Hinted (X=1, all) | 1838 [912–3892] | 7280 | 12784 (1598) |
| Unhinted (X=0, all) | 1583 [808–3527] | 6786 | 12784 (1598) |
| Hinted, followed (X=1, Y=1) | 2289 [1110–4978] | 8862 | 3688 (756) |
| Hinted, not followed (X=1, Y=0) | 1718 [839–3519] | 6543 | 9094 (1390) |

**Hint effect (Hinted − Unhinted):** Δ median = +256 tok · median ratio = 1.16× · GM ratio (paired, GM of the 8 per-dataset cell GMRs) = 1.12×

**Truncation (both arms):** 1.0% (253/25568)

## Grader-hacking hint

| Condition | Median [Q1–Q3] | p90 | n runs (inst) |
|---|---|---|---|
| Hinted (X=1, all) | 2058 [920–4294] | 7834 | 12784 (1598) |
| Unhinted (X=0, all) | 1583 [808–3527] | 6786 | 12784 (1598) |
| Hinted, followed (X=1, Y=1) | 2858 [1369–5522] | 9513 | 3638 (804) |
| Hinted, not followed (X=1, Y=0) | 1804 [815–3876] | 7018 | 9146 (1415) |

**Hint effect (Hinted − Unhinted):** Δ median = +475 tok · median ratio = 1.30× · GM ratio (paired, GM of the 8 per-dataset cell GMRs) = 1.19×

**Truncation (both arms):** 1.1% (283/25568)

## Unethical hint

| Condition | Median [Q1–Q3] | p90 | n runs (inst) |
|---|---|---|---|
| Hinted (X=1, all) | 2024 [1003–4570] | 8005 | 12784 (1598) |
| Unhinted (X=0, all) | 1583 [808–3527] | 6786 | 12784 (1598) |
| Hinted, followed (X=1, Y=1) | 2044 [1120–4553] | 8098 | 9212 (1435) |
| Hinted, not followed (X=1, Y=0) | 1934 [699–4602] | 7788 | 3572 (823) |

**Hint effect (Hinted − Unhinted):** Δ median = +441 tok · median ratio = 1.28× · GM ratio (paired, GM of the 8 per-dataset cell GMRs) = 1.24×

**Truncation (both arms):** 1.0% (254/25568)
