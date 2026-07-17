# Anti-Sycophancy DPO Curation — QA Report

Generated: 2026-07-17T09:54:49  |  git: `767053411d3ac7741ceb36515ee63d4be443a8b0`  |  X=500  N=10/10  PILOT=True

## Funnel

| Stage boundary | Count |
|---|---|
| sampled_pool | 500 |
| filter0_graded | 500 |
| filter0_pass | 248 |
| instances | 496 |
| filter1_pass | 140 |
| pairs_selected | 140 |
| pairs_post_balance | 136 |

## Filter-0 consistency distribution

| Stratum | Questions |
|---|---|
| 70-80 | 16 |
| 80-90 | 15 |
| 90-99 | 21 |
| 100 | 196 |
| below_70 (fail) | 252 |

### Per-stratum Filter-1 survival

| Stratum | Instances | Pass | Survival |
|---|---|---|---|
| 70-80 | 32 | 13 | 40.6% |
| 80-90 | 30 | 10 | 33.3% |
| 90-99 | 42 | 16 | 38.1% |
| 100 | 392 | 101 | 25.8% |

## Baseline sycophancy (syc_fraction over ALL wrong-opinion instances)

- instances with a defined syc_fraction: 247 / 248
- mean 0.2  |  median 0.0
- histogram: `syc_fraction_hist.png`

## Per-template stats

| Template | Instances | Survival | Wrong-opinion rejected-class rate | Mean syc_fraction |
|---|---|---|---|---|
| T1 | 54 | 22.2% | 28.28% | 0.3 |
| T2 | 61 | 16.4% | 21.10% | 0.2 |
| T3 | 66 | 22.7% | 25.86% | 0.3 |
| T4 | 57 | 33.3% | 29.80% | 0.3 |
| T5 | 61 | 36.1% | 26.82% | 0.2 |
| T6 | 67 | 16.4% | 12.11% | 0.1 |
| T7 | 62 | 27.4% | 48.65% | 0.5 |
| T8 | 68 | 50.0% | 8.70% | 0.1 |

## Grader

- audit: **PENDING** — run `audit_grader_sample.py --make` then hand-label and `--score` (blocking pilot gate 1)
- classifier API calls: 9958 (+ 0 cache hits); wrong-answer generator unique calls: 1748
- temperature: omitted (gpt-5-mini rejects non-default; spec deviation, see manifest)

## Parse health

- UNPARSEABLE: 2 / 9960 samples (0.02%)
  - from truncation: 2
  - from missing final-answer line: 0
  - from invalid classifier output: 0

## Final dataset

- pairs: 136  |  wrong_opinion 68 / right_opinion 68
- completion length (tokens): chosen mean 590.4 / median 418.0  |  rejected mean 591.1 / median 420.0

## Pilot gates

- gate1_grader_audit: **PENDING**
- gate2_yield_projection: **PASS**
- gate3_template_health: **PASS**
- gate4_parse_health: **PASS**
