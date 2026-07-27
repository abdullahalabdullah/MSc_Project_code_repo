# Anti-Sycophancy DPO Curation — QA Report

Generated: 2026-07-16T21:38:14  |  git: `c841e085967ab7734699fd6029bdaf2332d9009b`  |  X=5  N=10/10  PILOT=False

## Funnel

| Stage boundary | Count |
|---|---|
| sampled_pool | 5 |
| filter0_graded | 5 |
| filter0_pass | 5 |
| instances | 10 |
| filter1_pass | 4 |
| pairs_selected | 4 |
| pairs_post_balance | 4 |

## Filter-0 consistency distribution

| Stratum | Questions |
|---|---|
| 70-80 | 0 |
| 80-90 | 0 |
| 90-99 | 1 |
| 100 | 4 |
| below_70 (fail) | 0 |

### Per-stratum Filter-1 survival

| Stratum | Instances | Pass | Survival |
|---|---|---|---|
| 90-99 | 2 | 2 | 100.0% |
| 100 | 8 | 2 | 25.0% |

## Baseline sycophancy (syc_fraction over ALL wrong-opinion instances)

- instances with a defined syc_fraction: 5 / 5
- mean 0.3  |  median 0.3
- histogram: `syc_fraction_hist.png`

## Per-template stats

| Template | Instances | Survival | Wrong-opinion rejected-class rate | Mean syc_fraction |
|---|---|---|---|---|
| T1 | 3 | 33.3% | 0.00% | 0.0 |
| T2 | 4 | 50.0% | 56.67% | 0.6 |
| T4 | 1 | 0.0% | 0.00% | None |
| T6 | 1 | 0.0% | 0.00% | 0.0 |
| T8 | 1 | 100.0% | 0.00% | None |

## Grader

- audit: **PENDING** — run `audit_grader_sample.py --make` then hand-label and `--score` (blocking pilot gate 1)
- classifier API calls: 150 (+ 0 cache hits); wrong-answer generator unique calls: 1505
- temperature: omitted (gpt-5-mini rejects non-default; spec deviation, see manifest)

## Parse health

- UNPARSEABLE: 0 / 150 samples (0.00%)
  - from truncation: 0
  - from missing final-answer line: 0
  - from invalid classifier output: 0

## Final dataset

- pairs: 4  |  wrong_opinion 2 / right_opinion 2
- completion length (tokens): chosen mean 855.2 / median 916.5  |  rejected mean 919.2 / median 978.5
