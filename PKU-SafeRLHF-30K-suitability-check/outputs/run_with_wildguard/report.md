# Harmlessness preference-pair pipeline — report

- Sampled prompts (denominator): **300**
- Responses per prompt (N): 10, max_tokens: 8192

## Filtering funnel
- filter-1 (safety-label diversity): dropped **210 / 300** (70.0%) → 90 enter scoring
- filter-3 (chosen=Safe & rejected=Unsafe): dropped **29 / 90** (32.2%)
- **End-to-end survival rate: 61 / 300 = 20.3%**

## Cost-spread distribution (rejected − chosen)
- mean: 5.151  median: 3.830
- min: 1.258  max: 13.906
- q25: 2.203  q75: 7.918
- histogram: `outputs/run_20260704_142241/cost_spread.png`

## Sign-convention sanity check
- mean Beaver cost of **Unsafe** answers: 3.4397 (n=425)
- mean Beaver cost of **Safe**   answers: 0.4631 (n=474)
- Unsafe > Safe (expected)? **PASS**

## Artifacts
- final pairs: `outputs/run_20260704_142241/final_pairs.joblib` (61 pairs)
- per-pair margins (CSV): `outputs/run_20260704_142241/final_pairs_margins.csv`
- sample pairs: `outputs/run_20260704_142241/sample_pairs.json` (10 pairs)
