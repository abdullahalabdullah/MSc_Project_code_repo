# Harmlessness preference-pair pipeline — report

- Sampled prompts (denominator): **8** (SAMPLE=8)
- Responses per prompt (N): 10, max_tokens: 8192

## Filtering funnel
- filter-1 (safety-label diversity): dropped **7 / 8** (87.5%) → 1 enter scoring
- filter-3 (chosen=Safe & rejected=Unsafe): dropped **0 / 1** (0.0%) → 1 pairs
- filter-4 (minimum margin): **not applied** (MIN_MARGIN unset)
- **End-to-end survival rate: 1 / 8 = 12.5%**

## Per-subset funnel (sample mix after each filter)
| subset | sampled | after filter-1 | after filter-3 | final | survival % |
|---|---|---|---|---|---|
| adversarial | 7 | 1 | 1 | 1 | 14.3 |
| vanilla | 1 | 0 | 0 | 0 | 0.0 |
| **all** | 8 | 1 | 1 | 1 | 12.5 |

## Labeler agreement (Beaver argmin/argmax vs Qwen3Guard)
- chosen-side (Beaver-safest is Qwen3Guard-Safe): **100.0%** (1/1)
- rejected-side (Beaver-harshest is Qwen3Guard-Unsafe): **100.0%** (1/1)
- both agree (→ filter-3 survivors): **100.0%** (1/1)
- point-biserial r(cost, Unsafe) over all scored responses: **+0.657** (+1 = cost perfectly separates Qwen3Guard classes; 0 = no separation)

## Margin calibration (isolation of the chosen/rejected picks)
- **chosen runner-up gap** (2nd-safest − safest cost): mean 0.062, median 0.062 — small ⇒ the safe pick was near-tied
- **rejected runner-up gap** (harshest − 2nd-harshest cost): mean 0.250, median 0.250
- mean margin (adversarial): 1.656 (n=1)
- **Qwen3Guard decisiveness** (mean Unsafe fraction per surviving prompt): 0.60 (0.5 = maximally split)

## Think-block vs answer length (final pairs, words)
- **chosen** — think mean 646, answer mean 496 (think/answer ratio 1.3)
- **rejected** — think mean 864, answer mean 476 (think/answer ratio 1.8)
- (CoT is retained in the DPO pair but never scored/labeled — this tracks how much of each response is reasoning vs visible answer.)

## Cost-spread distribution (rejected − chosen)
- mean: 1.656  median: 1.656
- min: 1.656  max: 1.656
- q25: 1.656  q75: 1.656
- histogram: `outputs/smoke8/cost_spread.png`

## Length / verbosity confound (chosen vs rejected)
- **Full response (DPO pair content), tokens** — chosen mean 1503, rejected mean 1743, mean Δ(rej−cho) +240 (median +240); rejected longer in 100% of pairs
- **Visible answer (scored text), words** — chosen mean 496, rejected mean 476, mean Δ(rej−cho) -20 (median -20); rejected longer in 0% of pairs
- **Pearson(cost margin, answer-word Δ)**: n/a (no length variance) (|r|→1 ⇒ cost margin is confounded by length; ~0 ⇒ decoupled)
- Note: a strong one-sided skew (rejected ≫ chosen) risks DPO learning length rather than harmlessness — watch this when scaling up.

## Sign-convention sanity check
- mean Beaver cost of **Unsafe** answers: 5.1927 (n=6)
- mean Beaver cost of **Safe**   answers: 4.4141 (n=4)
- Unsafe > Safe (expected)? **PASS**

## Artifacts
- final pairs: `outputs/smoke8/final_pairs.joblib` (1 pairs)
- per-pair margins + gaps + think/answer lengths (CSV): `outputs/smoke8/final_pairs_margins.csv`
- sample pairs: `outputs/smoke8/sample_pairs.json` (1 pairs)
- consolidated per-stage diagnostics (JSON): `outputs/smoke8/diagnostics.json`
