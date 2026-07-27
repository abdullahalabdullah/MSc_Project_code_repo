# Harmlessness preference-pair pipeline — report

- Sampled prompts (denominator): **4500** (SAMPLE=4500)
- Responses per prompt (N): 10, max_tokens: 8192

## Filtering funnel
- filter-1 (safety-label diversity): dropped **3348 / 4500** (74.4%) → 1152 enter scoring
- filter-3 (chosen=Safe & rejected=Unsafe): dropped **285 / 1152** (24.7%) → 867 pairs
- filter-4 (minimum margin): **not applied** (MIN_MARGIN unset)
- **End-to-end survival rate: 867 / 4500 = 19.3%**

## Per-subset funnel (sample mix after each filter)
| subset | sampled | after filter-1 | after filter-3 | final | survival % |
|---|---|---|---|---|---|
| adversarial | 2770 | 794 | 553 | 553 | 20.0 |
| vanilla | 1730 | 358 | 314 | 314 | 18.2 |
| **all** | 4500 | 1152 | 867 | 867 | 19.3 |

## Labeler agreement (Beaver argmin/argmax vs Qwen3Guard)
- chosen-side (Beaver-safest is Qwen3Guard-Safe): **85.6%** (986/1152)
- rejected-side (Beaver-harshest is Qwen3Guard-Unsafe): **88.3%** (1017/1152)
- both agree (→ filter-3 survivors): **75.3%** (867/1152)
- point-biserial r(cost, Unsafe) over all scored responses: **+0.509** (+1 = cost perfectly separates Qwen3Guard classes; 0 = no separation)

## Margin calibration (isolation of the chosen/rejected picks)
- **chosen runner-up gap** (2nd-safest − safest cost): mean 0.975, median 0.406 — small ⇒ the safe pick was near-tied
- **rejected runner-up gap** (harshest − 2nd-harshest cost): mean 1.081, median 0.500
- mean margin (adversarial): 4.727 (n=553)
- mean margin (vanilla): 8.081 (n=314)
- **Qwen3Guard decisiveness** (mean Unsafe fraction per surviving prompt): 0.50 (0.5 = maximally split)

## Think-block vs answer length (final pairs, words)
- **chosen** — think mean 459, answer mean 355 (think/answer ratio 1.3)
- **rejected** — think mean 571, answer mean 502 (think/answer ratio 1.1)
- (CoT is retained in the DPO pair but never scored/labeled — this tracks how much of each response is reasoning vs visible answer.)

## Cost-spread distribution (rejected − chosen)
- mean: 5.941  median: 4.818
- min: 0.562  max: 19.312
- q25: 2.523  q75: 8.771
- histogram: `outputs/pilot/cost_spread.png`

## Length / verbosity confound (chosen vs rejected)
- **Full response (DPO pair content), tokens** — chosen mean 1075, rejected mean 1465, mean Δ(rej−cho) +390 (median +337); rejected longer in 79% of pairs
- **Visible answer (scored text), words** — chosen mean 355, rejected mean 502, mean Δ(rej−cho) +147 (median +85); rejected longer in 70% of pairs
- **Pearson(cost margin, answer-word Δ)**: +0.399 (|r|→1 ⇒ cost margin is confounded by length; ~0 ⇒ decoupled)
- Note: a strong one-sided skew (rejected ≫ chosen) risks DPO learning length rather than harmlessness — watch this when scaling up.

## Sign-convention sanity check
- mean Beaver cost of **Unsafe** answers: 1.8822 (n=5788)
- mean Beaver cost of **Safe**   answers: -1.7790 (n=5711)
- Unsafe > Safe (expected)? **PASS**

## Artifacts
- final pairs: `outputs/pilot/final_pairs.joblib` (867 pairs)
- per-pair margins + gaps + think/answer lengths (CSV): `outputs/pilot/final_pairs_margins.csv`
- sample pairs: `outputs/pilot/sample_pairs.json` (10 pairs)
- consolidated per-stage diagnostics (JSON): `outputs/pilot/diagnostics.json`
