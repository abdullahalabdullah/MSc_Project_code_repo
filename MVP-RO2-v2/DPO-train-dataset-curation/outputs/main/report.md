# Harmlessness preference-pair pipeline — report

- Sampled prompts (denominator): **24000** (SAMPLE=ALL)
- Responses per prompt (N): 10, max_tokens: 8192

## Filtering funnel
- filter-1 (safety-label diversity): dropped **17988 / 24000** (75.0%) → 6012 enter scoring
- filter-3 (chosen=Safe & rejected=Unsafe): dropped **1660 / 6012** (27.6%) → 4352 pairs
- filter-4 (minimum margin): **not applied** (MIN_MARGIN unset)
- **End-to-end survival rate: 4352 / 24000 = 18.1%**

## Per-subset funnel (sample mix after each filter)
| subset | sampled | after filter-1 | after filter-3 | final | survival % |
|---|---|---|---|---|---|
| adversarial | 14917 | 4146 | 2799 | 2799 | 18.8 |
| vanilla | 9083 | 1866 | 1553 | 1553 | 17.1 |
| **all** | 24000 | 6012 | 4352 | 4352 | 18.1 |

## Labeler agreement (Beaver argmin/argmax vs Qwen3Guard)
- chosen-side (Beaver-safest is Qwen3Guard-Safe): **85.2%** (5122/6012)
- rejected-side (Beaver-harshest is Qwen3Guard-Unsafe): **85.7%** (5151/6012)
- both agree (→ filter-3 survivors): **72.4%** (4352/6012)
- point-biserial r(cost, Unsafe) over all scored responses: **+0.486** (+1 = cost perfectly separates Qwen3Guard classes; 0 = no separation)

## Margin calibration (isolation of the chosen/rejected picks)
- **chosen runner-up gap** (2nd-safest − safest cost): mean 1.058, median 0.438 — small ⇒ the safe pick was near-tied
- **rejected runner-up gap** (harshest − 2nd-harshest cost): mean 1.009, median 0.472
- mean margin (adversarial): 4.631 (n=2799)
- mean margin (vanilla): 7.951 (n=1553)
- **Qwen3Guard decisiveness** (mean Unsafe fraction per surviving prompt): 0.50 (0.5 = maximally split)

## Think-block vs answer length (final pairs, words)
- **chosen** — think mean 451, answer mean 360 (think/answer ratio 1.3)
- **rejected** — think mean 568, answer mean 504 (think/answer ratio 1.1)
- (CoT is retained in the DPO pair but never scored/labeled — this tracks how much of each response is reasoning vs visible answer.)

## Cost-spread distribution (rejected − chosen)
- mean: 5.816  median: 4.427
- min: 0.387  max: 19.438
- q25: 2.447  q75: 8.688
- histogram: `outputs/main/cost_spread.png`

## Length / verbosity confound (chosen vs rejected)
- **Full response (DPO pair content), tokens** — chosen mean 1069, rejected mean 1468, mean Δ(rej−cho) +399 (median +333); rejected longer in 78% of pairs
- **Visible answer (scored text), words** — chosen mean 360, rejected mean 504, mean Δ(rej−cho) +144 (median +86); rejected longer in 71% of pairs
- **Pearson(cost margin, answer-word Δ)**: +0.424 (|r|→1 ⇒ cost margin is confounded by length; ~0 ⇒ decoupled)
- Note: a strong one-sided skew (rejected ≫ chosen) risks DPO learning length rather than harmlessness — watch this when scaling up.

## Sign-convention sanity check
- mean Beaver cost of **Unsafe** answers: 1.8343 (n=30167)
- mean Beaver cost of **Safe**   answers: -1.6245 (n=29837)
- Unsafe > Safe (expected)? **PASS**

## Artifacts
- final pairs: `outputs/main/final_pairs.joblib` (4352 pairs)
- per-pair margins + gaps + think/answer lengths (CSV): `outputs/main/final_pairs_margins.csv`
- sample pairs: `outputs/main/sample_pairs.json` (10 pairs)
- consolidated per-stage diagnostics (JSON): `outputs/main/diagnostics.json`
