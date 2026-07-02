# Harmlessness preference-pair pipeline — report

- Sampled prompts (denominator): **300**
- Responses per prompt (N): 8, max_tokens: 8192

## Filtering funnel
- filter-1 (safety-label diversity): dropped **275 / 300** (91.7%) → 25 enter scoring
- filter-3 (chosen=Safe & rejected=Unsafe): dropped **7 / 25** (28.0%)
- **End-to-end survival rate: 18 / 300 = 6.0%**

## Cost-spread distribution (rejected − chosen)
- mean: 8.485  median: 6.893
- min: 4.578  max: 14.938
- q25: 5.958  q75: 10.617
- histogram: `/root/MSc_Project_code_repo/Reward-Hacking-in-Explanations/harmlessness-exp/outputs/cost_spread.png`

## Sign-convention sanity check
- mean Beaver cost of **Unsafe** answers: 2.1554 (n=90)
- mean Beaver cost of **Safe**   answers: -2.9607 (n=110)
- Unsafe > Safe (expected)? **PASS**

## Artifacts
- final pairs: `/root/MSc_Project_code_repo/Reward-Hacking-in-Explanations/harmlessness-exp/outputs/final_pairs.joblib` (18 pairs)
- sample pairs: `/root/MSc_Project_code_repo/Reward-Hacking-in-Explanations/harmlessness-exp/outputs/sample_pairs.json` (10 pairs)
