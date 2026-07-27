# Harmlessness preference-pair pipeline — report

- Sampled prompts (denominator): **300**
- Responses per prompt (N): 10, max_tokens: 8192

## Filtering funnel
- filter-1 (safety-label diversity): dropped **215 / 300** (71.7%) → 85 enter scoring
- filter-3 (chosen=Safe & rejected=Unsafe): dropped **14 / 85** (16.5%)
- **End-to-end survival rate: 71 / 300 = 23.7%**

## Cost-spread distribution (rejected − chosen)
- mean: 5.247  median: 4.000
- min: 0.906  max: 13.969
- q25: 2.206  q75: 8.129
- histogram: `outputs/run_20260703_121251/cost_spread.png`

## Sign-convention sanity check
- mean Beaver cost of **Unsafe** answers: 2.2277 (n=487)
- mean Beaver cost of **Safe**   answers: -1.2909 (n=363)
- Unsafe > Safe (expected)? **PASS**

## Artifacts
- final pairs: `outputs/run_20260703_121251/final_pairs.joblib` (71 pairs)
- sample pairs: `outputs/run_20260703_121251/sample_pairs.json` (10 pairs)


## Beaver ranker sanity check (WildJailbreak)

- Records scored: 2997 valid (prompt, answer) pairs (N=10 per prompt, 300 prompts).
- **Truncation over Beaver's 2048 ctx**: cost_full 80/2997 (2.7%), cost_vanilla 40 (adversarial only).

### 3+4. ROC-AUC of cost vs Qwen3Guard (higher cost = unsafe)

| condition | AUC | mean(unsafe) | mean(safe) | unsafe>safe | n (unsafe/safe) |
|---|---|---|---|---|---|
| vanilla / cost_full | 0.989 | 1.604 | -5.827 | PASS | 1260 (254/1006) |
| adversarial / cost_full | 0.873 | 3.486 | -0.551 | PASS | 1737 (1230/507) |
| adversarial / cost_vanilla | 0.849 | 3.746 | -0.908 | PASS | 1737 (1230/507) |

### 5. Pair agreement — Beaver vs WildGuard argmin/argmax (filter-1 survivors)

| subset | chosen match % | rejected match % | n prompts |
|---|---|---|---|
| vanilla | 40.7 | 29.6 | 27 |
| adversarial | 37.9 | 43.1 | 58 |

### 6. Length confound (adversarial): Pearson(cost_full, prompt token length)

- within **safe**-labeled answers:   r = 0.080  (n=507)
- within **unsafe**-labeled answers: r = -0.012  (n=1230)

### 7. Disagreements (Beaver sign vs Qwen3Guard, adversarial)

- total contradictions: 284; dumped 10 random -> `outputs/run_20260703_121251/beaver_sanity_disagreements.jsonl`
