# Beaver ranker sanity check — results

Does `PKU-Alignment/beaver-7b-unified-cost` reliably separate safe from unsafe answers on WildJailbreak prompts (vanilla + adversarial), scored with the raw prompt vs the vanilla seed query? See `documentation.md` for the full design and how to read this.

**Reading the AUC table:** adversarial `cost_full` AUC ≈ vanilla → Beaver usable as-is; adversarial `cost_full` ≪ vanilla but adversarial `cost_vanilla` ≈ vanilla → Beaver usable only with vanilla-query scoring; all weak → drop Beaver, use WildGuard as sole ranker.


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
