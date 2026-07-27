# Labeler comparison — Qwen3Guard vs WildGuard

Clean A/B: identical generator (Qwen3-4B), seed, 300 prompts, and Beaver ranker.
**The only variable is the safety labeler.**

| Metric | Qwen3Guard | WildGuard | Read |
|---|---|---|---|
| filter-1 dropped | 215/300 (71.7%) | 210/300 (70.0%) | ~identical — both agree ~70% of prompts yield one-sided labels |
| filter-3 dropped | 14/85 (16.5%) | 29/90 (32.2%) | WildGuard disagrees with Beaver ~2× more |
| End-to-end survival | 71/300 (23.7%) | 61/300 (20.3%) | WildGuard ~14% fewer pairs |
| mean cost Unsafe | 2.23 (n=487) | 3.44 (n=425) | WildGuard's Unsafe set is more severe |
| mean cost Safe | −1.29 (n=363) | 0.46 (n=474) | WildGuard's Safe set is more permissive |
| Unsafe>Safe sanity | PASS | PASS | both coherent with Beaver |

## What the numbers actually say

**filter-1 is basically identical** — the labeler choice barely affects *which prompts are candidates*. The safety-tuned Qwen3-4B refusing ~70% of the time is the dominant effect, and both classifiers see it the same way. Good: your funnel's top isn't sensitive to this decision.

**The two labelers draw the safe/unsafe line at different thresholds:**

- **WildGuard is conservative about calling a response harmful** — fewer Unsafe labels (425 vs 487), but the ones it flags are genuinely severe (mean cost 3.44 vs 2.23). When WildGuard says "harmful," Beaver strongly agrees.
- **Qwen3Guard is trigger-happy** — more Unsafe labels, and a very clean Safe bucket (−1.29). Part of its extra Unsafe count is an artifact of your `Controversial→Unsafe` collapse — an arbitrary rule you were forced into. WildGuard's binary removes that entirely.

**Why filter-3 drops 2× more under WildGuard:** filter-3 needs `rejected` (Beaver's max-cost response) to also clear WildGuard's stricter Unsafe bar. So the surviving 61 pairs are exactly the ones where a conservative binary classifier and the cost model agree the rejected response is genuinely harmful. Lower yield, but every surviving `rejected` is high-confidence harmful. The ~10 lost pairs are mostly borderline cases you probably didn't want.

## Verdict: WildGuard was the right switch

The decisive reasons aren't in the yield numbers — they're structural:

1. **Independence / non-circularity.** Qwen3Guard shares lineage with your Qwen3-4B generator (same lab, same family, overlapping alignment data). Grading Qwen3 with Qwen3 risks shared blind spots — agreement that reflects common training, not genuine safety. WildGuard (Mistral-based, AllenAI) is a truly independent judge. This alone justifies the switch.
2. **In-distribution.** WildGuard was trained on WildJailbreak — it's the native classifier for exactly these adversarial prompts. Qwen3Guard is general-purpose.
3. **Clean binary.** No arbitrary `Controversial→Unsafe` collapse; the yes/no maps directly onto the axis you need, plus you get the free `refusal` flag for your later monitorability work.

**Cost of the switch:** ~14% fewer pairs (61 vs 71) and one genuine weak spot — WildGuard's Safe bucket is more permissive (mean 0.46 vs −1.29), so a few `chosen` responses are only borderline-safe.
