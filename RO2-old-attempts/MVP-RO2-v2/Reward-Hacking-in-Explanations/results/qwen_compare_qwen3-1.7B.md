# Base vs DPO+SK-Llama | Majority@16 | qwen3-1.7B

| Metric | Base | DPO+RM | Change |
|---|---|---|---|
| Valid responses orig % | 99.8 | 98.5 | -1.3 |
| Valid responses CF % | 99.5 | 98.1 | -1.5 |
| Acc orig % | 94.1 | 93.3 | -0.8 |
| Acc CF % | 85.0 | 83.9 | -1.2 |
| Acc diff % | 9.1 | 9.4 | +0.4 |
| Ack orig % | 27.2 | 23.2 | -3.9 |
| Ack CF % | 0.0 | 0.4 | +0.4 |
| Ack diff % | 27.2 | 22.8 | -4.3 |
| **Deceptive % (Maj@16)** | **2.0** | **2.8** | **+0.8** |

n (test examples): 254

## Paper reference (LLaMA-3.1-8B-IT, SK-LLAMA-8B, Maj@16, Math Book)

Full metric comparison, matching the layout above (from paper Table 2 for
Acc/Ack and Table 1 for the deceptive rate). Orig = protected feature present
(PF ✓); CF = protected feature removed (PF ×). Validity is not reported by the
paper.

| Metric | Base | DPO+RM | Change |
|---|---|---|---|
| Valid responses orig % | — | — | — |
| Valid responses CF % | — | — | — |
| Acc orig % | 79.4 | 89.8 | +10.4 |
| Acc CF % | 53.0 | 56.2 | +3.2 |
| Acc diff % | 26.4 | 33.6 | +7.2 |
| Ack orig % | 0.0 | 0.3 | +0.3 |
| Ack CF % | 0.0 | 0.0 | 0.0 |
| Ack diff % | 0.0 | 0.3 | +0.3 |
| **Deceptive % (Maj@16)** | **27.2** | **34.0** | **+6.8** |

Deceptive rate (headline) with error bars:

| | Deceptive % |
|---|---|
| Base | 27.2 ± 1.5 |
| DPO+RM | 34.0 ± 0.7 |
| Δ | +6.8 |



--- 


It's a weak positive signal, and here's why.

**What the AQuA DPO result actually shows:** A capability-focused objective (math reasoning) with no relation to safety or honesty still moved deceptiveness by ~40% relatively (2.0%→2.8%). It also moved g-mean² substantially on knowledge tasks in your earlier RO1 analysis (+0.119 on mmlu, +0.069 on agieval) — entirely through CoT length changes. The point is: even a tangential objective with no theoretical connection to monitorability shifted it measurably.

**Why that's encouraging for your idea:** Harmlessness, helpfulness, and honesty DPO have *more direct* mechanistic pathways to monitorability changes than math reasoning DPO does. Harmlessness trains the model to produce safe-sounding outputs — this can directly suppress hint acknowledgment on WMDP-type prompts by incentivising CoTs that look aligned regardless of what drove the answer. Helpfulness trains agreeableness, which plausibly increases hint-following (a sycophancy pressure). Anti-sycophancy does the inverse. These aren't tangential effects — they're the core of what your hypothesis predicts. If a math objective moves the needle incidentally, the objectives you're actually targeting should move it more intentionally.

**The concern the result raises:** The near-floor deceptive rate (2.0%) tells you the *current evaluation setup* — math task, Qwen3-1.7B — is too easy for the model to stress-test these effects. You'd train harmlessness DPO and then try to measure its effect on a task where the model is already 94% accurate and barely ever follows wrong hints. There's no room for the effect to manifest. This isn't a problem with the idea, it's a problem with the instrument being used to test it.

**The bottom line:** Your idea has real potential — the theoretical pathways are sound and even indirect DPO already moves monitorability. But the result confirms you need the right instrument to detect it: gpqa and wmdp (where the model has genuine uncertainty and actual susceptibility), g-mean² as the metric, and a model with non-trivial baseline TE. The math task deceptive % is just not sensitive enough to be the yardstick.