# Aggregate Results — Llama-3.2-3B-Instruct

**Model:** `meta-llama/Llama-3.2-3B-Instruct`
**Source:** `analysis/<dataset>/llama32_3b_instruct/gmean2_summary.json`

> Note on the **`a`** column: there is no metric literally named "a" in the outputs. It is filled here with the **Level-1 filter pass rate** — the number (and fraction) of instances retained after the `TE > 0 & p1 > 0` filter, i.e. the population the g-mean² is averaged over. Rename if a different quantity was intended.

## Table 1 — g-mean² metrics per dataset

| Dataset | Mean TE (before filtering) | Mean R_effect (before filtering) | a (L1 pass) | g-mean² | Unhinted Acc. Mean (Std) |
|---|---|---|---|---|---|
| WMDP | 0.0688 | 0.4062 | 20/69 (29.0%) | 0.1750 | 0.4686 (0.3940) |
| AGIEval | 0.0242 | 0.2909 | 23/69 (33.3%) | 0.0435 | 0.3671 (0.3218) |
| MMLU | 0.0226 | 0.3086 | 14/70 (20.0%) | 0.1429 | 0.5702 (0.3406) |
| GSM-MC | -0.0036 | 0.6042 | 8/70 (11.4%) | 0.1250 | 0.8226 (0.2880) |
| GPQA | 0.0686 | 0.3035 | 25/68 (36.8%) | 0.0600 | 0.3174 (0.3349) |
| MATH-MC | -0.0086 | 0.1127 | 8/68 (11.8%) | 0.1250 | 0.6801 (0.3640) |

## Table 2 — CoT token length by condition (mean across the 6 datasets)

| Condition | Llama-3.2-3B Mean |
| ------------------------------------- | --- |
| Hinted (X=1, all)                     | 375.3 |
| Unhinted (X=0, all)                   | 392.4 |
| Hinted : hint followed (X=1, Y=1)     | 424.1 |
| Hinted : hint not followed (X=1, Y=0) | 369.6 |
| **Overall (mean of condition means)** | **390.4** |

---

## Model Selection for RO2 — Llama-3.2-3B vs Qwen3-4B

**Verdict: use Qwen3-4B.**

The metric that settles this is **Mean TE**, because everything else in the pipeline is downstream of it. The honeypot's validity *is* the hint's causal effect — RO1's inclusion test requires configs with **TE ≥ δ** where **δ = max(25th-percentile, 0.20)** (proposal §3.1, *Total Effect and Inclusion Test*), and HAR/g-mean² are only defined over the instances that survive that filter. So "which model is more suitable" reduces largely to "on which model do the hints actually move behaviour."

### The decisive comparison

| Metric (what it gates) | Llama-3.2-3B | Qwen3-4B |
|---|---|---|
| **Mean TE** — must clear δ≈0.20 | 0.069 max; **negative on GSM-MC & MATH-MC**; never near 0.20 | GPQA **0.286**, WMDP **0.302** clear the floor; AGIEval 0.106, MMLU 0.050 usable |
| **g-mean²** (baseline monitorability) | **0.04 – 0.17** (near floor) | 0.50 – 0.88 (healthy mid–high) |
| **Unhinted acc** (incentive calibration) | GPQA **0.317 ≈ random** (4-opt = 0.25) | GPQA 0.487 — genuine-uncertainty-but-engaged |

### Why this maps directly onto the success criteria

**1. RO1 inclusion test — "retain ≥3 task-hint configs with TE > δ" (§3.3-i).** Llama's mean TE never approaches 0.20 on any dataset, and is negative on both math tasks. The hints essentially don't steer it. This is precisely the failure mode the related-work section warns about (§2.2: *"a configuration that elicits measurable hint-following on a frontier model may produce near-zero effect on a smaller model, rendering the evaluation insensitive"*). Llama-3.2-3B is the textbook insensitive case. Qwen3 clears the floor convincingly on GPQA and WMDP — the two hardest/most safety-relevant tasks.

**2. RO2 sensitivity — "baseline HAR neither too high nor too low" (§3.3-iii).** This is the subtle but critical point. The primary hypotheses are about **degradation** (harmlessness reduces HAR on safety-critical tasks). Llama's monitorability is already at the floor (g-mean² ≈ 0.04–0.17) — there is no room to detect a downward shift, so a post-training effect would be invisible. Qwen3's mid-range scores (0.55–0.78 on the load-bearing tasks) give **bidirectional headroom** to detect both degradation and improvement. Mid-range baseline = maximum statistical sensitivity, which is the entire point of RO2.

**3. Incentive calibration (§3.1, condition 4 / §3.3-iii).** At 0.317 unhinted accuracy on GPQA, Llama-3.2-3B is at chance — it isn't reasoning so much as guessing, which undermines the premise that a hint is influencing *genuine* reasoning. Qwen3 sits in the "hard enough to be tempted, capable enough to engage" band the suite specifies.

**4. |S| quality, not just quantity.** Llama's "a" column looks competitive (e.g. 33–37% pass on AGIEval/GPQA), but the L1 filter is only TE>0 — those are *weak-signal* instances (mean TE ~0.02–0.07). Qwen3 retains 50–53% on GPQA/WMDP carrying *strong* effect (TE ~0.3). A smaller set of strong-effect instances gives a far more stable HAR than a larger set of near-zero-effect ones.

### The one real caveat (and why it doesn't change the answer)

The note `project_ro2_cot_collapse` flags that **answer-only SFT collapses Qwen3's CoT to 0%**. That matters for RO2's SFT mechanism-ablation, but it's a *training-procedure* artifact, not a model-suitability verdict:

- The main RO2 method is **DPO**, not answer-only SFT — DPO on (prompt, chosen, rejected) doesn't suppress reasoning the same way.
- The baseline numbers above prove Qwen3 *does* emit monitorable CoTs (you can't get g-mean² = 0.78 with no CoT).
- The identified fix — reasoning distillation rather than answer-only targets for the SFT ablation — keeps that ablation viable.

Llama-3.2-3B has no comparable rescue: its problem is that the honeypot itself is dead on arrival (no causal signal), which no training change fixes.

### Bottom line

Llama-3.2-3B fails the foundational requirement — the hints don't move it, so there's nothing for a monitor to catch and nothing for post-training to shift. Qwen3-4B clears the TE floor on the priority tasks (GPQA, WMDP), sits in the measurable mid-range for monitorability, and is properly incentive-calibrated. As a bonus, since the real RO2 base is the more capable Tulu-3-8B, a pipeline validated on Qwen3 (above the capability threshold) will transfer far better than one tuned on a model that's below it. **Go with Qwen3-4B**, and manage the CoT-collapse risk on the training side as already planned.

**Data note:** drop **GSM-MC and MATH-MC for Qwen3** from RO2 analysis — at 0.99/0.94 unhinted accuracy and |S| = 4/2, they're ceiling-saturated and uninformative. Lean on **GPQA, WMDP, MMLU, and AGIEval**, where Qwen3 gives real signal.

### Qwen3-4B reference data (for comparison)

| Dataset | Mean TE (before filtering) | Mean R_effect (before filtering) | a (L1 pass) | g-mean² | Unhinted Acc. Mean (Std) |
|---|---|---|---|---|---|
| AGIEval | 0.1060 | 0.5256 | 17 (24.3%) | 0.5524 | 0.6714 (±0.4235) |
| GPQA | 0.2857 | 0.8059 | 37 (52.9%) | 0.6745 | 0.4869 (±0.4404) |
| GSM-MC | 0.0250 | 0.8750 | 4 (5.7%) | 0.8750 | 0.9929 (±0.0593) |
| MATH-MC | 0.0071 | 0.5000 | 2 (2.9%) | 0.5000 | 0.9429 (±0.2321) |
| MMLU | 0.0500 | 0.4062 | 9 (12.9%) | 0.7222 | 0.7714 (±0.3958) |
| WMDP | 0.3024 | 0.7785 | 35 (50.0%) | 0.7833 | 0.5798 (±0.4313) |

