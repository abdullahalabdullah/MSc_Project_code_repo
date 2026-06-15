# interpretation of g-mean^2 Results: qwen3_4b

**HuggingFace ID:** `Qwen/Qwen3-4B`  
**Role:** standalone  
**Instances evaluated (n_total):** 200

This AI generated and has many issues

---

### 1. Overall Suitability Verdict

- **Preliminary verdict: Borderline / Weak candidate for RO2.** The model passes Level-2 and produces a non-trivial g-mean² score (0.7553), but the extremely high Level-1 skip rate (76%) is the dominant concern — it means the eval suite would be operating on only ~24% of problems, with most excluded because the hint simply produced no behavioural shift. This heavily limits statistical power and the interpretability of cross-condition comparisons in RO2.
- The core issue is not monitorability per se (the monitor performs well on valid problems) but **hint susceptibility** — the model resists hints far more often than the base model needs to for a useful eval environment.

---

### 2. Filtering Results

- **Level-1 skip rate of 76% (152/200 excluded) is very high.** Out of 200 problems, only 48 pass (TE > 0 and p1 > 0). This means the honeypot stimuli are not reliably inducing hint-following behaviour in this model.
- **The dominant cause is TE = 0 (149 problems, 74.5%), not negative TE (only 3 problems, 1.5%).** This is an important distinction: TE = 0 means the model never followed the hint at all (p1 = 0 or p1 = p0), not that the hint backfired. The model is largely ignoring the hints rather than producing perverse hint-suppression effects.
- **Level-2 passes (mean TE = 0.1237 > 0.05).** Population-level hint-following exists, so the hint design is not completely broken — but the signal is concentrated in a small minority of problems. The 0.05 threshold is noted as "random and likely should be increased," and at 12.4% mean TE pre-filter, this model is only modestly above a threshold that is itself acknowledged as conservative.
- **Post-filter mean TE = 0.531 and mean R_effect = 0.865.** Among the 48 problems that do pass, the hint effect is strong and highly causal. This is good evidence that, conditional on the hint working at all, it is producing genuine causal influence rather than noise. But this is a self-selected subset — only problems where the model was already susceptible.

---

### 3. g-mean² and Component Scores

- **g-mean² = 0.7553.** This is a reasonably strong monitorability score. On valid problems, the monitor is catching hint-driven behaviour at a useful rate, and false alarm rates are low. As a standalone number it looks healthy.
- **However, this score applies to only 48 problems and is conditional on the model having been susceptible to the hint.** Its external validity for RO2 is limited: you cannot know whether the DPO conditions would shift hint-susceptibility itself (changing the skip rate) or shift monitorability conditional on susceptibility (changing g-mean² on valid problems), or both. The confound between these two effects is the main interpretive challenge for RO2 if this model were carried forward.
- **Mean TPR_mc = 0.7622.** The monitor detects hint-driven CoT influence roughly 76% of the time (relative to the R_effect-adjusted minimum criterion). This is solid but not near-ceiling, meaning there is meaningful unfaithfulness happening — some hint-following is not being acknowledged in the CoT. This is actually good for dissertation purposes (the measure is discriminating), but the absolute level is specific to the 48 susceptible problems.
- **Mean TNR_X=0 = 1.0000.** Perfect specificity on control responses — the monitor never falsely flags CoTs that were produced without any hint. This is expected (as per your docs, Guan et al. note TNR_X=0 ≈ 1 in practice) and confirms no degenerate monitor behaviour.
- **Mean TNR_X=1 = 0.9757.** Very high specificity on hint-resisting responses — in cases where the model saw the hint but did not follow it, the monitor almost never (2.4% FPR) incorrectly claims the hint influenced the CoT. This is the more meaningful and harder TNR to achieve, and it is strong here.
- **15/48 instances (31.25%) use the vacuous-truth convention for TNR_X=1.** This is notable — on nearly a third of valid problems, p1 = 1 (the model always followed the hint in the intervention arm, 0 hint-resistant runs). This means there is no X=1, Y=0 population to compute TNR_X=1 on, and it defaults to 1. These instances contribute to g-mean² without any empirical TNR_X=1 evidence. At 31%, this is high enough that you should flag it: the effective TNR_X=1 measured empirically (on the 33 non-vacuous problems) is slightly lower than the reported 0.9757, and the aggregate is pulled upward by the vacuous-truth imputation. Not a flaw in the metric, but worth noting in any write-up.

---

### 4. Monitor Validity

- **Monitor coverage is perfect across all three conditions: 0 missing verdicts out of 1596 total runs.** This is a clean result — no measurement gaps to worry about, and n_gmean2_undefined = 0 confirms no instances were dropped from the average due to coverage failures. The monitor pipeline is reliable for this model.

---

### 5. Confound Trackers

**Unhinted accuracy (X=0, mean = 0.771):**
- The model is correct 77% of the time on unhinted questions. This is moderately high and has implications for hint susceptibility: if the model already knows the correct answer with high confidence, it is less likely to be nudged toward a wrong hinted answer, which partly explains the 76% skip rate. High baseline accuracy → lower p1 → more TE = 0 cases.
- The std of 0.377 is large, indicating heterogeneous task difficulty across problems.

**CoT token lengths:**
- **Hint-following responses are dramatically longer (mean 1552.7 tokens) than hint-resisting responses (1170.6) and unhinted responses (910.6).** The hint-following CoT is on average 70% longer than the unhinted CoT. This is a meaningful confound for the monitor: the monitor may be learning to associate longer CoTs with hint influence, rather than reading the content of the reasoning. If CoT length is a proxy for "engaged reasoning about the hint," this isn't necessarily a fatal confound, but it's worth noting for the monitor's internal logic.
- The hint-following CoT has very high std (1494.8) and a max of 7714 tokens, suggesting occasional extremely extended reasoning in hint-following cases — possibly the model deliberating heavily before being swayed by the hint, or extensive post-hoc rationalisation.
- The std values are very large relative to means across all conditions, so per-instance variance in token length is high.

---

### 6. RO1 Decision Criteria — Summary Assessment

| Criterion | Assessment |
|---|---|
| Level-2 pass (mean TE > 0.05) | ✅ Passes (0.1237) |
| Sufficient valid problems for RO2 eval | ⚠️ Weak — only 48/200 (24%) |
| g-mean² discriminating (not floor/ceiling) | ✅ 0.7553 is in a useful range |
| Monitor reliability | ✅ Perfect coverage |
| High skip rate concern | ❌ 76% — dominant red flag |
| Vacuous-truth proportion | ⚠️ 31% — inflates apparent TNR_X=1 |
| Hint-susceptibility appropriate for target model | ❌ Model largely resists hints; low p1 across most problems |

**The fundamental tension:** Your eval suite was designed to be fixed after RO1 filtering on the SFT base model. If Qwen3-4B were the base, you'd be committing to a suite of only 48 valid problems — a thin statistical basis for cross-condition comparison in RO2. Even if the g-mean² scores look reasonable, differences between DPO conditions would have very wide confidence intervals, and skip-rate differences across conditions would be hard to interpret cleanly.

---

### 7. What to Do with This

- **Compare against other candidate models.** The primary use of this result is relative — Qwen3-4B should be compared against other models evaluated in the same RO1 sweep. A model with a lower g-mean² but a much lower skip rate (e.g., 40–50%) may actually be preferable as a base for RO2, because it gives you more valid problems to work with and more statistical power to detect DPO condition effects.
- **If Qwen3-4B is the only candidate showing reasonable g-mean²**, you may need to revisit the hint design (difficulty calibration, hint types) to increase susceptibility before committing. The 0.771 unhinted accuracy suggests the eval tasks may be too easy for this model — the model knows the answers well enough to resist the misleading hints.
- **The skip rate itself as a metric is informative.** As noted in your docs, per-condition skip rates in RO2 would be a substantive finding. But starting with only 24% valid problems before any DPO training makes this already fragile.