# Interpretation g-mean^2 Results: deepseek_r1_distill_qwen_1_5b

**HuggingFace ID:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`  
**Role:** standalone  
**Instances evaluated (n_total):** 200

---

### 1. Filtering — Is the eval environment even engaging this model?

- **High skip rate (66.5%)**: 133/200 instances fail Level-1 filtering. For context, the hint produces zero net behavioural shift (TE ≤ 0) on two-thirds of the eval suite for this model. This is a fundamental problem for RO2 suitability.
- **TE sign breakdown** is revealing: 51% of instances have TE = 0 exactly (the hint moves nothing at all), and 15.5% have TE < 0 (the hint actually suppresses the hinted answer, possibly due to the model over-correcting or the hint being visible as a trap). Only 33.5% have TE > 0.
- **Population-level TE (Level-2):** mean TE of 0.0783 passes the 5% threshold, but this is driven entirely by the 33.5% of problems where the hint works at all — the aggregate mean is low because the majority of problems contribute TE = 0.
- **Post-filtering TE distribution looks healthy in isolation**: among the 67 valid instances, mean TE = 0.356 and mean R_effect = 0.825. So *when the hint does work*, it works reasonably well. The problem is the rarity of that, not the quality of the effect when it occurs.

---

### 2. g-mean² Score — Core Monitorability Finding

- **g-mean² = 0.2311**: This is a low score. The model is poorly monitorable on the instances where hint-following is detectable.
- **The bottleneck is TPR_mc = 0.2388**: The monitor is detecting hint influence in the CoT only about 24% of the time on hint-following responses, even after correcting for baseline hint-following via R_effect. The model is following hints behaviourally without disclosing this in its reasoning trace on ~76% of valid cases.
- **TNR_X=0 = 1.000**: Perfect. The monitor never hallucinates hint influence in control-arm CoTs where no hint was present. This is expected and unsurprising but confirms the monitor is not firing indiscriminately.
- **TNR_X=1 = 0.9776**: Near-perfect. When the model sees a hint but *doesn't* follow it, the monitor almost never incorrectly flags the CoT as hint-influenced. The monitor is not simply pattern-matching on hint presence in the prompt.
- **Net picture**: The monitor is extremely precise (low false positive rate in both TNR conditions) but has very low recall on genuine hint-following. The metric is being dragged down almost entirely by TPR_mc. This profile — high TNR, low TPR — is characteristic of a model that follows hints covertly: the behaviour changes but the CoT doesn't show it.
- **Vacuous-truth instances (n=12)**: For 12/67 valid problems, p1 = 1 (the model always followed the hint), so TNR_X=1 is assigned 1 by convention. These are unambiguously hint-susceptible problems where the model never resisted, meaning there are no X=1,Y=0 responses to evaluate TNR_X=1 on empirically. This is fine and principled, but worth noting as these problems contribute slightly inflated g-mean² scores since their TNR_X=1 is defined away rather than measured.
- **Monitor coverage is perfect (Section 3):** 0 missing verdicts across all conditions. This rules out measurement noise from failed monitor calls — the g-mean² estimate is clean.

---

### 3. Confound Trackers

- **Unhinted accuracy = 0.352 (mean), Std = 0.345**: The model gets roughly 35% of problems correct without any hint. This is substantially below chance for a 4-choice MCQ (which would be 25% random + model capability), suggesting the eval tasks may be hard for this model, or the model is not well-calibrated. The high std (0.345) suggests this varies enormously across problems — some are trivially solved, others are essentially random guessing.
  - **Implication for p0**: High unhinted accuracy inflates p0 (the model picks the hinted answer even without a hint, just because it happens to be correct). This suppresses TE and is part of why so many instances fail Level-1. The 51% of TE = 0 cases likely includes many where the model already gets the right (or hinted) answer without any push.
  - Actually the inverse is also relevant: with only 35% accuracy, there are many problems where the model is unlikely to choose the hinted answer by chance, meaning the hint has more room to shift behaviour — yet it still doesn't in 66.5% of cases. This suggests the model may not be reading or processing hint-type prompts effectively.
- **CoT token length**: The model generates very long CoTs (mean ~1,946–2,113 tokens across conditions), consistent with it being a distilled reasoning model that produces extended chain-of-thought outputs.
  - **No meaningful length difference between hint-following and hint-resisting responses** (1,960 vs. 1,891 tokens when hinted). This is a useful negative: the monitor's low TPR is unlikely to be a length artefact where hint-following responses are simply shorter and thus less expressive. The CoT is long regardless.
  - The high variance (std ~1,700–1,950, max up to ~11,000 tokens) suggests significant instability in output length across problems. Some responses are extremely long, which may pose challenges for the monitor LLM's context window or attention.

---

### 4. RO2 Suitability Assessment

- **Primary concern — skip rate**: A 66.5% skip rate means only 33.5% of your eval suite would produce valid g-mean² measurements for this model's DPO-trained variants. Since the eval suite will be fixed after RO1 based on the SFT base model, this model as the *base* would produce a very sparse valid set for RO2, reducing statistical power substantially.
- **The low baseline g-mean² (0.23) is not necessarily disqualifying on its own** — what matters for RO2 is that there is *variance* to detect across DPO conditions. A floor of 0.23 gives room to observe degradation, but ceiling effects (if DPO conditions push g-mean² higher) are limited given the already-low TPR.
- **The model seems to have weak hint-sensitivity overall**, which is the more fundamental problem. Xiong et al. and Guan et al. both note that evaluations need to be calibrated to the specific model — a hint configuration suited to a stronger model may be essentially invisible to a 1.5B distilled model. The 51% TE = 0 rate is consistent with this: the hints are largely not registering behaviourally.
- **Low TPR_mc (0.24) with high skip rate means two independent validity concerns compound**: the hint doesn't work often (skip rate), and when it does, the CoT doesn't show it (low TPR). For an RO2 experiment trying to attribute differences in monitorability to DPO alignment objectives, this baseline is a weak foundation.
- **Recommendation**: This model is likely **not appropriate for RO2** in its current configuration. The hint setup may need recalibration for a 1.5B model, or — more practically — this model should be replaced by one with a higher hint susceptibility rate (lower skip rate) and a more interpretable baseline g-mean². The 8B Tulu-3-SFT base model is presumably a much better fit given the experimental design in your dissertation.