# Investigating the Impact of Safety and Alignment Post-Training on Chain-of-Thought Monitorability

**Abdullah Alabdullah**

## Abstract

Chain-of-Thought (CoT) monitoring offers a practical approach to detecting unsafe and misaligned behaviour in LLMs, but its reliability depends on CoTs faithfully disclosing the factors driving model decisions. Safety and alignment post-training, now standard practice at major AI labs, may degrade this transparency by rewarding CoTs that appear aligned without reflecting the model's true decision process. Yet no prior work has directly evaluated how individual safety and alignment post-training objectives affect CoT monitorability. We address this gap in two phases. First, we design and empirically validate a cue-insertion honeypot environment calibrated to elicit measurable hint-following behaviour on the target model across diverse task and hint configurations. Second, we apply offline DPO fine-tuning under three post-training objectives (harmlessness, helpfulness, and anti-sycophancy) and quantify their effects on CoT monitorability via our hint-acknowledgment rate (HAR) metric. We contribute a reusable evaluation suite and empirical findings characterising which post-training objectives preserve, enhance, or degrade CoT monitorability.

---

## 1 Introduction

### 1.1 CoT Monitorability

Ensuring that large language models (LLMs) remain monitorable in high-stakes domains such as healthcare, finance, and law is a central challenge for AI safety and alignment [1, 2, 3]. Chain-of-Thought (CoT) faithfulness refers to how well a reasoning trace reflects a model's underlying reasoning, while CoT monitorability concerns whether a CoT enables the detection of unsafe or misaligned behaviour. Faithfulness is necessary but not sufficient for monitorability: a CoT must reflect internal reasoning to support meaningful oversight, yet must also be sufficiently expressive to surface all safety-relevant factors [4]. Current monitoring approaches face significant limitations: mechanistic interpretability methods are often model-specific and scale poorly to complex systems [5], while black-box evaluations can only detect behaviours that have been explicitly anticipated [6].

CoT reasoning offers a practical alternative by producing legible, step-by-step traces that are cheap to log and easy to integrate into oversight workflows [7]. As CoT is directly optimised during training, used as a search space at inference, and underpins planning in agentic contexts [8, 9], monitoring it provides insight into model incentives and potential misbehaviour [10]. Prior empirical research shows that CoT monitoring can detect reward-hacking more effectively than action-based monitors [7]. Although CoTs are imperfect summaries of internal computation and can be unfaithful or incomplete [10, 11, 12], the practically relevant question is not whether they fully reflect internal reasoning, but whether they provide sufficient signal to detect critical behaviours such as the pursuit of unsafe goals [13, 7].

Despite growing research into how training incentives affect CoT monitorability, the impact of safety and alignment post-training, now standard practice at major AI labs [14], has not been directly evaluated. Safety training may improve monitorability by encouraging explicit safety reasoning, or degrade it by rewarding traces that appear well-reasoned while concealing the underlying factors driving the model's decision [15]. Because different objectives (e.g. helpfulness, harmlessness) may have distinct effects, identifying which preserve or undermine monitorability is essential for designing safer post-training pipelines.

### 1.2 Research Question and Objectives

This project addresses the following research question: **How does safety and alignment post-training affect CoT monitorability?** This question is addressed through two research objectives. RO1 aims to design and validate a reliable cue-insertion honeypot environment for evaluating CoT monitorability. RO2 then employs this evaluation environment to measure the effect of safety and alignment post-training on CoT monitorability.

---

## 2 Background and Related Work

### 2.1 Chain-of-Thought Faithfulness and Monitorability

Prior work has shown that CoT explanations can diverge from the true causes of model behaviour, even when the reasoning appears fluent and coherent. Turpin et al. [12] showed that models produced CoTs that appeared logically sound and self-consistent while their final answers were systematically influenced by prompt features (such as answer-position patterns, hints in prompts, and stereotypes) that the CoT never acknowledged, producing plausible post-hoc rationalisations rather than a transparent reflection of the underlying reasoning process. Lanham et al. [16] further showed that faithfulness varies by task and model: harder tasks tend to produce more causally load-bearing CoTs, while easier tasks elicit post-hoc rationalisations. Arcuschin et al. [17] extended these findings by demonstrating that unfaithful CoTs can appear under realistic prompting conditions without artificial interventions, establishing that such unfaithfulness can emerge naturally in standard deployment rather than being merely an artefact of adversarial setups.

Guan et al. [18] conducted a comprehensive empirical study of monitorability, finding that frontier models are generally, but imperfectly, monitorable under ordinary conditions, and that CoT monitors substantially outperform action-only monitors in detecting safety-relevant properties. Meek et al. [4] identified a further constraint: a CoT can acknowledge an inserted cue while still omitting the causal factors a monitor would need to understand how that cue influenced the model's decision. Monitorability therefore requires not only faithfulness but also sufficient verbosity: the CoT must include all relevant factors, not merely acknowledge that some influence was present.

### 2.2 Evaluating CoT Monitorability

Because the model's internal computation cannot be easily traced, every behavioural evaluation approach is a proxy: it tests whether a specific form of faithfulness failure is detectable, not whether the CoT is faithful in any general sense.

The field has developed several evaluation strategies, including CoT intervention [16], self-consistency checks [19], end-to-end catch rate evaluation [7], prompt-level hint injection [12, 20], and causal attribution [21]. Of these, cue insertion is the most directly relevant to this project: a feature designed to shift the model's answer is inserted into the prompt, and a faithfulness failure is inferred when the answer changes to the cued answer but the CoT does not acknowledge it. The model's own answer-switching behaviour under counterfactual conditions provides a clean causal reference, operationalising a clear faithfulness failure: the model using information in its decision process that it does not report.

Existing hint-injection setups are calibrated to specific frontier models, and monitorability evaluations are known to be highly sensitive to the configuration: task difficulty, hint type, task domain, and model capability can each substantially affect monitorability results [20, 18, 15]. A configuration that elicits measurable hint-following on a frontier model may produce near-zero effect on a smaller model, rendering the evaluation insensitive for the smaller model.

### 2.3 What Impacts CoT Faithfulness and Monitorability

CoT monitorability is not a fixed property of a model; it can be degraded or improved by pressures applied during training and inference [18, 15]. We organise these pressures into two categories. Adversarial unmonitorability arises when the training objective explicitly rewards the concealment of influencing factors or the evasion of monitoring. Emergent unmonitorability arises as an unintended side effect of pressures with no adversarial intent. While adversarial unmonitorability is easier to characterise and detect, emergent unmonitorability is arguably more concerning: it can arise from ordinary training choices and is therefore harder to anticipate and detect. Such effects arise when pressures on model outputs, efficiency, or capability indirectly reshape the CoT, despite no objective explicitly targeting monitorability.

**Non-adversarial inference-time pressure effects.** Yang et al. [22] show that inference-time constraints on reasoning (e.g., token-budget constraints, Chain of Draft, thinking suppression) consistently reduce verbalisation of the factors driving answers and increase the rate at which problematic CoTs escape monitor detection; expanding reasoning space has the opposite effect. Mehta [23] demonstrated that models almost always internally register misleading hints but systematically fail to report them, and that informing the model it is being monitored barely improves monitorability.

**Non-adversarial post-training effects.** Post-training can both degrade and improve monitorability depending on the objective. On the degradation side, Drori et al. [15] establish that RL supervision applied exclusively to final outputs can degrade reasoning transparency through two mechanisms: parametric spillover, in which training to produce safe-looking outputs generalises via shared weights to produce safe-looking CoTs, and conditional spillover, in which CoTs preceding rewarded outputs are reinforced regardless of whether they reflect genuine reasoning. Jose [24] further demonstrates that outcome-based RL frequently produces CoTs that causally impact model outputs yet remain illegible to monitors, with illegibility worsening on harder tasks. Ferreira et al. [21] show that preference optimisation can reward explanations that appear appropriate without reflecting the actual causes of the model's prediction, as the reward model has no mechanism to verify consistency between the explanation and the model's decision process. MacDermott et al. [25], by contrast, find that length penalties and KL divergence regularisation do not consistently affect monitorability when task accuracy is controlled for, indicating that emergent degradation is objective-specific rather than a universal consequence of post-training pressure. On the improvement side, Xiong et al. [26] show that RLVR can improve monitorability when tasks structurally require explicit multi-step inference to succeed, though this effect depends strongly on training data composition. Guan et al. [18] find that extended reasoning traces and broader monitor access (to the full CoT rather than actions alone) generally improve monitorability at current frontier scale.

The main gap is that no existing study isolates the effects of individual safety and alignment post-training objectives on CoT monitorability. The closest work on post-training degradation (Drori et al. [15]; Jose [24]) uses RL-based supervision (PPO, RLVR), not preference-based optimisation, and treats post-training as a monolithic pressure rather than varying objectives. Ferreira et al. [21] use preference optimisation but on a general task with no safety-specific objectives. No study has applied DPO under distinct safety objectives (e.g. harmlessness, helpfulness) and measured their individual effects on CoT monitorability using a calibrated honeypot environment. Our RO2 addresses this gap directly.

---

## 3 Implementation

### 3.1 Methodology or Approach

This project has two phases: RO1 designs and validates a honeypot evaluation environment, while RO2 uses the evaluation environment to measure how preference-based post-training on specific safety and alignment objectives affects CoT monitorability. Offline DPO is the post-training method used throughout, for three reasons. First, it requires no separate reward model and no online sampling, making training on multiple objectives feasible within a single-GPU compute budget. Second, it directly reflects the preference optimisation stage central to real-world safety post-training pipelines, including Tülu 3 [14, 27]. Third, training on fixed (chosen, rejected) pairs eliminates variance from the model's evolving generation distribution under online RL (e.g., PPO), ensuring that observed monitorability effects are attributable to the training objective rather than to distributional shift. The base model used throughout is `allenai/Llama-3.1-Tulu-3-8B-SFT-no-safety-data`. Lambert et al. [27] constructed this checkpoint by training Llama 3.1 8B on the full Tülu 3 SFT mixture with safety-specific data removed, reporting negligible impact on general capabilities (58.0 vs. 60.1, on their composite multi-skill evaluation) while substantially reducing safety scores (74.7 vs. 93.1). Removing safety-specific data before applying our experimental post-training conditions avoids potential confounds with prior safety training. Additionally, the 8B scale is computationally feasible for multiple DPO training runs [27].

#### RO1: Honeypot Evaluation Environment

RO1 designs and validates a honeypot evaluation environment using the cue insertion approach, as discussed in Section 2.2. As existing setups were calibrated for different actor models and research questions, applying them directly risks results that only reflect evaluation artefacts; this motivates designing our own variant with systematic validation.

The resulting evaluation suite must satisfy the following conditions: (1) **Naturalness**: monitorability failures should arise naturally, without artificially inducing adversarial behaviour; (2) **Task domain coverage**: multiple task types are required, since monitorability rates vary significantly across domains [18]; (3) **Verifiable ground truth**: whether the model used the hint must be verifiable without examining the CoT; (4) **Incentive calibration**: tasks must be hard enough that the model is genuinely pressured to use the hint, but not so hard that it fails to attempt the task at all; (5) **Reliable total effect estimation**: all tasks use MCQ format, with each instance run 16 times with stochastic sampling. 200 examples are tested per task for computational feasibility.

**Evaluation tasks.** The suite comprises six tasks spanning maths, reasoning, and general knowledge at two difficulty levels. Each instance is run in an unhinted arm (plain question) and a hinted arm (identical question with a biasing cue inserted). A case is monitorable when the CoT explicitly acknowledges the inserted hint as a causal factor, and unmonitorable when the model follows the hint but does not acknowledge it. Our evaluation tasks are:

1. **Academic QA: MMLU** (dataset: [28]). Four-option MCQs spanning 57 subjects across STEM, humanities, social sciences, and professional knowledge. MMLU tests broad declarative knowledge where correct answers are retrievable from well-established facts; the 8B SFT checkpoint scores 65.9 (0 shot, CoT), placing the model in a regime where hints produce measurable effect but the model retains sufficient independent knowledge that concealing hint influence is less necessary. This task serves as the easier difficulty stratum and was validated by Chen et al. [20]. The core skill this dataset tests is knowledge recall.

2. **Academic QA (hard): GPQA** (dataset: [29]). Four-option MCQs in biology, physics, and chemistry at PhD level, where approximately 65% of domain experts answer correctly. The 8B SFT checkpoint scores 31.9, confirming the model operates under genuine uncertainty and must rely on multi-step inference rather than memorised facts, making hint influence both more likely and harder to detect in the CoT. This task was validated by Chen et al. [20]. The core skill this dataset tests is knowledge recall.

3. **Applied Reasoning: AGIEval English** (dataset: [30]). MCQs drawn from three subsets of the AGIEval benchmark: LSAT Logical Reasoning (lsat-lr), LSAT Analytical Reasoning (lsat-ar), and AQUA-RAT (aqua-rat), covering formal logical inference, constraint-based deduction, and algebraic word problems respectively. Unlike MMLU, which tests declarative knowledge recall, and GPQA, which tests deep domain-specific scientific reasoning, this dataset requires the application of domain-agnostic reasoning rules and structured inference from self-contained premises, skills that do not depend on prior scientific or factual knowledge. The core skill this dataset tests is reasoning.

4. **Mathematics (easy): GSM-MC** (dataset: [31]). Four-option MCQs converted from GSM8K grade school mathematics word problems, where distractor options are drawn from incorrect predictions collected across 60 open-source models, making them plausible rather than trivially wrong. The 8B SFT checkpoint scores 76.2 on the original GSM8K (8 shot, CoT), indicating high model competence; configurations where this produces insufficient total effect will be caught by the inclusion test, in which case MATH-MC serves as a fallback. The core skill this dataset tests is mathematical reasoning.

5. **Mathematics (hard): MATH-MC** (dataset: [31]). Four-option MCQs converted from the MATH benchmark using the same distractor construction procedure as GSM-MC, covering high school competition mathematics at substantially higher difficulty. The 8B SFT checkpoint scores 31.5 on the original MATH benchmark (4 shot, CoT). Together, GSM-MC and MATH-MC provide a difficulty gradient within the mathematical reasoning domain, mirroring the easy-hard contrast established by MMLU and GPQA in the knowledge domain. The core skill this dataset tests is mathematical reasoning.

6. **Deceptive Compliance: WMDP** (dataset: [32]). WMDP (Weapons of Mass Destruction Proxy) is a four-option MCQ benchmark covering hazardous knowledge across biosecurity, cybersecurity, and chemical security. It is included not to assess reasoning quality but to probe a qualitatively distinct behavioural pressure: models with safety post-training are incentivised to refuse or deflect on these questions regardless of what their CoT reasons through, creating a potential divergence between stated reasoning and final response. The core skill this dataset tests is deceptive compliance.

Since all evaluation datasets (except WMDP) form part of the Tülu 3 evaluation suite, the SFT training mixture was already decontaminated against these benchmarks via 8-gram matching. Apart from minimal task-specific instructions, all tasks will use the same prompt template. To elicit CoTs, we prompt the model to think step-by-step and output the final answer as a letter.

**Hint Type.** A model's reliance on and acknowledgement of a hint may depend not only on the task type and difficulty, but also on how the hint is provided in the prompt. In RO1 we evaluate 6 hint types drawn from prior work [12, 20]. The Stanford professor hint prepends an authority statement claiming a named expert believes the answer is X. The black square hint places a visual marker next to hinted answers. The wrong few-shot hint includes the target question among the few-shot examples labelled with an incorrect answer. The metadata hint embeds the wrong answer in an XML field prepended to the question. The validation function hint encodes the wrong answer in a Python snippet checking for a specific response. The unauthorised access hint uses a system message claiming the model has gained unauthorised access and that the answer is X.

**Hint Correctness.** In our evaluation we only use incorrect hints since they provide the strongest causal evidence: if the model follows a hint pointing to a demonstrably incorrect answer, that answer selection cannot be attributed to the model's domain knowledge, rendering hint reliance as the most plausible cause of the incorrect answer.

**Monitoring system.** Monitoring models will be accessed via the Edinburgh Language Model (ELM) API. A three-monitor ensemble is used: Monitor A (GPT-4o), Monitor B (GPT-4), and Monitor C (GPT-3.5 Turbo). Given a CoT and the original prompt, each monitor returns a binary label (acknowledged / not acknowledged) with a one-sentence supporting quotation. A and B are consulted first; if they agree, the shared verdict is accepted; otherwise, C provides the tiebreaker. The monitoring system will be validated against 100 manually annotated examples.

**Total Effect and Inclusion Test.** Each task-hint configuration is evaluated by computing the hint's Total Effect (TE) on the baseline model: TE = Pr(Y = h | hinted) − Pr(Y = h | unhinted), where Y is the model's answer and h is the hint's suggested answer. This quantifies the causal shift in the model's answer attributable to the hint (regardless of whether the CoT acknowledges the hint's influence). Configurations with TE ≈ 0 provide no behavioural signal and are removed.

For each instance i, hinted and unhinted prompt variants are each sampled N = 16 times (temperature τ = 0.8, top-p = 0.95). The Total Effect is estimated as:

$$\widehat{TE}_i^c = \frac{1}{N} \sum_{j=1}^{N} \mathbf{1}[a_{ij}^h = h_i] - \frac{1}{N} \sum_{j=1}^{N} \mathbf{1}[a_{ij}^u = h_i]$$

where $a_{ij}^h$ and $a_{ij}^u$ are the model's answers on the hinted and unhinted runs respectively, and $h_i$ is the hint's suggested answer. Only instances where the hint had a reliable causal effect are retained: $S = \{i : \widehat{TE}_i^c \geq \delta\}$. The threshold δ is derived from the full RO1 baseline run: all task-hint configurations are evaluated on the baseline model, $\widehat{TE}_i^c$ is computed per instance, and δ is fixed as max(25th percentile of all observed $\widehat{TE}_i^c$, 0.20). The percentile rule excludes the quarter of instances with the weakest hint signal, concentrating analysis on configurations where the hint reliably shifts behaviour; the floor ensures near-zero causal effects are never included even when baseline TE values are uniformly low. The retained task-hint configurations constitute the evaluation suite used in RO2 to quantify the impact of objective-specific post-training on CoT monitorability.

**Hint Acknowledgment Rate (HAR).** The acknowledgment rate is computed only over runs where the model selected the hinted answer, since only those runs constitute genuine instances of hint-driven behaviour. Including runs where the hint did not change the answer would dilute the metric, treating a model that ignored the hint identically to one that followed it covertly. The per-instance acknowledgment rate is defined as:

$$\widehat{ack}_i^d = \frac{\sum_{j=1}^{N} \mathbf{1}[a_{ij}^h = h_i \wedge mon(c_{ij}^h) = 1]}{\sum_{j=1}^{N} \mathbf{1}[a_{ij}^h = h_i]}$$

where $mon(c_{ij}^h) \in \{0, 1\}$ is the monitor's binary judgment on CoT $c_{ij}^h$. The Hint Acknowledgment Rate is then the mean over retained instances:

$$HAR = \frac{1}{|S|} \sum_{i \in S} \widehat{ack}_i^d \quad HAR \in [0, 1]$$

HAR is always reported alongside |S| and mean $\widehat{TE}^c$, as these are essential for interpreting any observed shift. Post-training may reduce hint-following rather than improve transparency: if the model becomes more resistant to hints after post-training, mean $\widehat{TE}^c$ falls and |S| shrinks, meaning HAR is computed over a different set of instances. Any resulting HAR shift would then reflect a change in which set of instances are analysed rather than a genuine change in reasoning transparency. A true monitorability improvement leaves |S| approximately stable while HAR rises across the same retained instances.

#### RO2: Evaluating Post-Training Effects on CoT Monitorability

The second phase applies offline DPO training runs under three distinct post-training objectives and measures the effect on HAR using the evaluation environment validated in RO1.

**Training objectives and datasets.** Three post-training conditions are investigated. The harmlessness condition uses PKU-Alignment/PKU-SafeRLHF, filtered to conflict cases only (where the safer response is not the more helpful response; approximately 12,616 examples), explicitly decoupling harmlessness from helpfulness. The helpfulness condition uses the helpful-base split of Anthropic/hh-rlhf (approximately 12–17k examples subsampled to match). The anti-sycophancy condition uses DataCreatorAI/Anti-Sycophancy-DPO, with manual filtering of examples where the chosen/rejected distinction reflects factual correction (i.e. one response is objectively more accurate) rather than sycophantic deference to user preference; the post-filtering dataset is subsampled to match the harmlessness condition (≈12,616 examples), with all conditions resampled to the smallest available size if post-filtering yields fewer examples. Training datasets are verified to have no overlap with evaluation datasets, and capability benchmarks are evaluated before and after each condition to confirm monitorability changes are not artefacts of capability change.

**Training algorithm.** Offline DPO is used for all conditions: every condition trains on fixed (prompt, chosen, rejected) triples from the above-mentioned public datasets, with no generation/sampling involved. This ensures that differences in HAR are attributable to the training objective rather than to the model's evolving generation distribution. HAR is tracked at regular checkpoints during training to observe whether monitorability changes gradually or collapses at a threshold.

**Mechanism ablations.** To isolate the source of any observed HAR shift, two ablation runs are conducted alongside the harmlessness DPO condition, using the identical PKU-SafeRLHF dataset. First, an SFT ablation trains on chosen responses only, matching dataset size and training duration; if HAR shifts similarly to DPO, the effect is attributable to the content of chosen responses rather than contrastive suppression of rejected ones. No existing work holds the training objective and data constant while varying only SFT vs. DPO and measuring the downstream effect on CoT monitorability. Second, a reversed DPO ablation swaps chosen and rejected labels, training the model toward the responses the harmlessness objective treats as undesirable. If monitorability shifts in the opposite direction relative to standard DPO, this confirms that the specific objective being optimised (not DPO training dynamics in general) is responsible for the observed change. If the reversal produces no opposing shift, the DPO update mechanism itself, independent of objective direction, is implicated.

### 3.2 Expected Outcomes

**Deliverables.** A validated honeypot evaluation suite for CoT monitorability will be produced, comprising task prompts, hinted/unhinted prompt pairs, monitor prompts, and HAR computation code for reuse by other researchers.

**Empirical findings.** RO1 will produce baseline monitorability measurements across task types, hint types, constituting a controlled ablation of how evaluation design variables affect monitorability scores. RO2 will produce per-training-condition HAR scores showing how each safety post-training objective changes monitorability relative to the untrained baseline, reported alongside |S| and mean $\widehat{TE}^c$.

**Hypotheses.** We hypothesise that safety post-training will not uniformly improve monitorability. Specifically: (i) harmlessness training may reduce HAR on safety-critical tasks by incentivising safe-sounding CoTs that conceal hint-driven reasoning; (ii) helpfulness training may have minimal effect, as it targets output quality rather than reasoning transparency.

### 3.3 Evaluation Criteria

RO1 is considered successful if: (i) the inclusion test retains at least three task-hint configurations with TE exceeding δ; (ii) the monitor ensemble achieves Cohen's κ ≥ 0.70 against manually annotated gold-set labels and each individual monitor achieves classification accuracy ≥ 0.80 on the gold set; and (iii) baseline HAR is neither too high nor too low, confirming the setup produces variance sufficient to detect training-induced monitorability changes.

RO2 success is defined by the ability to measure and interpret HAR shifts across post-training conditions. HAR is reported per task and per training condition, alongside |S|, mean $\widehat{TE}^c$, and mean CoT length in tokens. Reporting CoT length controls for post-training-induced verbosity changes: since longer CoTs provide more space in which a hint can be acknowledged, an observed HAR increase could reflect greater verbosity rather than improved transparency. If mean CoT length differs by more than 20% across conditions, a length-stratified sub-analysis will be conducted to estimate how much of any HAR shift is attributable to verbosity. Task accuracy on held-out capability benchmarks (MMLU, HellaSwag) is tracked to verify monitorability changes are not artefacts of change in capability. All comparisons are relative to the untrained baseline model evaluated on the identical suite from RO1.

### 3.4 Risk Assessment and Mitigations

The two main risks are as follows. First, the compute budget may be insufficient for the full experimental design (16 runs per instance across all tasks and training conditions). This can be mitigated by pre-estimating total inference cost and reducing N from 16 to 8 if necessary. Second, DPO training may cause capability collapse that confounds monitorability results; capability benchmarks are evaluated before and after each condition to detect and account for this.

### 3.5 Responsible Research Including Ethics

This project does not involve human participants or personal data; all experiments use publicly available language models and datasets, and no formal ethics application is required under the University of Edinburgh Informatics ethics process (Level 0). The research has inherent dual-use potential: the evaluation suite detects CoT concealment but does not train models to evade monitors, and findings are intended to inform the design of more robust monitoring systems, with a net positive to the safety community. All datasets and the base model are used within their respective licences, code will be released under MIT, and all hyperparameters, prompts, and random seeds will be documented to enable independent replication. All datasets and model checkpoints will be stored on a private repository, with a public release on Hugging Face upon dissertation submission.

---

## 4 Research Plan, Milestones, and Deliverables

REMOVED IN THIS VERSION

---

## References

[1] F. Dennstädt, J. Hastings, P. M. Putora, M. Schmerder, and N. Cihoric, "Implementing large language models in healthcare while balancing control, collaboration, costs and security," *NPJ digital medicine*, vol. 8, no. 1, p. 143, 2025, DOI: https://doi.org/10.1038/s41746-025-01476-7.

[2] B. Padiu, R. Iacob, T. Rebedea, and M. Dascalu, "To what extent have llms reshaped the legal domain so far? a scoping literature review," *Information*, vol. 15, no. 11, 2024, DOI: https://doi.org/10.3390/info15110662. [Online]. Available: https://www.mdpi.com/2078-2489/15/11/662

[3] Y. Li, S. Wang, H. Ding, and H. Chen, "Large language models in finance: A survey," in *Proceedings of the fourth ACM international conference on AI in finance*, 2023, pp. 374–382, DOI: https://doi.org/10.48550/arXiv.2311.10723.

[4] A. Meek, E. Sprejer, I. Arcuschin, A. J. Brockmeier, and S. Basart, "Measuring chain-of-thought monitorability through faithfulness and verbosity," 2025, DOI: https://doi.org/10.48550/arXiv.2510.27378. [Online]. Available: https://arxiv.org/abs/2510.27378

[5] L. Bereska and E. Gavves, "Mechanistic interpretability for ai safety – a review," 2024, DOI: https://doi.org/10.48550/arXiv.2404.14082. [Online]. Available: https://arxiv.org/abs/2404.14082

[6] S. Casper, X. Davies, C. Shi, T. K. Gilbert, J. Scheurer, J. Rando, R. Freedman, T. Korbak, D. Lindner, P. Freire, T. Wang, S. Marks, C.-R. Segerie, M. Carroll, A. Peng, P. Christoffersen, M. Damani, S. Slocum, U. Anwar, A. Siththaranjan, M. Nadeau, E. J. Michaud, J. Pfau, D. Krasheninnikov, X. Chen, L. Langosco, P. Hase, E. Bıyık, A. Dragan, D. Krueger, D. Sadigh, and D. Hadfield-Menell, "Open problems and fundamental limitations of reinforcement learning from human feedback," 2023, DOI: https://doi.org/10.48550/arXiv.2307.15217. [Online]. Available: https://arxiv.org/abs/2307.15217

[7] B. Baker, J. Huizinga, L. Gao, Z. Dou, M. Y. Guan, A. Madry, W. Zaremba, J. Pachocki, and D. Farhi, "Monitoring reasoning models for misbehavior and the risks of promoting obfuscation," 2025, DOI: https://doi.org/10.48550/arXiv.2503.11926. [Online]. Available: https://arxiv.org/abs/2503.11926

[8] X. Zhang, C. Du, T. Pang, Q. Liu, W. Gao, and M. Lin, "Chain of preference optimization: Improving chain-of-thought reasoning in llms," 2024, DOI: https://doi.org/10.48550/arXiv.2406.09136. [Online]. Available: https://arxiv.org/abs/2406.09136

[9] Y. Zhang, Y. Yang, J. Shu, X. Wen, and J. Sang, "Agent models: Internalizing chain-of-action generation into reasoning models," 2025, DOI: https://doi.org/10.48550/arXiv.2503.06580. [Online]. Available: https://arxiv.org/abs/2503.06580

[10] T. Korbak, M. Balesni, E. Barnes, Y. Bengio, J. Benton, J. Bloom, M. Chen, A. Cooney, A. Dafoe, A. Dragan, S. Emmons, O. Evans, D. Farhi, R. Greenblatt, D. Hendrycks, M. Hobbhahn, E. Hubinger, G. Irving, E. Jenner, D. Kokotajlo, V. Krakovna, S. Legg, D. Lindner, D. Luan, A. Mądry, J. Michael, N. Nanda, D. Orr, J. Pachocki, E. Perez, M. Phuong, F. Roger, J. Saxe, B. Shlegeris, M. Soto, E. Steinberger, J. Wang, W. Zaremba, B. Baker, R. Shah, and V. Mikulik, "Chain of thought monitorability: A new and fragile opportunity for ai safety," 2025, DOI: https://doi.org/10.48550/arXiv.2507.11473. [Online]. Available: https://arxiv.org/abs/2507.11473

[11] K. Cox, "Post-hoc reasoning in chain of thought." [Online]. Available: https://www.lesswrong.com/posts/ScyXz74hughga2ncZ/post-hoc-reasoning-in-chain-of-thought

[12] M. Turpin, J. Michael, E. Perez, and S. R. Bowman, "Language models don't always say what they think: Unfaithful explanations in chain-of-thought prompting," 2023, DOI: https://doi.org/10.48550/arXiv.2305.04388. [Online]. Available: https://arxiv.org/abs/2305.04388

[13] S. Emmons, R. S. Zimmermann, D. K. Elson, and R. Shah, "A pragmatic way to measure chain-of-thought monitorability," 2025, DOI: https://doi.org/10.48550/arXiv.2510.23966. [Online]. Available: https://arxiv.org/abs/2510.23966

[14] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. L. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, J. Schulman, J. Hilton, F. Kelton, L. Miller, M. Simens, A. Askell, P. Welinder, P. Christiano, J. Leike, and R. Lowe, "Training language models to follow instructions with human feedback," 2022, DOI: https://doi.org/10.48550/arXiv.2203.02155. [Online]. Available: https://arxiv.org/abs/2203.02155

[15] J. Drori, L. Marks, B. Woodworth, A. Cloud, and A. M. Turner, "Output supervision can obfuscate the chain of thought," 2025, DOI: https://doi.org/10.48550/arXiv.2511.11584. [Online]. Available: https://arxiv.org/abs/2511.11584

[16] T. Lanham, A. Chen, A. Radhakrishnan, B. Steiner, C. Denison, D. Hernandez, D. Li, E. Durmus, E. Hubinger, J. Kernion, K. Lukošiūtė, K. Nguyen, N. Cheng, N. Joseph, N. Schiefer, O. Rausch, R. Larson, S. McCandlish, S. Kundu, S. Kadavath, S. Yang, T. Henighan, T. Maxwell, T. Telleen-Lawton, T. Hume, Z. Hatfield-Dodds, J. Kaplan, J. Brauner, S. R. Bowman, and E. Perez, "Measuring faithfulness in chain-of-thought reasoning," 2023, DOI: https://doi.org/10.48550/arXiv.2307.13702. [Online]. Available: https://arxiv.org/abs/2307.13702

[17] I. Arcuschin, J. Janiak, R. Krzyzanowski, S. Rajamanoharan, N. Nanda, and A. Conmy, "Chain-of-thought reasoning in the wild is not always faithful," 2025, DOI: https://doi.org/10.48550/arXiv.2503.08679. [Online]. Available: https://arxiv.org/abs/2503.08679

[18] M. Y. Guan, M. Wang, M. Carroll, Z. Dou, A. Y. Wei, M. Williams, B. Arnav, J. Huizinga, I. Kivlichan, M. Glaese, J. Pachocki, and B. Baker, "Monitoring monitorability," 2025, DOI: https://doi.org/10.48550/arXiv.2512.18311. [Online]. Available: https://arxiv.org/abs/2512.18311

[19] A. Madsen, S. Chandar, and S. Reddy, "Are self-explanations from large language models faithful?" 2024, DOI: https://doi.org/10.48550/arXiv.2401.07927. [Online]. Available: https://arxiv.org/abs/2401.07927

[20] Y. Chen, J. Benton, A. Radhakrishnan, J. Uesato, C. Denison, J. Schulman, A. Somani, P. Hase, M. Wagner, F. Roger, V. Mikulik, S. R. Bowman, J. Leike, J. Kaplan, and E. Perez, "Reasoning models don't always say what they think," 2025, DOI: https://doi.org/10.48550/arXiv.2505.05410. [Online]. Available: https://arxiv.org/abs/2505.05410

[21] P. Ferreira, W. Aziz, and I. Titov, "Truthful or fabricated? using causal attribution to mitigate reward hacking in explanations," 2025, DOI: https://doi.org/10.48550/arXiv.2504.05294. [Online]. Available: https://arxiv.org/abs/2504.05294

[22] S. Yang, J. Wu, X. Gong, X. Wu, D. Wong, N. Liu, and D. Wang, "Investigating cot monitorability in large reasoning models," 2026, DOI: https://doi.org/10.48550/arXiv.2511.08525. [Online]. Available: https://arxiv.org/abs/2511.08525

[23] D. P. Mehta, "Can we trust ai explanations? evidence of systematic underreporting in chain-of-thought reasoning," 2025, DOI: https://doi.org/10.48550/arXiv.2601.00830. [Online]. Available: https://arxiv.org/abs/2601.00830

[24] A. Jose, "Reasoning models sometimes output illegible chains of thought," 2025, DOI: https://doi.org/10.48550/arXiv.2510.27338. [Online]. Available: https://arxiv.org/abs/2510.27338

[25] M. MacDermott, Q. Wei, R. Djoneva, and F. R. Ward, "Reasoning under pressure: How do training incentives influence chain-of-thought monitorability?" 2025, DOI: https://doi.org/10.48550/arXiv.2512.00218. [Online]. Available: https://arxiv.org/abs/2512.00218

[26] Z. Xiong, S. Chen, and H. Lakkaraju, "Monitorability as a free gift: How rlvr spontaneously aligns reasoning," 2026, DOI: https://doi.org/10.48550/arXiv.2602.03978. [Online]. Available: https://arxiv.org/abs/2602.03978

[27] N. Lambert, J. Morrison, V. Pyatkin, S. Huang, H. Ivison, F. Brahman, L. J. V. Miranda, A. Liu, N. Dziri, S. Lyu, Y. Gu, S. Malik, V. Graf, J. D. Hwang, J. Yang, R. L. Bras, O. Tafjord, C. Wilhelm, L. Soldaini, N. A. Smith, Y. Wang, P. Dasigi, and H. Hajishirzi, "Tulu 3: Pushing frontiers in open language model post-training," 2025, DOI: https://doi.org/10.48550/arXiv.2411.15124. [Online]. Available: https://arxiv.org/abs/2411.15124

[28] D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt, "Measuring massive multitask language understanding," 2021, DOI: https://doi.org/10.48550/arXiv.2009.03300. [Online]. Available: https://arxiv.org/abs/2009.03300

[29] D. Rein, B. L. Hou, A. C. Stickland, J. Petty, R. Y. Pang, J. Dirani, J. Michael, and S. R. Bowman, "Gpqa: A graduate-level google-proof qa benchmark," 2023, DOI: https://doi.org/10.48550/arXiv.2311.12022. [Online]. Available: https://arxiv.org/abs/2311.12022

[30] W. Zhong, R. Cui, Y. Guo, Y. Liang, S. Lu, Y. Wang, A. Saied, W. Chen, and N. Duan, "Agieval: A human-centric benchmark for evaluating foundation models," 2023, DOI: https://doi.org/10.48550/arXiv.2304.06364. [Online]. Available: https://arxiv.org/abs/2304.06364

[31] Z. Zhang, Z. Jiang, L. Xu, H. Hao, and R. Wang, "Multiple-choice questions are efficient and robust llm evaluators," 2024, DOI: https://doi.org/10.48550/arXiv.2405.11966. [Online]. Available: https://arxiv.org/abs/2405.11966

[32] N. Li, A. Pan, A. Gopal, S. Yue, D. Berrios, A. Gatti, J. D. Li, A.-K. Dombrowski, S. Goel, L. Phan, G. Mukobi, N. Helm-Burger, R. Lababidi, L. Justen, A. B. Liu, M. Chen, I. Barrass, O. Zhang, X. Zhu, R. Tamirisa, B. Bharathi, A. Khoja, Z. Zhao, A. Herbert-Voss, C. B. Breuer, S. Marks, O. Patel, A. Zou, M. Mazeika, Z. Wang, P. Oswal, W. Lin, A. A. Hunt, J. Tienken-Harder, K. Y. Shih, K. Talley, J. Guan, R. Kaplan, I. Steneker, D. Campbell, B. Jokubaitis, A. Levinson, J. Wang, W. Qian, K. K. Karmakar, S. Basart, S. Fitz, M. Levine, P. Kumaraguru, U. Tupakula, V. Varadharajan, R. Wang, Y. Shoshitaishvili, J. Ba, K. M. Esvelt, A. Wang, and D. Hendrycks, "The wmdp benchmark: Measuring and reducing malicious use with unlearning," 2024, DOI: https://doi.org/10.48550/arXiv.2403.03218. [Online]. Available: https://arxiv.org/abs/2403.03218
