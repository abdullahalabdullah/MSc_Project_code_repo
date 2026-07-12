# SFT Arm — Developer Specification

**Project:** Impact of Safety & Alignment Post-Training on CoT Monitorability
**Component:** SFT training arm (harmlessness objective first; anti-sycophancy and helpfulness to follow the same recipe)
**Status:** Approved for implementation
**Audience:** Developer implementing training + supporting analysis. This document explains *what* to build and *why*. Implementation details (framework, code structure) are your call unless stated otherwise.

---

## 1. Background and Goal

### 1.1 What we have already

We have completed offline **DPO (with LoRA)** training of **Qwen3-4B (thinking mode)** on on-policy preference data for the harmlessness objective (and anti-sycophancy is done/underway). The preference data was curated as follows:

1. Prompts drawn from WildJailbreak (plain harmful + adversarial).
2. The **base model itself** generated 16 candidate responses per prompt.
3. Judges scored **only the final answer** (never the CoT): WildGuard for safe/unsafe labelling, beaver-7b-unified-cost for harmlessness ranking, Qwen3Guard for final validation.
4. Prompts where all 16 responses were uniformly safe or uniformly unsafe were dropped.
5. The safest response became `chosen`, the most harmful became `rejected`.

We then measure CoT monitorability of the trained model vs the base model on our frozen cue-insertion evaluation suite (g-mean², TE, accuracy, CoT length).

### 1.2 What we are adding and why

DPO's gradient does two things simultaneously: it **pushes up** the probability of the chosen response and **pushes down** the probability of the rejected response. The rejected response contains a full CoT, so DPO applies a *negative* gradient to reasoning tokens even though no judge ever evaluated the CoT.

We want to know whether the monitorability effects we observe from DPO are caused by that **negative gradient**, or whether they appear under the mildest possible form of output-level supervision. To isolate this, we train a second model with **SFT on the chosen responses only** — i.e., DPO with the negative term removed — and run the identical monitorability evaluation.

This is a **controlled ablation**. Its entire scientific value depends on the SFT arm being identical to the DPO arm in every respect except where necessary e.g. the loss function, the learning rate.....

Note for framing/write-up: SFT on best-of-n filtered self-generated responses is not a strawman baseline — it is rejection-sampling fine-tuning (RAFT/RFT), a real stage in real industrial pipelines (e.g., Llama post-training). We compare two realistic methods.

### 1.3 What success looks like

Three outcomes are acceptable; one is not:

- SFT reaches a safety gain comparable to DPO → we compare monitorability at matched safety. Ideal.
- SFT gives a real but smaller safety gain → we compare monitorability at *equal safety dose* using intermediate checkpoints of both arms. Also fine, and itself a finding ("the negative gradient is load-bearing").
- SFT produces essentially **no** safety gain at any reasonable setting → we report SFT as an ineffective treatment. The monitorability numbers for that model are then **not interpretable** and must not be presented as a monitorability result.
- **Not acceptable:** running the monitorability eval and reporting its numbers without first establishing whether the SFT training changed safety behaviour at all.

---

## 2. Data: Converting the DPO Dataset to an SFT Dataset

### 2.1 Primary dataset (the strictly-matched arm)

The conversion is deliberately trivial, because triviality is what preserves the ablation:

1. Start from the **exact, final, post-Qwen3Guard-validation DPO pair file** that was fed to DPO training. Not an earlier intermediate. If there is any ambiguity about which file that is, resolve it before writing a line of code.
2. For each pair `(prompt, chosen, rejected)`:
   - **Keep** `prompt` and `chosen`.
   - **Drop** `rejected` entirely.
3. Each SFT example is therefore `(prompt, target = chosen)`, where `chosen` is the **full response including its CoT** (the `<think>…</think>` block and the final answer), byte-for-byte as it appeared in the DPO data.
4. Do **not** re-filter, re-judge, re-rank, deduplicate, truncate, or re-sample. Same examples, same count, same order is fine to shuffle at train time, but the *set* must be identical to the DPO chosen set.

Record the SHA/hash of the source DPO file and the derived SFT file in the run metadata so the provenance is auditable.

### 2.2 Prompt formatting — highest-risk area, be paranoid

The single most likely way to silently break this experiment is a formatting mismatch. Requirements:

- The tokenised training sequence (chat template, system prompt if any, role tags, `<think>` handling, EOS placement) must be (where possible) **byte-identical** to what the DPO trainer produced for the chosen sequence, and consistent with what the evaluation harness sends at inference.
- **Loss masking:** the loss must be computed on the **response tokens only** (CoT + final answer). Prompt tokens masked out. This mirrors DPO, which computes log-probs over response tokens only.
- The EOS token at the end of `chosen` must be included in the loss (otherwise the model doesn't learn when to stop — see §5 on length).


---

## 3. Training Configuration

### 3.1 Matched by default

Hold everything identical to the DPO run unless this spec says otherwise:

- Base model & mode: Qwen3-4B, thinking mode, same revision/checkpoint of the base weights.
- LoRA config: same rank, alpha, dropout, target modules as the DPO run.
- Same max sequence length, same precision, same seed policy (fix and record seeds).
- Optimiser family and schedule: start from the DPO run's settings (only make changes to this where needed e.g. where the DPO choices are not good/optimal for SFT).
- Epochs: start with the same number as DPO. It might be increased, but only later.
- Somethings must change, otherwise SFT will not work e.g. learning rate. To make a choice about this, consult literature. 

### 3.2 Checkpointing — mandatory

Save intermediate checkpoints at the same cadence used in the DPO run (or denser). The core analysis is a **dose–response curve**: monitorability plotted against achieved safety gain across checkpoints, for both arms. Without intermediate SFT checkpoints that analysis is impossible. Treat missing checkpoints as a failed run.

### 3.3 During-training monitoring

- log Training loss, obviously.


---

## 4. Evaluation Protocol and Gates

Run these **in order**.

### Gate 1 — Treatment check: did safety actually improve?
Run the **existing** safety and refusal classifiers (same ones used to validate the DPO model) on the same held-out safety evaluation set, for: base model, and final SFT model. Also compute the DPO checkpoints' curve on the same set if not already done — we need both arms on the same safety axis.

- Meaningful safety gain (even if smaller than DPO's) → proceed to Gate 3.
- No measurable gain → **stop.** Report to the project lead; decision will be made on Fallback A/B or harder training. Do not run the monitorability suite on a null-treatment model — it burns a full eval budget for uninterpretable numbers.

### Gate 2 — Full monitorability evaluation
You do not need to code this -- we already have the code.
Run the frozen evaluation suite exactly as it was run for base and DPO models: all datasets × all hint types × control/intervention arms × N=8 samples, same monitor (GPT-5-mini), same instance-level and configuration-level filters, same metrics (TE, Reffect, g-mean², accuracies, CoT length). **Zero changes to the suite** — it was validated and frozen before training for exactly this reason.

Then run the suite on the SFT checkpoints selected for the dose–response analysis (coordinate with the lead on which checkpoints, since each is a full eval bill).

### 4.1 Confound instrumentation (report alongside every result)

- **CoT length** per dataset, per arm — first-class outcome next to g-mean², for detecting length-mediated effects. (already done in the eval suite)
- **Response diversity**: self-distillation sharpens the output distribution; report a simple diversity/self-consistency measure across the N=8 samples (e.g., answer-choice entropy per instance) for base vs SFT vs DPO.
- **Over-refusal** on the eval suite: rate of unparseable/refused answers per arm. A rise in refusals shifts TE and accuracy for reasons unrelated to monitorability. (already done in the eval suite)
