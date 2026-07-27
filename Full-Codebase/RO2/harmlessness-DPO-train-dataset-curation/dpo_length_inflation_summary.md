# DPO length inflation — diagnosis and resolution

**Project:** harmlessness DPO on SFT'd Qwen3-4B (`er-abd/harmless-dpo-qwen3-4B`)
**Date:** 2026-07-12
**Outcome:** Root cause identified (EOS collapse from policy squeezing, not verbosity bias). Fix requires **zero additional training compute** — ship `step-147`.

---

## 1. The original question

> "Standard DPO teaches the model to generate longer responses. Is there a variant that avoids this?"

The premise needed unpacking. DPO length bias has **two independent sources**:

| Source | Mechanism | Fixed by |
|---|---|---|
| **Mechanical (objective-level)** | Implicit reward `r = β·Σ_t log[π_θ/π_ref]` is a *sum* over tokens, so reward is extensive in length. Optimizer can lower loss by making chosen sequences longer. | Length normalization (SimPO, LN-DPO, `ld_alpha`) |
| **Data-level confound** | In most preference datasets chosen is longer than rejected, so "longer = better" is genuinely in the labels. | Fixing the data / length-controlled eval |

No method removes both. Objective-level variants only address the first.

**Key candidates discussed:**
- **SimPO** (Meng et al. 2024) — length-normalized reward `(β/|y|)·log π`, drops π_ref, adds target margin γ.
- **R-DPO** (Park et al. 2024) — explicit penalty ∝ `(|y_w| − |y_l|)`.
- **LD-DPO** (`ld_alpha` in TRL) — down-weights tail tokens beyond the shorter completion in each pair. Single sweepable knob; α=1 recovers vanilla DPO.
- **IPO** — *not* a length method. Fixes overfitting on near-deterministic preferences via squared loss; TRL's implementation happens to length-normalize as a side effect, and it redefines `beta`. Changing two things at once = bad experimental hygiene.

**On SFT vs DPO:** SFT on chosen responses inherits only the *data-level* confound — it matches the length distribution of its targets and does not have DPO's runaway amplification dynamic. DPO can push length *beyond* what's in the data; SFT is bounded by imitation.

---

## 2. The data — reversed confound

The preference pipeline report showed the **opposite** of the standard setup:

- **Rejected is longer in 78% of pairs** (chosen mean 1,069 tokens, rejected 1,468, Δ +399).
- Visible-answer words: chosen 360, rejected 504 (Δ +144, rejected longer in 71%).
- `Pearson(cost margin, answer-word Δ) = +0.424`.

So the data's length pressure points toward **brevity**, not verbosity. This ruled out the data-level confound as the driver.

### Two data-quality issues flagged (still unresolved)

1. **The chosen set is enriched for short refusals.** Among safe responses, length vs. Beaver cost has **r = +0.49** (longer safe answers score *worse*); among rejected, r = −0.04. Selection is `argmin(cost)` with a **median runner-up gap of only 0.438** — so length is effectively the tiebreaker. Mechanistically: a short safe answer is a flat refusal; a long safe answer engages with the topic and accrues cost. **Risk: over-refusal, not verbosity.**
   - *Suggested fixes (not yet applied):* length-matched chosen selection among Qwen3Guard-Safe candidates; or residualize cost on length; set `MIN_MARGIN` (currently unset).

2. **The think block is in the DPO loss but was never labeled.** Chosen think mean 451 words, rejected 568 — and most of the +399-token gap lives here. Gradients run over ~450–570 words/response of unsupervised reasoning.
   - *Suggested fix (not yet applied):* mask the think block from the DPO loss, computing it only over the labeled visible answer.

---

## 3. The evidence — what training actually did

### Vanilla DPO (β=0.1, `loss_type="sigmoid"`), W&B curves

- `rewards/chosen`: **0 → −9**
- `rewards/rejected`: 0 → −17
- `rewards/margins`: 0 → ~10
- `rewards/accuracies`: ~0.85 by step ~150 → ~0.93 by step 735

**`rewards/chosen` going to −9 is the whole finding.** At β=0.1 that implies `log[π_θ(y_w)/π_ref(y_w)] ≈ −90 nats` — the model made **its own training targets** ~e⁻⁹⁰ less likely. Both sides collapsed; the margin grew only because rejected fell faster. This is the **squeezing effect**.

Note the timing: **preference signal was essentially learned by step ~150** (accuracy already 0.85+, chosen-reward still ≈ 0). Everything after ~step 300 bought margin by destroying the policy.

### Out-of-domain length inflation (unhinted, median tokens)

| Dataset | Base | step-735 | Ratio |
|---|---:|---:|---:|
| mmlu_pro | 1,049 | 1,350 | 1.29× |
| gpqa_diamond | 4,299 | 5,203 | 1.21× |
| wmdp | 810 | 1,018 | 1.26× |
| codemmlu | 1,150 | 1,485 | 1.29× |
| lsat_ar | 4,704 | 5,335 | 1.13× |
| lsat_lr | 959 | 1,180 | 1.23× |
| medxpertqa | 1,408 | 1,906 | 1.35× |
| lexam | 1,671 | 2,237 | 1.34× |

A **harmlessness** dataset moved GPQA, LSAT and MMLU-Pro lengths by 13–35%. None of these are in the training distribution. That is a **KL-drift symptom**, not something a training-pair reweighting could cause.

---

## 4. The mechanism — failure to terminate

**squeezing → chosen *and* rejected log-probs collapse → EOS (`<|im_end|>`) probability collapses → model fails to stop → outputs get longer**

Every rejected response ends in EOS, so every gradient step suppressing rejected text also suppresses EOS. Normally the chosen side (pushed *up*) offsets this. Here chosen was pushed *down too*, so nothing propped EOS back up.

This distinguishes two stories that both produce "longer outputs" but need **opposite** tools:

| Story | What's happening | Fix |
|---|---|---|
| **Verbosity bias** | Model chooses to write more; still terminates cleanly | Length normalization (SimPO / `ld_alpha`) |
| **Failure to terminate** | Model *can't* stop; EOS mass destroyed | **β / early stopping** — restore the KL anchor |

**Truncation rate is the diagnostic** (fraction hitting `max_tokens` without emitting EOS) — and it is *more* informative than mean length, because mean length is understated by capping.

---

## 5. The length-normalized run — and why it made things worse

`er-abd/harmless-dpo-norm-qwen3-4B`, SimPO-style, **β=5**, reference-free.

**Training curves: identical collapse.** Final `rewards/chosen ≈ −9`, `rewards/rejected ≈ −17`, margin ≈ 10, accuracy ≈ 0.93 — indistinguishable from vanilla. Only the *timing* differed (held flat until ~step 200, then collapsed along the same path). Length normalization removed the token-summation lever, and the collapse happened anyway.

**Evaluation (step-438, gpqa_diamond/professor, unhinted):**

| | base | step-735 (standard) | step-438 (length-norm) |
|---|---:|---:|---:|
| unhinted mean, valid (tokens) | 4,806 | 5,518 | **6,705** |
| truncation rate | 0.98% | 1.64% | **9.53%** |

**Longer, not shorter — and ~6× the truncation.** This *confirmed* failure-to-terminate directly (the report noted truncated transcripts are typically unparseable, no answer extracted).

### Why β=5 was not "50× larger" — it was ~20× more aggressive *per token*

The report flagged `beta=5` vs `beta=0.1` as "a 50x difference." That framing is wrong, and the correction explains the result:

- **Standard DPO:** `r = β · Σ_t log[π_θ/π_ref]`. β=0.1, chosen-reward −9 → **−90 nats total** over ~1,000 tokens ≈ **−0.09 nats/token**.
- **Length-normalized:** `r = (β/|y|) · Σ_t log[π_θ/π_ref]` = β × *mean per-token* log-ratio. β=5, chosen-reward −9 → **−1.8 nats/token**.

**~20× more drift per token.** Normalization shrinks the reward, so a large β is needed to compensate — and β=5 massively overcompensated. Compounding it: **SimPO is reference-free**, so the KL anchor restraining drift was gone entirely. Harder push, no leash.

**⚠️ This was therefore not a fair test of length normalization.** It tested "much stronger drift pressure with no reference model." Also, step-438 sits well past the collapse (the norm run's `rewards/chosen` is ≈ 0 only until step ~200).

---

## 6. The resolution — step-147

`gpqa_diamond` / `professor`, hinted arm:

| | base | **step-147** | step-735 |
|---|---:|---:|---:|
| hinted mean (tokens) | 5,058 | **5,328 (+5.3%)** | 5,878 (+16.2%) |
| hint-followed mean | 5,569 | 5,746 | 6,307 |
| hint-not-followed mean | 4,577 | 4,932 | 5,508 |
| followed − not-followed gap | 992 (19.6%) | 814 (15.3%) | 799 (13.6%) |
| **truncation rate** | 0.98% | **0.63%** ✅ | 1.64% |

**step-147 terminates cleanly — better than base.** The pathology is absent.

Dose-response across all checkpoints, tracking `rewards/chosen` exactly:

| checkpoint | `rewards/chosen` | truncation | length vs base |
|---|---:|---:|---:|
| base | — | 0.98% | — |
| **step-147** | **≈ 0** | **0.63%** ✅ | +5.3% |
| step-735 | −9 | 1.64% | +16.2% |
| norm step-438 | −9 | **9.53%** 🔴 | +38% |

Where the policy hasn't collapsed, EOS is intact. Where it has, EOS falls apart.

**The residual +5.3% at step-147 is ordinary mild verbosity** — the model writes slightly more and still stops properly. It is not the pathology (step-735 = 3× the inflation *plus* broken EOS; step-438 = 7× *plus* 10× the truncation).

---

## 7. Decision

**Ship `step-147`.** Zero additional training compute.

Remaining checks:
1. Confirm **harmlessness** holds at step-147 vs step-735.
2. **Benign-prompt over-refusal check** (XSTest-style, ~100 prompts is enough). Given the short-refusal-enriched chosen set (§2), a harmlessness score that improves *because the model refuses everything* is not a harmlessness improvement — and only this eval catches it.
3. Optionally evaluate **norm step ~200** (chosen-reward ≈ 0, accuracy ≈ 0.90 — the widest clean window of any checkpoint).

**Do not run another objective variant.** Two independent results show the objective isn't the problem: LN-DPO squeezed to the same `rewards/chosen ≈ −9`, and its checkpoint came back *longer* with 6× the truncation.

**If compute frees up, in priority order:**
1. One run: standard DPO (`sigmoid`, keep π_ref), **β = 0.5**, capped at ~200 steps.
2. Mask the think block from the loss (§2).
3. Fix chosen selection: length-matched or cost-residualized, set `MIN_MARGIN` (§2). Target `|Pearson(margin, length Δ)| < 0.1`.
4. Only *then* is `ld_alpha` legitimate — at step-147 termination is clean, so the residual +5.3% really is verbosity rather than EOS collapse, which is exactly the regime that knob is for. Sweep {1.0, 0.7, 0.5, 0.2}.

---

## 8. Reporting notes

**Report hint effects as ratios or log-length, never raw token deltas.** A global length shift contaminates absolute deltas. Example from step-735:

| | absolute Δ (hinted − unhinted) | ratio (hinted / unhinted) |
|---|---|---|
| gpqa · unethical | base +362 → DPO +588 (+62%) | 1.075 → 1.107 |
| lsat_lr · unethical | base +554 → DPO +897 (+62%) | 1.374 → 1.496 |
| codemmlu · unethical | base +832 → DPO +978 (+18%) | 1.406 → **1.363 ↓ (sign flips)** |

Most of the apparent "DPO amplifies hint-taking" signal is global length scaling. At step-147's +5.3% this matters much less — a 5% shift cannot manufacture or erase an 814-token gap — but use ratios regardless.

**Two real, non-artifactual findings:**
- The **followed − not-followed gap shrinks monotonically** (992 → 814 → 799; 19.6% → 15.3% → 13.6% of mean). This moves *opposite* to the inflation, so it is not a length artifact.
- **Always report truncation rate alongside mean length.** Mean is downward-biased by capping (step-438's 6,705 excludes 151 responses that would have run longer).

**Claims the data supports** (vs. the weaker "SimPO didn't work"):
1. The length inflation is **failure to terminate** (truncation 1.64% → 9.53%), not verbosity preference.
2. It is driven by **policy collapse away from π_ref** (`rewards/chosen → −9` in *both* runs).
3. The SimPO config **amplified** that collapse (~20× per-token drift, no KL anchor) rather than fixing it.
4. Both are avoided entirely by **stopping before the collapse** — preference accuracy is already 0.85+ at step ~150, before `rewards/chosen` leaves zero.
