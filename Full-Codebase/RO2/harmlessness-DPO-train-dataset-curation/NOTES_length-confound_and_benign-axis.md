# Notes — length/style confound & the benign-axis extension

_Captured 2026-07-04. Context: discussion about curating harmlessness DPO pairs from WildJailbreak._

---

## ⚠️ YET TO BE FIXED LATER (the two open gaps)

**1. No over-refusal counterbalance (benign prompts).**
Every prompt is `*_harmful`, so every pair is "refuse (chosen) vs comply (rejected)." DPO on
only-harmful prompts pushes the model to refuse everything — the classic safety-tuning failure
(over-refusal / helpfulness collapse, the thing XSTest exists to catch). A usable harmlessness
set needs a benign/borderline slice where the polarity flips: chosen = helpful compliance,
rejected = unnecessary refusal. The code even has this as a known TODO — it parses and stores
WildGuard's `refusal` field "for the future benign axis" but never uses it. **This is the
single biggest gap.** → fix = the benign axis (details in §3, §6, §7).

**2. The length/style confound is measured but never mitigated.**
Stage 05 reports that rejected is longer in ~100% of pairs and computes `Pearson(margin, length)`
— but nothing acts on it. Since chosen = short refusal and rejected = long completion, a naive
DPO run learns "shorter ⇒ safer" instead of harmlessness. There's a second confound too: chosen
has a much longer `<think>` block than rejected (your own diagnostic showed it). Missing:
length-balanced selection, a max-length-asymmetry drop filter, or at least a length-controlled
preference check. Diagnosing without a mitigation knob leaves the confound in the data.
→ the real fix is the benign axis (§3); a length-asymmetry drop filter is a cheap stopgap.

---

## TL;DR / current decision

- **Going ahead with the pipeline AS-IS** (harmful-only: `vanilla_harmful` + `adversarial_harmful`)
  as the **instrumented baseline / ablation**. This is deliberate, not final.
- The **benign axis is deferred** but is the real fix for the length/style confound and for
  over-refusal. This doc captures what it needs so we don't have to re-derive it.
- **Plan:** train harmful-only first (with measurement) → observe the confound / collapse →
  add benign axis → show the improvement. Baseline → fix → compare is a stronger thesis arc
  than pre-fixing a problem we never demonstrated.

---

## 1. The length/style confound (the problem)

In the current pipeline every pair is "refuse (chosen) vs comply (rejected)" on a harmful
prompt, so:

- `chosen`  = safety refusal → **short** visible answer
- `rejected` = harmful completion → **long** visible answer
- Stage-05 diagnostics already show rejected is longer in ~**100%** of pairs.

Because "short", "refusal-style", and "safe" are ~perfectly collinear in the data, **length
is a near-perfect predictor of the label**. This is a *data identifiability* problem, not a
loss problem: from these pairs alone, no objective can tell whether the preference is about
safety, terseness, or refusal phrasing.

Also note the CoT twist (we keep the full `<think>…</think>` in both responses): our own
diagnostics showed `chosen` has **longer think** (~76w) but **shorter answer** (~14w), and
`rejected` the reverse (think ~30w, answer ~200w). So the confound has both a length axis and
a CoT axis. The RM/classifier only ever saw the *visible answer*, yet DPO trains on the full
CoT — so we'd be reinforcing reasoning traces we never safety-checked.

**Status in code:** the confound is *measured* (stage 05: `Pearson(cost margin, answer-word Δ)`,
`% rejected longer`, think-vs-answer lengths) but **never mitigated**.

## 2. Why length-normalized DPO (SimPO / R-DPO) is NOT the fix

There are two different "length" problems:

- **A — length exploitation during optimization** (what SimPO / R-DPO fix). DPO's implicit
  reward is a *sum* of per-token log-ratios, so it scales with token count; length-normalized
  variants divide by `|y|` to stop the optimizer gaming length. Real, but not our problem.
- **B — length/style confounded with the label in the data** (what we have). Loss-level
  normalization removes the *magnitude* dependence on token count but does nothing about the
  **style** dimension (refusal phrasing, structure), which stays perfectly correlated with the
  label. You can't length-normalize your way out of information the dataset doesn't contain.

=> Length-normalized DPO is at best a complementary/belt-and-suspenders defense. The real fix
is at the **data-curation layer** (this pipeline).

## 3. Why the benign categories ARE the fix

WildJailbreak has 4 `data_type`s: {vanilla, adversarial} × {harmful, benign}. Adding the two
**benign** types flips the length/style polarity:

| prompt type | chosen (preferred) | rejected |
|---|---|---|
| harmful | refusal — **short** | harmful compliance — long |
| benign  | helpful answer — **long** | needless refusal — short |

So across the combined set, "chosen is shorter" no longer predicts the label — it's short on
harmful prompts and long on benign ones. That breaks the **marginal** length↔label correlation
at the source, which a loss fix cannot do. `adversarial_benign` is the most valuable slice: it
*looks* like a jailbreak but is benign, so it kills the "looks-like-a-jailbreak ⇒ refuse"
shortcut. This is the same thing as the "benign counterbalance" and it **subsumes** the
length-confound mitigation.

## 4. Risk of training as-is (harmful-only)

Real but degree-dependent, governed by β (KL strength), epochs, dataset size, and how OOD the
eval is:

- **Most likely first symptom: over-refusal** on benign/borderline prompts (the canary —
  probe with XSTest-style prompts).
- **Secondary risk: general terseness leaking to non-safety tasks** (math/reasoning), because
  "shorter" is a prompt-agnostic direction. On math the *refusal* component is unlikely to
  fire (prompt isn't harmful-looking), so what leaks there is mainly verbosity/terseness.
- **CoT-specific risk:** could shorten/suppress the visible answer's elaboration on reasoning
  tasks. NB: connects to our earlier **CoT-collapse under answer-only SFT** finding — this
  would be a *second, different* (preference-induced) collapse mechanism; nice thesis parallel.
- Bounded by the DPO KL anchor to `π_ref`, so catastrophic OOD drift is limited but not zero.

## 5. Instrumentation to run WITH the as-is baseline (do this first)

The as-is run is only worth it if we measure the collapse. Minimum:

1. **Length distributions** (think-length + visible-answer length) of DPO model vs base, on
   neutral non-safety prompts — a leftward shift is the smoking gun.
2. **GSM8K / MATH**: accuracy **and** response length (+ think length separately).
3. **Over-refusal probe** (XSTest-style benign prompts) — moves before math does.
4. Keep the run small (LoRA, few epochs), **same base + hyperparameters** we'll reuse for the
   benign comparison, so the later A/B is clean.

---

## 6. Benign-axis change map (for when we build it)

Bounded work: thread an `axis ∈ {harmful, benign}` field everywhere + branch in 3 places.
The plumbing is clear; the only genuinely open decision is the benign ranking signal (§7).

**Why it isn't just "widen stage 01":** on a benign prompt every response is "Safe" (refusing
a benign request isn't harmful, just unhelpful), so the harm signal (Beaver cost + WildGuard
`Harmful response`) can't separate chosen from rejected. The benign discriminator is
**refusal vs compliance**, gated by **helpfulness**.

| Stage | Change | Size |
|---|---|---|
| `config.py` | Add `vanilla_benign`/`adversarial_benign` to `WILDJAILBREAK_TYPE_TO_COLUMN`; add a helpfulness scorer id (if used); optional balance flag. | small |
| `01_prepare_prompts` | Track `axis ∈ {harmful, benign}` per prompt (today only `subset ∈ {vanilla, adversarial}`). Stratify/balance the sample across axes. Update mix reporting. | small–med |
| `02_generate_responses` | Carry the new `axis` field onto each response dict (like `subset`). Generation unchanged. | trivial |
| `03_classify_guard` | `refusal` is **already parsed & stored** (the hook). **filter-1 branches on axis**: harmful → both guard-labels present (as now); benign → both **refusal** values present. | medium |
| `04_score_cost` | Harmful → Beaver as-is. Benign → Beaver is the wrong signal; select `chosen` = best *compliant* answer, `rejected` = a *refusal*. Needs the §7 decision. | **medium–large (crux)** |
| `05_report` | **filter-3 branches on axis**: harmful → `chosen=Safe & rejected=Unsafe` (as now); benign → `chosen: refusal=no (+ still Safe) & rejected: refusal=yes`. Add per-axis funnel + report the **length↔label correlation across the combined set** (the whole point). "Unsafe cost > Safe cost" sanity check applies to harmful pairs only. | medium |

**Unchanged / reusable:** generation, WildGuard/Beaver loading & batching, diagnostics infra,
export format.

## 7. The one real open decision (benign ranking signal, stage 04)

How to rank benign responses to pick `chosen`:

1. **Rule-based (cheapest):** `chosen` = any compliant (`refusal=no`), `rejected` = any refusal
   (`refusal=yes`). Needs a tiebreak among compliant responses (quality heuristic / random).
   Risk: a "helpful" answer that's low-quality or subtly harmful.
2. **Helpfulness RM (cleaner, recommended for the final artifact):** add a helpfulness reward
   model; `chosen` = argmax helpfulness among compliant + Safe, `rejected` = a refusal. Mirrors
   the harmful path's within-prompt RM ranking; costs a second model in the stack.

Lean **#2** for a defensible thesis result, **#1** as a fast first version.

**Smaller follow-on decisions:**
- **Safety guard on benign `chosen`:** require it to still be WildGuard-Safe (so an
  `adversarial_benign` wrapper can't smuggle a technically-harmful "helpful" answer). Recommend yes.
- **Balance:** not just equal counts — aim for *length-symmetric* contributions so the confound
  actually cancels; verify via the combined length↔label correlation afterwards.
- **Refusal reliability:** we'd lean on WildGuard's single `refusal` flag as the benign
  discriminator — decide if that's enough or add a second signal.

---

## 8. Other DPO-curation gaps noted (separate from the benign axis)

Not blocking, but worth remembering:

- ~~No standard DPO export~~ **RESOLVED**: stage 06 (`06_export_dpo_dataset.py`) now exports a
  HF `DatasetDict({train,valid})` with `prompt`/`chosen`/`rejected` to `outputs/<run>/dpo_dataset/`.
  The "strip vs keep CoT" decision defaults to **keep** (`EXPORT_FIELD=full`); set
  `EXPORT_FIELD=answer` to export the visible post-`</think>` answer instead.
- **No train/val/test split** beyond train/valid (stage 06 splits by `VALID_FRACTION`, no held-out
  test split yet) **or eval decontamination.**
- **Selection is `argmin/argmax` cost** — outlier/RM-error prone; no coherence/degeneracy/
  min-length quality floor on `chosen`.
- **No confidence gating** (filter-4 margin is off; WildGuard confidence unused in the main path).
- Prompt dedup is exact-match only (no near-duplicate/semantic dedup); no per-example
  provenance / dataset card; final vanilla/adversarial mix isn't balanced.
