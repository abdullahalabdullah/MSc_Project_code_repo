# CoT Suppression in RO2 SFT — Investigation Notes

**Date:** 2026-06-20
**Model:** `Qwen/Qwen3-4B` (LoRA SFT, MVP-RO2)
**Data:** PKU-SafeRLHF conflict cases → safer (chosen) responses
**Status:** Root issue identified; the masking fix proved **necessary but not sufficient**. Real fix (reasoning distillation) not yet built.

---

## 1. The observed problem

After SFT, the fine-tuned model **stops producing chain-of-thought (CoT)**. In Qwen3
thinking mode it no longer emits a `<think>…</think>` block — it answers directly.

This was first reported qualitatively ("model stops generating CoT after SFT") and later
**confirmed quantitatively** with a per-epoch think-token measurement over the eval set:

| Model | Mean think tokens (200 eval prompts, greedy) | `frac_non_empty_think` | CoT retained vs base |
|---|---|---|---|
| **base Qwen3-4B** | **413.5** | 1.000 | — (reference) |
| epoch 1 (`checkpoint-305`) | **0.0** | 0.000 | **0.0%** |
| epoch 2 (`checkpoint-610`) | **0.0** | 0.000 | **0.0%** |
| epoch 3 (`checkpoint-915`) | **0.0** | 0.000 | **0.0%** |

Generation sanity check on epoch 3 (`has <think>: False`):
- *"What is 17 × 23? Show your reasoning."* → answers `391` directly, no `<think>` tag, then loops.
- *"A train travels 60 km in 45 minutes…"* → `The average speed of the train is 80 km/h.` directly, no `<think>` tag.

Collapse is **complete and immediate** — already 0% at epoch 1, on all 200 prompts.

---

## 2. What we did (timeline)

1. **Diagnosed the original mechanism.** Qwen3's chat template injects an empty
   `<think>\n\n</think>` block into the assistant turn. Verified empirically that this
   happens for a complete assistant turn (`add_generation_prompt=False`) in **both**
   `enable_thinking=True` and `False` — they render byte-for-byte identically. So the
   `enable_thinking` training flag is effectively a **no-op** for our data; only the
   masking matters.

2. **Fixed the masking (`sft/data.py`).** The original code masked only up to the
   assistant header, leaving the empty `<think>\n\n</think>` in the **supervised** region.
   The fix folds the empty think block into the **masked** scaffold (label `-100`) so only
   the safe answer after `</think>` is supervised. Verified against the real Qwen3-4B
   template that the supervised region contains no `<think>`/`</think>` tokens.

3. **Added a guard.** New assertion in `verify_masking()`:
   `assert "<think>" not in kept_text and "</think>" not in kept_text` — turns a silent
   re-introduction of the bug into a hard failure. Confirmed it fires on a simulated leak.

4. **Built quantitative CoT tooling.** `sft/think_stats.py` + `scripts/05_think_token_eval.py`:
   run the base model on the held-out eval set to store a CoT baseline, then re-run on any
   checkpoint to report "% of base CoT retained." `sft/generate.py` refactored to share a
   `count_think_tokens()` helper.

5. **Config + orchestration.** Bumped train batch to effective 32 (`32×1`, dynamics
   unchanged), kept gradient checkpointing on (safe). Wrote `run_all.sh`: token preflight
   (real write-probe) → seq preflight → sanity gate → base CoT baseline → training →
   per-epoch CoT verification → push.

6. **Ran the full pipeline.** Sanity gate passed all 6 checks (incl. push). Training ran
   3 epochs (~39 min, 915 steps, train_loss 3.35 → 1.43). Then measured CoT → **0% retained.**

---

## 3. What we assumed — and what proved wrong

### Assumption (WRONG): "Masking the empty `<think>` block will preserve CoT."

The reasoning was: the empty-think tokens were the suppression signal because they sat in
the supervised region; mask them (no gradient) and the model's reasoning distribution stays
untouched. We even verified the masking is correct and the supervised region is clean.

**Why it was wrong:** the masking removed the most *explicit* cause (training the model to
emit empty think tokens), but the model **still** collapsed to 0% CoT. Masking the empty
block was **necessary but not sufficient.**

### What actually proved out (the deeper, correct explanation)

The training targets are **bare, direct answers** — which is exactly Qwen3's *non-thinking*
response format. Fine-tuning ~9,731 examples × 3 epochs on direct-answer targets teaches the
model that the correct response to these prompts is "answer immediately, don't reason." Two
contributing facts:

- The answer is always conditioned on an **empty** think context during training, so the
  model never learns to produce an answer *after real reasoning*.
- **LoRA edits shared weights** (q/k/v/o/gate/up/down across all layers), including at the
  position where the model decides whether to open a `<think>` block. So even though the
  think tokens received **no gradient**, the model's *preference* to emit `<think>` at all
  was driven to ~0.

Notably the model does **not** reproduce the training format (empty `<think></think>` then
answer) — it drops the `<think>` tag **entirely**. This is consistent with general
fine-tuning drift overriding the thinking behavior, not with direct supervision of the
think tokens.

**Conclusion:** *answer-only SFT suppresses CoT regardless of masking.* This is a data
problem, not a hyperparameter problem (0% at epoch 1 rules out LR/epoch tuning as a fix).

### Secondary assumptions, also clarified along the way

- **"`enable_thinking=True` during training would change the logits / help."** Wrong — it
  renders identical tokens for our complete-turn data, so identical logits. SFT is
  teacher-forcing: it grades fixed text and never generates, so it cannot "reason during
  training." The only way to get reasoning into training is to put reasoning **text** in the
  data (distillation) or move to a generate-then-train loop (rejection sampling / RFT).
- **Dataset size.** The current PKU-SafeRLHF revision yields **10,813** conflict cases, not
  the spec's ~12,616. Non-fatal; record the real number.

---

## 4. What is correct / validated (don't redo these)

- Masking is correct and guarded (assertion). Keep it — it's still required.
- Token preflight (write-probe), sanity gate, training, baseline tooling, and push all work.
- The CoT measurement is a clean, reportable instrument: it caught the suppression exactly
  as designed (base 413.5 → 0.0).
- The 3 pushed adapters (`er-abd/qwen3-4b-sft-harmlessness-pku-adapter-epoch{1,2,3}`) are
  **CoT-suppressed** — usable only as a **negative-result baseline**, not for a
  CoT-preserving eval.

---

## 5. The real fix (planned — option A: reasoning distillation)

Train on data that is already in **thinking format**, then mask the reasoning and supervise
only the safe answer, so the training distribution matches thinking-mode inference:

1. For each PKU conflict prompt, generate the base model's own `<think>…</think>` reasoning
   in thinking mode.
2. Pair that reasoning with the **PKU safe answer** (ideally rejection-sample so the
   reasoning is consistent with arriving at a *safe* conclusion).
3. Store `prompt → <think>{reasoning}</think> {safe answer}`.
4. Run the **same masked SFT** (mask the reasoning, supervise the answer) — the existing
   masking/training/eval infrastructure is reused unchanged.

Expected effect: the model learns "reason, then give the safe answer" instead of "skip
reasoning," preserving CoT while injecting the harmlessness behavior. Validate with the same
`scripts/05_think_token_eval.py` baseline comparison (target: retention near ~100%).

---

## 6. Key takeaway (one line)

> Masking the empty `<think>` block stops the model from being *explicitly taught* to emit
> empty thinking, but **answer-only SFT still collapses CoT** because the supervised targets
> are direct, non-reasoning answers. Preserving CoT requires reasoning **in the training
> data**, not a template flag and not just masking.
