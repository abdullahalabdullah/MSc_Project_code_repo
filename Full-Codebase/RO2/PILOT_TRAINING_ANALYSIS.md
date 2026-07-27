# Pilot Training — Signal Analysis & HP-Validation Decisions

Analysis of the small-scale pilot DPO run (`pilot_hp`, Qwen3-4B LoRA DPO on 780 train / 87
valid pairs from the 4,500-prompt pilot curation). Purpose: validate the training
hyperparameters from the training signals **before** committing to the full-scale run.

> Scope: these are **optimization-health signals** (loss, reward-accuracy, margins, grad-norm,
> memory) — they say whether the HPs are good, **not** whether the model is actually safer or
> more monitorable. The behavioural evaluation (harmlessness + CoT monitorability) is a
> separate offline step run on the pushed checkpoints.

---

## Run summary

| Field | Value |
|---|---|
| Run name | `pilot_hp` |
| Dataset | 780 train / 87 valid pairs (pilot curation, `SAMPLE=4500`) |
| Model | Qwen/Qwen3-4B, LoRA (r=16, α=32, dropout 0.05) |
| Config | LR 5e-6, cosine, warmup 0.1, β 0.1, 3 epochs, batch 1×16, MAX_LENGTH 12288 |
| Total optimizer steps | 147 |
| Best checkpoint | **step 147 (final)** — reward-accuracy 0.874, loss 0.360 |
| Peak GPU memory | 75.16 / 85.02 GB (9.9 GB headroom) |

---

## The trajectory (all eval points)

| Step | Epoch | eval_loss | reward_acc | margin |
|-----:|------:|----------:|-----------:|-------:|
| 0    | 0.00  | 0.6931    | 0.000      | 0.000  |
| 29   | 0.59  | 0.5348    | 0.839      | 0.416  |
| 58   | 1.18  | 0.4281    | 0.828      | 1.209  |
| 87   | 1.78  | 0.3910    | 0.862      | 1.658  |
| 116  | 2.37  | 0.3733    | 0.851      | 1.820  |
| 147  | 2.96  | 0.3602    | 0.874      | 1.891  |

**Best checkpoint: step 147 (final), reward-accuracy 0.874, loss 0.360.**

### What this tells you

- **Loss falls monotonically 0.693 → 0.360**, smoothly, no oscillation or spikes. LR 5e-6 is
  well-tuned — no instability, no signs of being too low (it moves decisively from step 0).
- **Reward-accuracy jumps to 0.84 by the first eval (step 29) and settles at 0.87** — on
  held-out pairs the model assigns higher reward to the safe response 87% of the time. Strong,
  healthy; DPO is clearly learning the preference.
- **Margins grow steadily 0.42 → 1.89** (the chosen–rejected reward gap widening) — the desired
  shape.
- **No overfitting.** eval_loss is still gently decreasing at the final step (0.373 → 0.360) and
  the best checkpoint is the last step. The curve is flattening but has not turned up.

---

## Supporting signals

### Grad-norm / LR (per-step training log)

Grad-norm: 16.5 during warmup → settles flat at ~8–11 for the rest, **no spikes, no NaN/Inf**.
LR ramped 3e-6 → 4.99e-6 (warmup) then cosine-decayed to ~5e-8 — exactly the intended schedule.

### W&B eval curves

- `eval/rewards/margins` rises smoothly 0 → ~1.9 and flattens at the end (steps 116→147 barely
  move) — textbook DPO saturation; signals 3 epochs is near diminishing returns.
- `eval/rewards/chosen` (0 → −3.5) and `eval/rewards/rejected` (0 → −5.4) both go negative
  (normal for DPO); rejected falls faster — the model learns mainly by **pushing down harmful
  responses** rather than pushing up safe ones (expected for harmlessness).
- Throughput panels (`steps_per_second`, `samples_per_second`, `runtime`) are eval *speed*, not
  quality — ignore for any scientific conclusion (they drift as CUDA caches warm up).

### Per-step train curves

- `train/loss` 0.68 → ~0.30 (noisy but monotone down); `train/rewards/accuracies` 0.4 → ~0.9.
- Tail wobble (steps ~120–140) in margins/accuracy/logps is **benign**: cosine LR tail
  (~5e-8) + small-batch noise. Eval over the same steps was smooth and still improving, and
  grad-norm stayed flat — real instability would show in eval too, and it doesn't.
- The per-step jitter is the visible cost of the small effective batch (16) — not a problem,
  just the one empirical argument that effective batch 32 *could* be tried (via
  `gradient_accumulation_steps 32`, which halves optimizer steps — not worth it here).

---

## HP-validation decisions

### 1. Is LR 5e-6 right? → **Yes, keep it.**

Three independent checks pass:
- **Loss slope**: monotonic 0.693 → 0.360, decisive early (not too low), no oscillation (not
  too high).
- **Grad-norm stability**: 16.5 (warmup) → flat ~8–11, no spikes, no NaN/Inf.
- **Warmup behaved**: 3e-6 → 4.99e-6 → cosine to ~5e-8.

A higher LR might reach the plateau faster, but warmup grad-norms of 16 say stability headroom
isn't large, and there's no accuracy left to chase. **No change.**

### 2. Can the full run use batch 2×8? → **No. Keep 1×16.**

Peak reserved **75.2 / 85.0 GB — only 9.9 GB headroom (88% full).** Doubling per-device batch
roughly doubles activation memory (another concatenated chosen+rejected pair at up to 12,288
tokens) → exceeds 10 GB → **OOM**. The high memory comes from `MAX_LENGTH=12288` (worst-case
allocation) × DPO holding chosen+rejected together. It ran fine (no OOM), but there is no room
to grow the batch and **no free speedup** for the full run. **No change (memory-forced).**

### 3. Is 3 epochs right? → **Yes, keep 3.**

- **No overfitting**: eval_loss never turns up (still falling at the final step).
- **Best checkpoint = final step**, accuracy 0.874.
- **Diminishing returns**: per-eval loss deltas −0.158, −0.107, −0.037, −0.018, **−0.010**;
  margins plateaued (1.82 → 1.89 over the last third).

Because best = the final step, a 4th epoch might extract a sliver more — but I recommend
**against** it: gains are near-zero, and over-training raises the risk of exploiting the length
shortcut (r=0.40 confound) and degrading CoT, which would confound the monitorability study.
**No change.**

### 4. Were MAX_TOKENS / MAX_LENGTH sized right? → **Yes, both.**

`check_max_tokens.py` verdict: **ABOUT RIGHT.** Truncation 0.26% over 45,000 responses
(negligible waste), median completion 1,236 tokens, and the thin p99.9 tail *does* reach 8,192
(budget not over-provisioned). Trainer side: p99 completion 2,911 tokens; worst case 12,288 =
MAX_LENGTH fits ("0 spare" is a pathological case that never actually occurs). **No change.**

### 5. Real survival rate & exact top-up SAMPLE

**Survival = 19.3%** (867/4,500); per-subset adversarial 20.0%, vanilla 18.2%. Have 867 pairs;
need ~4,130 more for ~5,000 total:

- 4,130 ÷ 0.193 ≈ 21,400 prompts; +15% safety margin → **`SAMPLE ≈ 24,500`**.
- At 19.3% → ~4,730 new pairs → **~5,600 total** (comfortable overshoot). Clears 5,000 even at
  the pessimistic 18%. Overshoot is free; undershoot means another pod.

Top-up command (128K prompts remain after excluding the pilot's 4,500):

```bash
SAMPLE=24500 EXCLUDE_RUNS=outputs/pilot RUN_OUTPUT_DIR=outputs/main bash run_all.sh
```

### 6. Does the machinery work end-to-end? → **Yes — code flawless; one process gap.**

Everything the code drives worked:
- ✅ Preflight (HF auth + gated WildJailbreak access) passed before GPU time.
- ✅ Step-0 baseline captured (loss 0.6931, acc 0.0) — clean pre-training control.
- ✅ ~20% eval schedule fired at steps 0, 29, 58, 87, 116, 145, 147.
- ✅ All 7 checkpoints pushed to HF (`...-step-0/29/58/87/116/145/147`).
- ✅ W&B logged (the curves).
- ✅ `eval_metrics.jsonl` created and auto-pushed (in `er-abd/harmless-dpo-qwen3-4B-pilot`).
- ✅ `gpu_memory.json` created (its number, 75/85 GB, echoed into `logs/train.log`).

**Process gap (not code):** the manual `hf upload` backups of `outputs/pilot` and the training
dir never ran (the `hf` CLI 401'd — `hf auth login` was never completed). Nothing critical was
lost this time (metrics auto-pushed; the memory number survived in the log), but on the full
run that gap would lose the ~5,600-pair dataset. **Fix for next pod:** run
`hf auth login --token ...` right after setup, verify with `hf auth whoami`, then backups run.

---

## Net decision

**Change nothing in the training config.** LR, β, epochs, batch, LoRA, MAX_TOKENS/MAX_LENGTH
are all validated by a clean, monotonic, overfitting-free DPO curve ending at 87%
reward-accuracy. Action items:

1. Set `SAMPLE=24500` for the top-up curation (`EXCLUDE_RUNS=outputs/pilot`).
2. Complete `hf auth login` on the next pod so the `hf upload` backups actually run.
3. Carry the **length confound (r=0.40)** into the behavioural eval — the training signals
   can't tell whether the model learned "safer" or "shorter"; reward/chosen dropping to −3.5
   while reward/rejected fell faster to −5.4 is consistent with either. Only the eval resolves
   it: watch for collapsed CoT/response length and degenerate short refusals post-training.

---

## Where the pilot artifacts live

| Artifact | Location |
|---|---|
| Training console log (full trajectory) | `logs/train.log` (local Mac copy) |
| `eval_metrics.jsonl` | HF `er-abd/harmless-dpo-qwen3-4B-pilot` |
| Best adapter (step 147) | HF `er-abd/harmless-dpo-qwen3-4B-pilot-step-147` |
| All step adapters | HF `er-abd/harmless-dpo-qwen3-4B-pilot-step-{0,29,58,87,116,145,147}` |
| `gpu_memory.json` | Not backed up; value (75.16/85.02 GB) preserved in `logs/train.log` |
| Curation dataset (`outputs/pilot`) | Pod-only — **not backed up** (re-curate or recover if needed) |
