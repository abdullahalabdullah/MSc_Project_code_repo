# Harmlessness-DPO length inflation: checking an earlier checkpoint (step-147)

Date: 2026-07-12. Follow-up to [`harmless_dpo_response_length_inflation.md`](harmless_dpo_response_length_inflation.md)
(base vs. `step-735`, full hinted+unhinted, all 8 monitorability datasets). This
note adds a third data point — `step-147`, an earlier checkpoint in the same
training run — to check whether the length inflation documented there grows
monotonically with training step, or was specific to the final checkpoint.

## Problem / hypothesis

The existing note established, from `step-735` vs. base `Qwen3-4B`, that
harmlessness-DPO training inflates response length: every one of 8 eval
datasets showed the trained model producing longer CoT than baseline, on both
hinted and unhinted prompts, with the hinted/unhinted gap also widening in
22/23 dataset x hint-type rows. The mechanism section there traces this to a
structural property of the DPO loss as configured (`loss_type="sigmoid"`,
unnormalized sum-of-logprobs, no `ld_alpha`) — not the length asymmetry in the
training pairs, which if anything should have pushed the model toward
*brevity*. That note's own "next step to confirm causally" flagged
`step-441` as worth checking, since accuracy plateaued at step-441 while
training continued to step-735 with only margin growth (no more pairs correct)
— exactly the kind of extra training the length-inflation mechanism predicts
would keep inflating length for no accuracy benefit.

This check used the checkpoint on hand, `step-147` (RO1's monitorability-eval
sweep was already extending checkpoint coverage for `er-abd/harmless-dpo-qwen3-4B`
beyond `step-735`), rather than `step-441` specifically. It still serves the
same purpose: a checkpoint from earlier in training, to see whether length
grows with step count or jumps straight to its `step-735` level.

**Hypothesis under test:** if the loss-structure mechanism in the existing note
is right, mean response length should increase monotonically with training
step (base < step-147 < step-735), independent of dataset-specific content.

## Method — and a scope caveat

Run: `monitorability-eval` inference, `gpqa_diamond`, `professor` hint,
N=8, on `er-abd/harmless-dpo-qwen3-4B-step-147` (merged adapter). Due to a
GPU-box disk-space constraint at the time, this was run as a **hinted-arm-only**
one-off (a temp script monkeypatched `unhinted_arm_complete()` to skip
generating the unhinted arm) rather than through the standard
`run_inference.sh` pipeline. Consequences:

- No `unhinted.jsonl` exists for this run, so the hinted/unhinted-gap-widening
  check from the existing note (the "reactive to hints" finding) **cannot** be
  computed for `step-147` — only the hinted-arm mean length.
- The standard `08_cot_token_analysis.py` requires both arms (it feeds
  `compute_gmean2()`, which intersects hinted/unhinted instance IDs) and
  cannot run on this output. A one-off hinted-only variant was written instead
  (`/tmp/hinted_only_cot_stats.py` on the GPU box, not part of the repo) reusing
  the same bucket/truncation/raw-stats helpers from `src/cot_tokens.py`, plus a
  hand-rolled valid-run condition split (not truncated + parsed answer present,
  by followed/not-followed hint) mirroring `compute_gmean2()`'s methodology.
- The monitor phase (step 3) and TE/g-mean^2 metrics were not run for this
  checkpoint/dataset — only the raw CoT token-length stats below exist.

## Results

### Raw output — `step-147`, `gpqa_diamond`, `professor`, hinted arm only

```json
{
  "model_name": "er-abd_harmless-dpo-qwen3-4B-step-147",
  "hf_id": "er-abd/harmless-dpo-qwen3-4B-step-147",
  "dataset": "gpqa_diamond",
  "hint_type": "professor",
  "note": "HINTED ARM ONLY - no unhinted.jsonl for this run.",
  "condition_stats_hinted_only": {
    "hinted_all": {
      "mean": 5328.255400254129,
      "std": 3138.9142831364484,
      "min": 487,
      "max": 16357,
      "n": 1574
    },
    "hint_followed": {
      "mean": 5746.229765013055,
      "std": 3125.681228035881,
      "min": 487,
      "max": 16357,
      "n": 766
    },
    "hint_not_followed": {
      "mean": 4932.007425742574,
      "std": 3101.7001034180116,
      "min": 530,
      "max": 16000,
      "n": 808
    }
  },
  "n_hinted_valid": 1574,
  "n_followed_hint": 766,
  "n_not_followed_hint": 808,
  "buckets": {
    "0-512": 1,
    "512-1024": 63,
    "1024-2048": 172,
    "2048-4096": 388,
    "4096-8192": 689,
    "8192-16384": 261,
    "16384-32768": 0,
    "32768+": 10
  },
  "truncation": {
    "total": 1584,
    "truncated": 10,
    "rate": 0.006313131313131313
  },
  "raw_stats": {
    "mean": 5398.051767676768,
    "std": 3249.2776634231923,
    "min": 487,
    "max": 16384,
    "n": 1584
  }
}
```

### Comparison across checkpoints — `gpqa_diamond` / `professor`, hinted arm

Base and `step-735` figures pulled from the existing two-arm
`cot_token_stats.json` artifacts (`outputs-and-results/monitorability-results/{model}/gpqa_diamond/professor/cot_token_stats.json`);
`step-147` from the hinted-only run above.

| | base `Qwen3-4B` | `step-147` | `step-735` |
|---|---:|---:|---:|
| hinted mean (tokens) | 5,058 | 5,328 (+5.3%) | 5,878 (+16.2%) |
| hint-followed mean | 5,569 | 5,746 | 6,307 |
| hint-not-followed mean | 4,577 | 4,932 | 5,508 |
| followed − not-followed gap | 992 (19.6% of mean) | 814 (15.3%) | 799 (13.6%) |
| truncation rate (hinted-only pool) | 0.98%* | 0.63% | 1.64%* |

\* base and step-735 truncation rates are computed over both arms combined
(n=3168); step-147's is hinted-arm only (n=1584) — not perfectly apples-to-apples,
included for reference only, not part of the length-inflation conclusion below.

## Interpretation

1. **Monotonic length growth with training step, consistent with the existing
   note's mechanism finding.** Hinted-arm mean length rises step-wise: 5,058 to
   5,328 to 5,878. With n in the 1,500-1,600 range per condition, SEM is
   roughly std/sqrt(n) ~ 75-80 tokens — an 820-token gap (base to step-735) is
   on the order of 10 SEM, not plausibly noise. This is a second, independent
   dataset point (on top of the 8-dataset base-vs-735 comparison already
   documented) supporting the same conclusion: the DPO loss's unnormalized
   sum-of-logprobs objective is inflating length progressively over training,
   not just as some final-checkpoint artifact.

2. **The followed/not-followed gap shrinks as a fraction of mean length with
   more training** (19.6% to 15.3% to 13.6%), even though the absolute gap
   stays roughly flat (~800-990 tokens). This is a different axis from the
   hinted/unhinted gap tracked in the existing note (which widens with
   training) — this one is about hint-followed vs. hint-not-followed *within*
   the hinted arm. Not deeply investigated here; noted for completeness, not
   used in the decision below.

3. **What this does NOT establish**: whether the extra length at step-147
   and step-735 corresponds to any accuracy or safety benefit over base, or
   whether step-147 specifically is already past the point of
   diminishing accuracy returns (the existing note's `eval_metrics.jsonl`
   table shows `step-147` already at 0.8368 accuracy vs. 0.9126 by
   `step-441`/`step-735` — so step-147 is mid-climb on accuracy, not yet at
   the plateau where the existing note's "margin-only training" concern
   applies most cleanly). A real test of "is the extra length buying
   anything" needs gpqa_diamond accuracy and/or harmlessness-eval safety rate
   at matching checkpoints, which have not been pulled alongside these length
   numbers yet.

## Decision / next steps

Not yet decided. Leaning toward proceeding with length-normalized (SimPO)
training for the next RO2 run (`train_simpo.py`, `METHOD=simpo` per
[[project_ro2_simpo_length_fix]]) given two independent pieces of evidence
now point the same direction (base-vs-735 8-dataset sweep, and this
3-checkpoint monotonic trend). Before committing:

- Pull gpqa_diamond accuracy and harmlessness-eval (WildGuard) safety rate for
  `step-147` and `step-735` alongside these length numbers, to rule out that
  the extra length is buying genuine accuracy/safety improvement rather than
  pure reward-hacking via the loss's length bias.
- If time allows, the existing note's original suggestion — running the same
  length measurement on `step-441` (already pushed to the Hub) — would give a
  cleaner test of the "post-accuracy-plateau training only inflates length"
  hypothesis than `step-147`, which is still pre-plateau on accuracy.
- If confirmed, rerun RO2 harmlessness-DPO training with `train_simpo.py`
  (length-normalized loss) and compare length inflation + accuracy/safety
  against this DPO run at matched checkpoints.
