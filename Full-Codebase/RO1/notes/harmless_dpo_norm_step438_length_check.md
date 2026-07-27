# Length-normalized DPO (SimPO) checkpoint: first length check (step-438)

Date: 2026-07-12. Follow-up to [`harmless_dpo_response_length_inflation.md`](harmless_dpo_response_length_inflation.md)
(base vs. `step-735` standard DPO, full analysis + mechanism) and
[`harmless_dpo_length_inflation_step147_check.md`](harmless_dpo_length_inflation_step147_check.md)
(3-checkpoint monotonic length-growth check on standard DPO). This note covers
the first RO1 monitorability length check on the **length-normalized**
harmlessness-DPO variant trained in response to those findings.

## Problem / hypothesis

The existing notes established that standard harmlessness-DPO training
(`er-abd/harmless-dpo-qwen3-4B`, `loss_type="sigmoid"`, unnormalized
sum-of-logprobs, `beta=0.1`) inflates response length progressively with
training step (base < step-147 < step-735 on `gpqa_diamond`/`professor`,
hinted arm), and traced the mechanism to the DPO loss's unnormalized
per-token-sum objective rather than the training data's length asymmetry.
Per [[project_ro2_simpo_length_fix]], RO2 built `train_simpo.py`
(`METHOD=dpo|simpo` in `train.sh`, length-normalized loss, `beta=5`) as the
fix, and trained `er-abd/harmless-dpo-norm-qwen3-4B` from it — checkpoints
pushed every ~20% interval, `harmless_dpo_norm` locally
(`RO2/checkpoints/qwen3-4B/harmless_dpo_norm/`, 283M, committed to git
alongside the small metadata files per the existing checkpoints/ tracking
convention).

**Hypothesis under test:** length-normalized training should reduce or
eliminate the length inflation seen in the standard-DPO checkpoints, since it
removes the mechanical bias (reward = sum of per-token logprobs, which favors
longer generations independent of quality) identified as the driver in the
existing mechanism analysis.

`step-438` was the checkpoint on hand from the just-completed `harmless_dpo_norm`
training run (RO2 training happened earlier in this session; not chosen for
any particular property like the accuracy-plateau checkpoint used in the
step-147 note).

## Method — and a scope caveat

Run: `monitorability-eval` inference, `gpqa_diamond`, `professor` hint, N=8,
on `er-abd/harmless-dpo-norm-qwen3-4B-step-438` (merged adapter). This run
was **unhinted-arm-only** (a temp script reimplementing the unhinted-only
generation loop, since `run_inference_qwen()` has no built-in switch to skip
the hinted arm — see the step-147 note for the mirror-image hinted-only case).
Consequences:

- No `hinted.jsonl` exists for this run, so hinted-arm length, the
  followed/not-followed split, and the hinted/unhinted-gap-widening check
  (tracked in the original note) **cannot** be computed for `step-438` yet.
- `08_cot_token_analysis.py` requires both arms and cannot run on this output
  (same reason as the step-147 case: it feeds `compute_gmean2()`, which
  intersects hinted/unhinted instance IDs). A one-off unhinted-only variant
  was used instead (`/tmp/unhinted_only_cot_stats.py` on the GPU box, not
  part of the repo), reusing `src/cot_tokens.py`'s bucket/truncation/raw-stats
  helpers plus a single overall valid-run condition stat (unhinted has no
  followed/not-followed split in `compute_gmean2()`'s own methodology — that
  split only applies to the hinted arm).
- The monitor phase, TE/g-mean^2 metrics, and the hinted arm were not run for
  this checkpoint/dataset — only the raw unhinted CoT token-length stats
  below exist so far.

## Results

### Raw output — `step-438` (length-normalized), `gpqa_diamond`, `professor`, unhinted arm only

```json
{
  "model_name": "er-abd_harmless-dpo-norm-qwen3-4B-step-438",
  "hf_id": "er-abd/harmless-dpo-norm-qwen3-4B-step-438",
  "dataset": "gpqa_diamond",
  "hint_type": "professor",
  "note": "UNHINTED ARM ONLY - no hinted.jsonl for this run.",
  "condition_stats_unhinted_only": {
    "unhinted_all": {
      "mean": 6705.230286113049,
      "std": 3289.4141348517605,
      "min": 727,
      "max": 16208,
      "n": 1433
    }
  },
  "n_unhinted_valid": 1433,
  "buckets": {
    "0-512": 0,
    "512-1024": 9,
    "1024-2048": 85,
    "2048-4096": 236,
    "4096-8192": 689,
    "8192-16384": 414,
    "16384-32768": 0,
    "32768+": 151
  },
  "truncation": {
    "total": 1584,
    "truncated": 151,
    "rate": 0.09532828282828283
  },
  "raw_stats": {
    "mean": 7627.890782828283,
    "std": 4227.545147427133,
    "min": 727,
    "max": 16384,
    "n": 1584
  }
}
```

### Comparison across checkpoints — `gpqa_diamond` / `professor`, unhinted arm

Base and `step-735` figures pulled from the existing two-arm
`cot_token_stats.json` artifacts (standard DPO run); `step-438` from the
unhinted-only run above (length-normalized/SimPO-style run).

| | base `Qwen3-4B` | step-735 (standard DPO) | **step-438 (length-normalized)** |
|---|---:|---:|---:|
| unhinted mean, valid runs (tokens) | 4,806 | 5,518 | **6,705** |
| unhinted mean, all responses (raw, incl. truncated) | ~5,044\* | ~5,874\* | **7,628** |
| truncation rate | 0.98%\* | 1.64%\* | **9.53%** |
| n (valid / total) | 1,566 / 3,168\* | 1,553 / 3,168\* | 1,433 / 1,584 |

\* base/step-735 `raw_stats`/truncation figures are computed over **both
arms combined** (n=3,168, from the existing two-arm artifacts); step-438's
are **unhinted-only** (n=1,584). Not perfectly apples-to-apples on an
absolute-count basis, but the gap between step-438 and step-735 (9.53% vs.
1.64%) is far too large to be explained by that difference alone.

## Interpretation

**This result runs counter to the hypothesis.** The length-normalized
checkpoint is longer, not shorter, than the standard-DPO checkpoint it was
meant to improve on — both in mean token length (6,705 vs. 5,518 valid-run
mean, +21.5%) and dramatically in truncation rate (9.53% vs. 1.64%, roughly
6x higher). A ~10% truncation rate is also a standalone data-quality concern
independent of the length-inflation question: those responses hit the
16,384-token cap mid-generation and are typically unparseable (no answer
extracted), which will bias any accuracy/TE computation on this checkpoint
until addressed.

Before treating this as a real negative result for length-normalized
training, three confounds need ruling out:

1. **Hyperparameter mismatch, not just loss normalization.** The
   length-normalized run used `beta=5` vs. the standard run's `beta=0.1` —
   a 50x difference in the DPO temperature parameter. `beta` alone
   substantially changes training dynamics (how sharply the policy is pushed
   away from the reference model); this could dominate the outcome
   independent of whether the loss is length-normalized.
2. **Step-count non-comparability.** `step-438` (norm) vs. `step-735`
   (standard) are not necessarily equivalent amounts of training if batch
   size, steps-per-epoch, or total dataset size differ between the two runs.
   Comparing by step number assumes they're on the same scale, which hasn't
   been verified.
3. **Possible training-side issue.** Worth checking
   `RO2/checkpoints/qwen3-4B/harmless_dpo_norm/eval_metrics.jsonl`
   (accuracy/margin/loss trajectory) the same way the original note used
   that file to catch the standard run's accuracy-plateau-past-step-441
   issue — before concluding anything about length behavior, confirm
   `step-438` is even a sensible/healthy checkpoint to be evaluating.

## Decision / next steps

Not yet decided — this is a single data point (one checkpoint, one dataset,
one arm) and the result is surprising enough to warrant verification before
acting on it either way:

- Check `harmless_dpo_norm/eval_metrics.jsonl` for accuracy/margin/loss
  trajectory, same as done for the standard run.
- Backfill the hinted arm for `step-438`/`gpqa_diamond`/`professor` so the
  standard `08_cot_token_analysis.py` pipeline can run normally and produce
  the full hinted/unhinted comparison (including the followed-hint split and
  the hinted/unhinted gap tracked in the original note).
- If the truncation rate holds up under the full two-arm run, investigate
  the truncated transcripts directly — is the model looping / failing to
  terminate, or just genuinely generating longer reasoning that happens to
  exceed the cap more often at this `beta`?
- Once confirmed real (not a beta or step-count artifact), decide whether
  `beta=5` needs retuning down, independent of keeping the length-normalized
  loss itself.
