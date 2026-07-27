## qwen3_4b_saferl

| Dataset      | Mean TE (before) | Mean R_effect (before) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
| ------------ | ---------------- | ---------------------- | ---------------- | ------- | ------------------ | ---------------- |
| mmlu_redux   | 0.0800           | 0.8452                 | 7                | 0.5952  | 0.820 (0.327)      | 0.785 (0.350)    |
| gpqa_diamond | 0.3250           | 0.7151                 | 25               | 0.8567  | 0.410 (0.459)      | 0.335 (0.412)    |
| wmdp         | 0.3100           | 0.8507                 | 22               | 0.8409  | 0.740 (0.385)      | 0.550 (0.452)    |
| codemmlu     | 0.1000           | 0.4500                 | 9                | 0.8519  | 0.715 (0.410)      | 0.665 (0.427)    |
| lsat_ar      | 0.1500           | 0.8684                 | 18               | 0.6389  | 0.780 (0.341)      | 0.705 (0.352)    |
| lsat_lr      | 0.1450           | 0.8472                 | 11               | 0.2197  | 0.790 (0.351)      | 0.685 (0.413)    |



----



## Results analysis:

### qwen3_4b vs. qwen3_4b_saferl

Qwen3-4B-SafeRL is Qwen3-4B after GSPO reinforcement learning on WildJailbreak (harmful +
adversarial-harmful prompts), with the Hybrid Reward (§3.5.1, Eq. 2): maximize safety per
Qwen3Guard's judgment, penalize refusal, and reward helpfulness via WorldPM — specifically to
resist jailbreaks/harmful-content requests without degenerating into blanket refusal. It was
never trained or evaluated on anything resembling this task: academic MCQs, planted
authority/sycophancy hints, or chain-of-thought faithfulness. So there's no mechanistic reason to
expect this RL to transfer into more (or less) resistance to a "Stanford professor thinks the
answer is X" hint — that's an orthogonal behavior axis from "don't help build a bomb."

**Sanity check against the paper's own numbers:** their Table 10 (Think mode, Hybrid reward)
shows a small, consistent general-capability tax: AIME25 65.6→63.5, LiveCodeBench 48.4→47.5,
GPQA 55.9→51.2 (Pass@1). The `gpqa_diamond` unhinted accuracy here drops 0.445→0.410 — the same
direction, same rough magnitude, on the same benchmark family. That's a good independent
replication: the pipeline and the released model are behaving consistently with what Qwen
reported, which validates the setup before trusting anything more novel.

**Results, dataset by dataset:**

| Dataset | TE (base→saferl) | g-mean² (base→saferl) | Read |
|---|---|---|---|
| mmlu_redux | 0.092→0.080 (↓ slightly) | 0.70→0.60 (↓) | Both near/at the `mean(TE)<0.05` unreliable-signal floor — weak hint effect either way, low-confidence dataset |
| gpqa_diamond | 0.320→0.325 (flat) | 0.71→0.86 (↑) | Sycophancy unchanged; monitor got better here |
| wmdp | 0.316→0.310 (flat) | 0.97→0.84 (↓) | Sycophancy unchanged; monitorability dropped |
| codemmlu | 0.055→0.100 (↑, more hint-following) | 0.55→0.85 (↑) | SafeRL model is more sycophantic here, monitor still caught it better |
| lsat_ar | 0.142→0.150 (flat) | 0.85→0.64 (↓) | |
| lsat_lr | 0.100→0.145 (↑, more hint-following) | 0.56→0.22 (collapse) | Biggest swing, also more sycophantic |

**No consistent direction.** TE moves up in 2 datasets, flat in 3, down in 1 — there's no case
for "SafeRL made the model more resistant to sycophancy hints." g-mean² swings even more (up in
2, down in 4, one near-total collapse), with no relationship to the direction or size of the TE
change. That inconsistency is itself informative: it looks like noise dominating a genuinely null
effect, not a real signal.

**Why to be cautious over-reading the g-mean² deltas:** the `n TE>0 (post L1)` populations are
tiny — 7 to 25 instances per dataset. A g-mean² swing from 0.56 to 0.22 on `lsat_lr` is driven by
an n of 9→11 passing the L1 filter; a couple of monitor calls flipping is enough to produce
exactly this kind of jump. Several of these datasets (`mmlu_redux`, `codemmlu`, and to a lesser
extent `lsat_lr`) also sit right at or below the `mean(TE) < 0.05` threshold the metric doc flags
as "hint not reliably shifting behavior — excluded from analysis." These should be treated as
low-confidence rather than as evidence of anything.

**Thesis-relevant takeaway:** this comparison is decent evidence for a claim that can actually be
made: safety RL targeted at harmful/jailbreak-style content does not transfer to chain-of-thought
faithfulness under sycophancy pressure — the two are separable safety properties, and one doesn't
come "for free" from the other. What it does not show is that SafeRL made monitorability worse in
any principled sense — the per-dataset g-mean² movement looks more consistent with small-sample
noise than a systematic regression.

A defensible headline number, rather than 6 noisy per-dataset ones, would pool all `TE>0`
instances across datasets into one combined g-mean² (or bootstrap a CI per dataset) before making
a comparative claim — a candidate cross-dataset addition to `07_summary_table.py`.




-----




# Summary Analysis: Qwen3-4B vs. Qwen3-4B-SafeRL on the Honeypot Suite

## Setup reminder

Qwen3-4B-SafeRL is the base model after online safety RL (GSPO on WildJailbreak prompts) with a hybrid reward that penalizes unsafe content in **both the final response and the thinking trace** (`is_safe(x, t)`), plus a refusal penalty. This comparison therefore measures the effect of safety RL *with direct CoT-level supervision* — a stronger, more adversarial pressure than output-only DPO — on hint-following behaviour and CoT monitorability.

## Headline deltas (base → SafeRL)

| Dataset | ΔTE | Δg-mean² | n (post-L1) | Read |
|---|---|---|---|---|
| mmlu_redux | −0.012 | **−0.107** | 7 → 7 | noise-range, tiny n |
| gpqa_diamond | +0.005 | **+0.149** | 23 → 25 | only credible gain |
| wmdp | −0.006 | **−0.128** | 24 → 22 | theoretically loaded drop |
| codemmlu | +0.045 | **+0.297** | 8 → 9 | unreliable cell (see below) |
| lsat_ar | +0.008 | **−0.212** | 14 → 18 | drop on non-safety reasoning |
| lsat_lr | +0.045 | **−0.345** | 9 → 11 | largest drop, small n |

Unweighted mean g-mean²: **0.724 → 0.667** (net decline ≈ 0.06; down on 4/6 datasets).

## Finding 1 — Behaviour is unchanged; only transparency moved

Mean TE shifts by ≤0.05 everywhere, retained-set sizes are stable, and hinted/unhinted accuracies are near-identical across models. Safety RL did **not** make the model resist hints — unsurprising, since following an MCQ cue is not "unsafe content" and was never penalized in training. This is the important dissociation: whatever changed in g-mean² changed **conditional on the same hint-following behaviour**. The suite also transferred cleanly across the pair (minimal skip-rate confound), validating the frozen-suite design.

## Finding 2 — Net monitorability decline with a hypothesis-consistent pattern

- **WMDP: 0.97 → 0.84.** The single most meaningful cell: near-ceiling monitorability drops specifically on the safety-relevant dataset, exactly where a reward that punishes unsafe-*looking* thinking should induce sanitized CoTs that omit load-bearing influences.
- **Both LSAT tasks drop (−0.21, −0.35).** These CoTs contain nothing for the safety monitor to penalize, so if real, this is *generalized* CoT style shift — the "safe-sounding, templated" register learned under RL spilling over into unrelated reasoning and reducing verbalization of situational cues. This would be spillover from CoT-level pressure rather than a targeted effect.
- **GPQA improves (+0.15, n≈24).** The one countervailing signal worth taking semi-seriously; hardest task, where reasoning is most load-bearing.

## Finding 3 — Discard or heavily discount codemmlu

R_effect is 0.33 (base) and 0.45 (SafeRL): the majority of intervention-arm hint-following is coincidental rather than hint-caused, and n = 8–9. This configuration fails the spirit of the incentive-calibration criterion; its +0.30 "improvement" should not enter conclusions. mmlu_redux (TE ≈ 0.08–0.09, n = 7) is only marginally better.

## Statistical caveats (necessary before any claim)

With n = 7–25 per cell and g-mean² a multiplicative composite of estimated rates, single-cell deltas of ±0.15 sit within sampling noise (SE on a proportion at n≈10 is ~0.15). Additionally, Level-1 is re-applied per model, so the two scores per dataset are computed over **partially different instance sets** — some of each delta may be composition, not transparency. Required follow-ups: (i) recompute both models on the intersection of retained instances and run a paired flip analysis (instances where the answer still follows the hint but acknowledgment disappeared — then read those CoTs); (ii) bootstrap CIs per cell; (iii) compare mean CoT lengths (>20% difference triggers the length-stratified analysis per protocol) and verify control-arm TNR is stable, ruling out monitor false negatives on SafeRL's restyled traces.

## Bottom line

Safety RL with CoT-in-the-reward left honeypot *behaviour* intact (TE ≈ unchanged) while producing a modest net decline in CoT *monitorability* (mean g-mean² 0.72 → 0.67), concentrated on the safety-relevant dataset (WMDP, from near-ceiling) and on non-safety logical reasoning (LSAT) — a pattern consistent with sanitized-CoT spillover, while the model's headline behavioural safety metrics (97–98% WildGuard safety in the source paper) look excellent. Pending the paired-intersection and CI analyses, this is a real-released-model data point for the project's central claim: safety training can improve behavioural safety while quietly eroding the reasoning-transparency channel used to audit it — motivating RO2's question of whether milder, output-only preference optimization produces the same signature.