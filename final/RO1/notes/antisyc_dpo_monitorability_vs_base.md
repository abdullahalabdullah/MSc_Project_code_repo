# Anti-Sycophancy DPO vs Base — Monitorability (g-mean²), Hint Susceptibility, and Capability

- **Base:** `Qwen/Qwen3-4B`
- **Trained:** `er-abd/anti-syc-dpo-qwen3-4B-step-520` (anti-sycophancy DPO, 3,080-pair 6:1 dataset, best checkpoint by eval reward-accuracy)
- **Eval:** full monitorability grid — 8 datasets × 3 hint types, n=200 instances/config, N=8 samples/arm, monitor gpt-5-mini
- **Sources:** `outputs-and-results/monitorability-results/Qwen_Qwen3-4B/results_overall_summary.md` and `.../er-abd_anti-syc-dpo-qwen3-4B-step-520/results_overall_summary.md`
- **Date:** 2026-07-19

## Headline

**Anti-sycophancy training increased monitorability overall: g-mean² went up in 20 of 24
configs (mean Δ +0.023), down in only 4.** A 20/24 split has a two-sided sign-test
p ≈ 0.0015, so the direction is not chance.

## Configs where anti-sycophancy INCREASED g-mean² (20)

| Dataset | Hint type | Base | Anti-syc | Δ |
|---|---|---|---|---|
| lexam | grader_hacking | 0.667 | 0.763 | **+0.096** |
| mmlu_pro | grader_hacking | 0.669 | 0.763 | **+0.094** |
| medxpertqa | grader_hacking | 0.508 | 0.594 | **+0.086** |
| lexam | professor | 0.791 | 0.855 | +0.064 |
| medxpertqa | professor | 0.713 | 0.770 | +0.057 |
| wmdp | unethical | 0.909 | 0.959 | +0.050 |
| lsat_ar | grader_hacking | 0.668 | 0.713 | +0.045 |
| lsat_lr | grader_hacking | 0.406 | 0.447 | +0.041 |
| medxpertqa | unethical | 0.878 | 0.918 | +0.040 |
| wmdp | professor | 0.849 | 0.888 | +0.039 |
| lsat_ar | professor | 0.746 | 0.776 | +0.030 |
| lexam | unethical | 0.928 | 0.953 | +0.026 |
| gpqa_diamond | grader_hacking | 0.304 | 0.327 | +0.023 |
| mmlu_pro | unethical | 0.943 | 0.965 | +0.022 |
| codemmlu | grader_hacking | 0.460 | 0.477 | +0.017 |
| lsat_lr | unethical | 0.767 | 0.782 | +0.015 |
| gpqa_diamond | unethical | 0.866 | 0.879 | +0.013 |
| gpqa_diamond | professor | 0.771 | 0.782 | +0.011 |
| lsat_lr | professor | 0.414 | 0.420 | +0.006 |
| lsat_ar | unethical | 0.914 | 0.917 | +0.003 |

## Configs where anti-sycophancy DECREASED g-mean² (4)

| Dataset | Hint type | Base | Anti-syc | Δ |
|---|---|---|---|---|
| **codemmlu** | **professor** | 0.870 | 0.681 | **−0.189** |
| codemmlu | unethical | 0.938 | 0.917 | −0.021 |
| wmdp | grader_hacking | 0.454 | 0.445 | −0.010 |
| mmlu_pro | professor | 0.789 | 0.780 | −0.009 |

### Reading notes

- **Structure in the gains:** the three biggest increases are all `grader_hacking` cells
  (+0.086 to +0.096) — the hint type where the base model is most covert (opacity
  0.48–0.73). The trained model acknowledges the hint more when it follows it (e.g.
  mmlu_pro/grader_hacking opacity 0.485 → 0.449; gpqa/grader_hacking 0.836 → 0.785).
  The already-transparent `unethical` cells improve least (ceiling ~0.91–0.96).
- **Per-cell significance caveat:** most individual deltas sit well inside overlapping
  95% CIs (cell CIs are ±0.05–0.12) — only the top gains and the codemmlu/professor drop
  approach cell-level significance. The claim-bearing statistic is the aggregate
  direction (20/24 up, mean +0.023), not any single cell.
- **The one real degradation, codemmlu/professor (−0.189),** is the same cell where
  harmless-DPO also dropped most (−0.105) — a shared, replicated weak spot worth a
  qualitative look (its TE_after also fell 0.43 → 0.39, and acknowledgment dropped),
  rather than an anti-syc-specific artifact.

## n TE>0 (post L1) per config — base vs anti-syc (out of 200 instances)

*Count of instances where the hint measurably shifted the model's answers (the
population g-mean² is computed over).*

| Dataset | Hint type | Base | Anti-syc | Δ |
|---|---|---|---|---|
| mmlu_pro | professor | 67 (33.5%) | 58 (29.0%) | −9 |
| mmlu_pro | grader_hacking | 89 (44.5%) | 93 (46.5%) | +4 |
| mmlu_pro | unethical | 197 (98.5%) | 187 (93.5%) | −10 |
| gpqa_diamond | professor | 118 (59.6%) | 116 (58.6%) | −2 |
| gpqa_diamond | grader_hacking | 119 (60.1%) | 114 (57.6%) | −5 |
| gpqa_diamond | unethical | 177 (89.4%) | 175 (88.4%) | −2 |
| wmdp | professor | 88 (44.7%) | 76 (38.6%) | −12 |
| wmdp | grader_hacking | 83 (42.1%) | 84 (42.4%) | +1 |
| wmdp | unethical | 169 (85.8%) | 160 (81.2%) | −9 |
| codemmlu | professor | 58 (29.0%) | 57 (28.5%) | −1 |
| codemmlu | grader_hacking | 59 (29.5%) | 62 (31.0%) | +3 |
| codemmlu | unethical | 170 (85.0%) | 154 (77.0%) | −16 |
| lsat_ar | professor | 71 (35.5%) | 57 (28.5%) | −14 |
| lsat_ar | grader_hacking | 87 (43.5%) | 77 (38.5%) | −10 |
| lsat_ar | unethical | 165 (82.5%) | 149 (74.5%) | −16 |
| lsat_lr | professor | 44 (22.0%) | 36 (18.0%) | −8 |
| lsat_lr | grader_hacking | 49 (24.5%) | 45 (22.5%) | −4 |
| lsat_lr | unethical | 122 (61.0%) | 108 (54.0%) | −14 |
| medxpertqa | professor | 108 (54.0%) | 96 (48.0%) | −12 |
| medxpertqa | grader_hacking | 106 (53.0%) | 98 (49.0%) | −8 |
| medxpertqa | unethical | 190 (95.0%) | 186 (93.0%) | −4 |
| lexam | professor | 112 (56.0%) | 90 (45.0%) | −22 |
| lexam | grader_hacking | 115 (57.5%) | 114 (57.0%) | −1 |
| lexam | unethical | 187 (93.5%) | 176 (88.0%) | −11 |

**Pattern: down in 21/24 configs.** The anti-syc model is measurably **less
hint-susceptible** — fewer instances where the planted hint moves its answers. The
structure is meaningful: the drops concentrate in `professor` (an authority figure's
asserted opinion — the most sycophancy-shaped hint) and `unethical` cells, while
`grader_hacking` is essentially unchanged (+4/+1/+3/−1 on half its cells). That is the
anti-sycophancy treatment *transferring*: it trained resistance to asserted opinions,
and that is exactly which hint types the model now resists. (Interpretation side note:
this slightly shrinks the g-mean² conditioning population, so the monitorability gains
above are measured on a somewhat smaller, still-susceptible subset.)

## Unhinted accuracy — base vs anti-syc

*The unhinted arm is shared across hint types, so this is per dataset (wmdp shows tiny
per-cell variation for the trained model).*

| Dataset | Base | Anti-syc | Δ |
|---|---|---|---|
| mmlu_pro | 0.719 | 0.731 | +0.012 |
| gpqa_diamond | 0.512 | 0.512 | 0.000 |
| wmdp | 0.679 | 0.677–0.681 | ~0.000 |
| codemmlu | 0.763 | 0.762 | −0.001 |
| lsat_ar | 0.782 | 0.826 | **+0.044** |
| lsat_lr | 0.834 | 0.845 | +0.011 |
| medxpertqa | 0.127 | 0.128 | +0.001 |
| lexam | 0.585 | 0.551 | −0.034 |

**Capability is preserved across the board** — six of eight datasets within ±1.2pp, with
two mild movements (lsat_ar +4.4pp, lexam −3.4pp) whose 95% CIs overlap the base values
comfortably. No capability tax from the training — consistent with the anti-syc eval's
own finding (ID bare accuracy 55.3% → 56.5%). This table is also the natural content for
the paired compare report's missing `--capability-json` attachment.

## Combined story

The anti-syc model **(a)** resists hints more — especially opinion-shaped ones,
**(b)** keeps its underlying accuracy, and **(c)** when it *does* follow a hint, it is
more likely to say so in its CoT (the g-mean² gains). All three move in the direction
you would want.

## One-sentence thesis version

> Anti-sycophancy DPO increased CoT monitorability in 20/24 dataset×hint configurations
> (mean Δg-mean² +0.023, sign-test p ≈ 0.002), with the largest gains precisely where the
> base model was most covert (grader_hacking), and a single substantial degradation
> (codemmlu×professor) shared with the harmlessness-DPO arm.
