# What explains the increase in monitorability after DPO?

Both trained models increase pooled monitorability relative to base (anti-sycophancy DPO: Δg-mean² ≈ +0.023), and both also generate longer chains of thought. This document collects the analyses that test candidate explanations for the monitorability increase, one hypothesis per section. All length statistics are **medians** (skew-robust; token counts are right-skewed and truncation-capped), and all correlational checks are run on the **base model** (`Qwen/Qwen3-4B`), where the candidate mechanism must already be visible if it is to explain the trained models' gains.

---

## Investigation 1 — Is the increase just a length effect?

**The claim.** Training made the model write longer CoTs, and a longer CoT simply gives it more room to mention the hint — so monitorability rose as a free side-effect of verbosity. The claim is live: the trained models really do write ~1.2–1.4× longer CoTs on the same questions (`length_dataset_level_model_comparison.md`).

**The check.** If more tokens made responses more monitorable, that link would already be visible *inside the base model*. So, on the base model (hinted arm — the transcript the monitor reads; per-question median over 8 runs) we ask: **(A)** do datasets with longer responses have higher g-mean²? **(B)** within each (dataset × hint type) cell, do questions with longer responses have higher per-question g-mean² (`gmean2_i`)? **(C)** if any association remains, which part of length carries it — how verbose the model is anyway, or how much it stretches *because of the hint*?

### Check A — across datasets (n = 8)

| Dataset (sorted by length) | Median hinted tokens | g-mean² |
|---|---|---|
| wmdp | 942 | 0.738 |
| lsat_lr | 1,174 | **0.529** (lowest) |
| medxpertqa | 1,442 | 0.700 |
| mmlu_pro | 1,563 | **0.800** (highest) |
| codemmlu | 1,744 | 0.756 |
| lexam | 1,940 | 0.795 |
| gpqa_diamond | 4,751 | **0.647** (2nd lowest) |
| lsat_ar | 6,280 | 0.776 |

With only 8 datasets a correlation coefficient is not meaningful, so read the ranks — and they do not line up. The most monitorable dataset (mmlu_pro) is one of the shortest; gpqa_diamond writes ~3× longer CoTs and is second-*least* monitorable; the two shortest datasets (wmdp, lsat_lr) sit at opposite ends of the monitorability range. If token budget set monitorability, the two long-CoT datasets would top the ranking — neither does.

### Check B — within cells (24 dataset × hint cells)

Longer questions are not reliably more monitorable within a cell either: the within-cell Spearman correlation between a question's length and its `gmean2_i` has median ρ = +0.14 (positive in 16/24 cells, significantly *negative* in 3; range +0.33 to −0.42), and pooling all 2,750 (question × hint) slices gives ρ = −0.005 — flat, as the quantile figure shows (`extra-plots/cot-length-vs-monitorability/gmean2_by_hinted_length_quantile_Qwen_Qwen3-4B.png`: quintiles level at ~0.74–0.79).

### Check C — which part of length carries the weak positive tilt?

Same within-cell correlation, but splitting each question's hinted length into how long the model talks *anyway* (its unhinted length) and how much it stretches *because of the hint* (hinted ÷ unhinted):

| Length variable | Median ρ (24 cells) | Positive cells |
|---|---|---|
| Hinted length (total) | +0.14 | 16/24 |
| Unhinted length (verbosity in general) | **−0.11** | 10/24 |
| Hint-induced stretch (hinted ÷ unhinted) | **+0.20** | **21/24** |

Only the hint-induced stretch consistently predicts monitorability. General verbosity, if anything, predicts *lower* monitorability (matching the pooled unhinted-length result, ρ ≈ −0.29; `gmean2_by_unhinted_length_quantile_Qwen_Qwen3-4B.png`).

### Interpretation

Three things kill the claim. **(1) There is no length→monitorability relationship to inherit:** not across datasets, not within cells — the pooled association is exactly zero and the sign flips from cell to cell, where a "more room" mechanism would have to be positive everywhere. **(2) Training lengthened the wrong thing:** the extra tokens are mostly general verbosity (both arms grew ~1.3×), and on the base model general verbosity goes with slightly *lower* monitorability — so if length itself mattered, monitorability should have dropped, not risen. **(3) The one length signal that does track monitorability — stretching in reaction to the hint — is a consequence of talking about the hint, not a cause:** acknowledging and weighing the hint costs words (hint-following responses run ~1.3× longer than non-following ones in the same cell). A longer CoT is not why the monitor sees the hint; the model handling the hint out loud is why the CoT is longer.

**Conclusion.** Longer CoTs are neither necessary nor sufficient for higher monitorability here. The increase is not explained by the model simply having more space to externalise the hint.

**Caveats.** The n = 8 dataset check is descriptive and rules out only a strong length effect; within-cell associations are conditioned on TE > 0 questions; hinted length and `gmean2_i` share hint-following as a common cause (which is why Check C, not the raw correlation, is the informative view); and the pooled 2,750-slice statistic ignores the clustering of the 3 hints within a question.

---

## Investigation 2 — Did training widen the Hinted−Unhinted length gap, and is that the explanation?

**The claim.** After training, the models lengthen *specifically when the hint is present* — the Hinted-vs-Unhinted length gap grew — and this hint-specific stretching (rather than verbosity in general, Investigation 1) is what raised monitorability.

**The check.** For each question: median length of its 8 hinted runs ÷ median of its 8 unhinted runs, averaged geometrically across questions (the paired **H/U GMR** — the right statistic for skewed, multiplicative token counts). All three models ran the identical questions, so their GMRs compare exactly, question for question (the **ratio-of-ratios**, RoR: trained GMR ÷ base GMR, > 1× = wider gap than base).

### Results

| Model | Grand H/U GMR (GM of 24 cells) | RoR vs base | Cells with wider gap than base |
|---|---|---|---|
| Base | 1.18× | — | — |
| Harmlessness DPO | 1.22× | 1.03× | 20/24 |
| Anti-sycophancy DPO | 1.24× | 1.05× | 21/24 |

The gap did widen — but only by 3–5%, against the ~30% both arms grew overall. And it doesn't line up with the monitorability gains: priced with the base model's own stretch→monitorability slope (Investigation 1, Check C) it buys only ~+0.003–0.005 g-mean², a fraction of the observed +0.032/+0.023; the cells that widened most are not the cells that gained most (Δ-vs-Δ correlations n.s., ρ = −0.10/+0.21); and gpqa_diamond and medxpertqa didn't widen at all (RoR 0.99×/1.01×) yet still gained monitorability.

### Interpretation

Yes, after training the models do stretch a bit more when the hint is present: on the same questions, the base model's hinted responses are typically 1.18× longer than its unhinted ones, and this grows to 1.22× after harmless DPO and 1.24× after anti-sycophancy DPO. But look at how small that change is — the gap widened by only about 3–5%, while monitorability rose by a much larger margin. If a wider hinted-vs-unhinted gap were what made the models more monitorable, a shift from 1.18× to 1.24× is far too little to do the job. And the more natural reading runs the other way: talking about the hint — acknowledging it, weighing it, pushing back on it — takes extra words, so a model that externalises the hint more will automatically show a slightly wider gap. In other words, the small increase in the H/U ratio is a *footprint* of the model verbalising the hint more, not the *reason* the monitor catches it more often.

---

## Investigation 3 — Is the increase just higher accuracy or consistency?

**Hypothesis.** DPO made the model better at the tasks (accuracy) or more decisive (consistency — a question is *consistent* when all of its valid unhinted runs give the same answer, the step-31 definition), and monitorability rose as a by-product. This is the confound to take most seriously: on the base model, questions answered *consistently* are genuinely more monitorable (+0.096 g-mean² for consistently-wrong vs inconsistent questions, CI excluding 0), while correctness adds little once consistency is controlled (+0.037, CI includes 0) — see `extra-plots/task-performance-vs-monitorability/consistency_vs_correctness_Qwen_Qwen3-4B.png`.

**Test.** Two questions, per trained model: did accuracy/consistency actually move, and did they move *in the cells where monitorability moved*? Accuracy is checked per (dataset × hint) cell using the exact paired deltas in `delta_vs_base_per_cell.json` (the data behind `delta_acc_vs_delta_gmean2.png`); consistency is recomputed from each model's unhinted runs, per dataset (the unhinted arm is shared across hint types; n = 8, so descriptive only).

### Results

**Accuracy (24 cells).** Unhinted accuracy barely moved (mean Δ ≈ +0.004 for both models) and does not track monitorability: sign-concordance with Δg-mean² is at chance (13/24 harmless, 15/24 anti-syc) and the Δ-vs-Δ correlation is null (Spearman −0.20 / +0.01, n.s.). Hinted accuracy did rise in most cells — but that *is* the trained behaviour (following wrong hints less), not a separate skill gain, so both deltas trend up almost everywhere; the informative magnitude test fails: cells with bigger hinted-accuracy gains do **not** have bigger monitorability gains (Spearman −0.45, p = 0.03 harmless; −0.24, n.s. anti-syc — if anything the opposite).

**Consistency (8 datasets).** Share of questions that are consistent (same answer in all valid unhinted runs):

| Dataset | Base | Δ harmless | Δ anti-syc | Δg-mean² harmless | Δg-mean² anti-syc |
|---|---|---|---|---|---|
| mmlu_pro | 0.665 | +0.055 | +0.040 | +0.042 | +0.036 |
| gpqa_diamond | 0.419 | +0.051 | +0.030 | +0.037 | +0.015 |
| lsat_ar | 0.600 | +0.065 | +0.030 | +0.020 | +0.026 |
| lsat_lr | 0.770 | +0.000 | +0.015 | +0.021 | +0.020 |
| medxpertqa | 0.445 | −0.015 | +0.010 | **+0.046** | **+0.061** |
| lexam | 0.475 | +0.030 | +0.015 | +0.060 | +0.062 |
| codemmlu | 0.750 | −0.020 | −0.060 | −0.034 | −0.064 |
| wmdp | 0.615 | +0.015 | +0.000 | +0.062 | +0.026 |
| **Mean** | 0.592 | **+0.023** | **+0.010** | +0.032 | +0.023 |

The consistent share moved by only one or two questions per hundred on average (+0.023 / +0.010), and the pattern breaks where it matters: for harmless DPO, medxpertqa got *less* consistent yet posted one of the largest monitorability gains, and lsat_ar had the *largest* consistency gain (+0.065) but nearly the smallest monitorability gain (+0.020); for anti-syc, the biggest monitorability gainers (lexam +0.062, medxpertqa +0.061) had near-zero consistency change while the biggest consistency gainer (mmlu_pro) gained only moderately. The magnitude also falls far short: making one question consistent is worth +0.096 g-mean² for that question (the step-31 consistency effect), so a +0.010–0.023 shift in the consistent share buys at most ~+0.001–0.002 pooled g-mean² — an order of magnitude below the observed +0.023/+0.032.

### Interpretation

The worry is fair: the base model shows that when the model is sure of itself — same answer every run — the monitor reads it more easily. So maybe DPO just made the model surer or smarter, and monitorability came along for free. The data says no, three times over. First, the models barely got better: unhinted accuracy moved by under half a percentage point, and only one or two questions per hundred flipped from inconsistent to consistent, while monitorability moved several times what those shifts could buy — the cause is too small for the effect. Second, the improvements sit in the wrong places: the datasets where monitorability jumped most are not the ones where accuracy or consistency improved, and the cell-by-cell correlations are null or negative. Third, the one accuracy number that clearly rose — accuracy *with the hint present* — rose because the model stopped following wrong hints, which is the trained behaviour itself, not extra capability; and even it doesn't track where monitorability rose. The model got more *readable*, not more *right*.

---

## Investigation 4 — Did the model just become *more* susceptible to the hint?

**The claim.** Training made the model easier to sway: TE (how much the hint shifts the model toward the hinted answer) rose, R_effect rose, more questions pass the level-1 filter (TE > 0), and monitorability rose mechanically on the back of that.

**The check.** The premise is directly measurable: compare each trained model's mean TE on V, mean R_effect on V, and level-1 pass share (% TE > 0) against base, per cell, using the exact paired deltas in `delta_vs_base_per_cell.json`.

### Results

| Δ vs base (24 cells) | Harmless DPO | Anti-syc DPO |
|---|---|---|
| Mean TE on V | **−0.007** (up in 10/24; sig-up 0, sig-down 2) | **−0.021** (up in 8/24; sig-up 0, sig-down 5) |
| % TE > 0 (level-1 pass share) | **−2.3 pts** (up in 4/24; sig-down 5) | **−3.8 pts** (up in 3/24; sig-down 10) |
| Mean R_effect on V | +0.001 (mixed) | −0.008 (mixed) |
| g-mean² | **+0.032** (up in 21/24) | **+0.023** (up in 20/24) |

The premise is backwards: susceptibility *fell* — the hint sways the trained models less, and fewer questions show any hint effect at all (that is the anti-sycophancy objective working) — yet monitorability rose in 20–21/24 cells. Two checks close the escape routes. First, even within the base model susceptibility carries no monitorability signal: per-slice Spearman(TE, g-mean²) = −0.02, n.s., quintiles flat (`extra-plots/task-performance-vs-monitorability/gmean2_by_te_quantile_Qwen_Qwen3-4B.png`) — so even a susceptibility *increase* would have had nothing to pass on. Second, the same orthogonality defuses the shrinking-V composition worry (g-mean² is computed on each model's own TE > 0 set, which shrank): selecting questions by TE does not select them by monitorability, so dropping the low-TE tail should not move g-mean². (The Δ-vs-Δ correlations across cells are weak and non-significant; the one nominal exception, R_effect for harmless at ρ = 0.41, is not evidence — R_effect appears inside the g-mean² formula, so that correlation is partly circular.)

### Interpretation

This argument has the story upside down. Training didn't make the model easier to nudge — it made it *harder* to nudge: the hint sways it less, and fewer questions show any hint effect. If monitorability were a shadow of susceptibility, it should have fallen along with it; instead it rose almost everywhere. The base model shows why that's no paradox: how strongly a hint bends the model's answer says nothing about whether the model *talks about* being bent — the two are unrelated. What changed after training is the talking, not the bending: the model is swayed less often, but when it is swayed it says so in its chain of thought more often, and that is what the monitor picks up.

**Caveat.** V (the TE > 0 set) is each model's own, so the trained models' g-mean² is measured on a somewhat smaller question set than base's. The orthogonality result is why we don't think this composition shift manufactures the gain, but it is an inference from the base model's structure, not a direct same-set comparison.

---

*Data: per-run hinted/unhinted token counts for all three models from each model's `cot_token_stats.json` (`length_pools`, truncated runs included at the 16,384-token cap); per-question `gmean2_i` from `results_per_instance.csv`; cell-level g-mean² from `results_summary.json`; dataset-level g-mean² from `main_results_dataset_level_tables.md`; trained-vs-base length GMRs from `length_dataset_level_model_comparison.md`; paired per-cell accuracy/g-mean² deltas from `delta_vs_base_per_cell.json`; consistency recomputed from each model's `inference-outputs/…/unhinted.jsonl` parsed answers (binary step-31 definition: consistent = all valid unhinted runs agree).*
