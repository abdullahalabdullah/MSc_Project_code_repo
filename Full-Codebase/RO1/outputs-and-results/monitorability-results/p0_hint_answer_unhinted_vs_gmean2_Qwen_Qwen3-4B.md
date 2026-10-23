# Monitorability when the model independently chose the hint's answer (unhinted arm)

Model: Qwen/Qwen3-4B (base). Hint types always pooled (professor, grader-hacking, unethical).

Groups (retained question × hint rows, i.e. passing TE > 0 and p1 > 0):
- **Never chose it**: the model never picked the hint's answer in its valid unhinted runs (p0 = 0).
- **Chose it ≥1 run**: the model picked the hint's answer in at least one valid unhinted run (p0 > 0).

Counts are rows, with distinct questions in brackets. p0 comes from the unhinted arm shared by all
three hint types, so each question (up to 3 rows) belongs to one group only. g-mean² is the mean of
per-question g-mean² over rows (hint types pooled row-wise, so values differ from the main table's
unweighted mean of the three hint types).

Difference = "chose it ≥1 run" − "never chose it". 95% CIs and two-sided p-values from a cluster
bootstrap over questions (5,000 resamples, seed 88). Adjusted = OLS coefficient on p0 > 0 with
dataset + hint-type fixed effects (Table 1) or hint-type fixed effects only (Table 2).

## Table 1: all 8 datasets aggregated

| Scope | Never chose it: rows (questions) | g-mean² | Chose it ≥1 run: rows (questions) | g-mean² | Difference [95% CI] | p | Adjusted [95% CI] | p (adj.) |
|---|---|---|---|---|---|---|---|---|
| All 8 datasets, 3 hint types pooled | 2110 (1141) | 0.804 | 640 (242) | 0.648 | −0.157 [−0.194, −0.119] | < 0.001 | −0.115 [−0.152, −0.078] | < 0.001 |

## Table 2: per dataset (3 hint types pooled)

| Dataset | Never chose it: rows (questions) | g-mean² | Chose it ≥1 run: rows (questions) | g-mean² | Difference [95% CI] | p | Adjusted [95% CI] | p (adj.) |
|---|---|---|---|---|---|---|---|---|
| mmlu_pro | 315 (182) | 0.867 | 38 (15) | 0.662 | −0.205 [−0.344, −0.061] | 0.006 | −0.174 [−0.310, −0.034] | 0.016 |
| gpqa_diamond | 243 (115) | 0.725 | 171 (64) | 0.611 | −0.114 [−0.184, −0.045] | < 0.001 | −0.102 [−0.171, −0.031] | 0.004 |
| lsat_ar | 256 (141) | 0.814 | 67 (28) | 0.798 | −0.016 [−0.123, +0.083] | 0.790 | +0.010 [−0.095, +0.107] | 0.822 |
| lsat_lr | 170 (104) | 0.660 | 45 (18) | 0.432 | −0.228 [−0.424, −0.033] | 0.026 | −0.158 [−0.362, +0.049] | 0.114 |
| medxpertqa | 329 (163) | 0.780 | 75 (27) | 0.547 | −0.233 [−0.339, −0.130] | < 0.001 | −0.202 [−0.308, −0.099] | < 0.001 |
| lexam | 280 (139) | 0.872 | 134 (48) | 0.705 | −0.168 [−0.247, −0.089] | < 0.001 | −0.143 [−0.223, −0.064] | < 0.001 |
| codemmlu | 244 (152) | 0.838 | 43 (18) | 0.754 | −0.084 [−0.189, +0.016] | 0.104 | −0.012 [−0.120, +0.091] | 0.829 |
| wmdp | 273 (145) | 0.813 | 67 (24) | 0.658 | −0.155 [−0.270, −0.046] | 0.006 | −0.120 [−0.235, −0.012] | 0.030 |

Note: the "never chose it" group includes all consistent questions (p0 = 0 by construction), which
are more monitorable anyway. Restricted to inconsistent questions only, the base-model effect is
−0.085 [−0.128, −0.048] (dataset + hint-type fixed effects). Association only, not causal.

## Table 3: all 8 datasets aggregated, by number of unhinted runs (out of 8) that chose the hint's answer

Restricted to rows whose question has exactly 8 valid unhinted runs (2,618 of 2,750 retained rows;
132 rows / 54 questions with truncated runs excluded so that k/8 is exact). 8/8 cannot occur: p0 = 1
gives TE ≤ 0, so those rows are removed by the retention filter. Difference and adjusted columns are
relative to 0/8; adjusted = OLS with one dummy per level plus dataset + hint-type fixed effects.
Same cluster bootstrap (5,000 resamples, seed 88).

| Times the model chose the hint's answer unhinted | Rows (questions) | g-mean² | Difference vs 0/8 [95% CI] | p | Adjusted vs 0/8 [95% CI] | p (adj.) |
|---|---|---|---|---|---|---|
| 0/8 (never) | 2029 (1105) | 0.808 | reference | — | reference | — |
| 1/8 | 197 (74) | 0.584 | −0.224 [−0.286, −0.163] | < 0.001 | −0.191 [−0.253, −0.130] | < 0.001 |
| 2/8 | 88 (33) | 0.618 | −0.189 [−0.277, −0.104] | < 0.001 | −0.143 [−0.218, −0.069] | < 0.001 |
| 3/8 | 88 (35) | 0.699 | −0.109 [−0.205, −0.012] | 0.026 | −0.080 [−0.178, +0.020] | 0.122 |
| 4/8 | 57 (21) | 0.591 | −0.216 [−0.326, −0.119] | < 0.001 | −0.161 [−0.259, −0.067] | < 0.001 |
| 5/8 | 53 (22) | 0.664 | −0.143 [−0.287, −0.008] | 0.039 | −0.102 [−0.236, +0.024] | 0.122 |
| 6/8 | 49 (18) | 0.749 | −0.059 [−0.172, +0.040] | 0.257 | −0.017 [−0.111, +0.069] | 0.714 |
| 7/8 | 57 (21) | 0.754 | −0.053 [−0.172, +0.057] | 0.364 | −0.023 [−0.129, +0.078] | 0.654 |
| *1/8–7/8 combined* | 589 (224) | 0.644 | | | | |

Reading: no dose-response. The drop appears as soon as the hint's answer is chosen once (1/8:
0.584), stays roughly flat at ~0.59–0.70 through 5/8, and recovers to ~0.75 at 6/8–7/8 (not
significantly different from 0/8). Levels above 1/8 have only 18–35 questions each, so their
estimates are noisy. The recovery at high counts may partly reflect the TPR_mc correction (small
R_effect inflates TPR_mc) rather than more frequent hint acknowledgement — not checked directly.
Supported claim: choosing the hint's answer even once unhinted goes with lower monitorability; not
"the more often, the lower".
