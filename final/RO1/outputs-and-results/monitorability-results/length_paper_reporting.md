# Response length — what to report in the paper text

The figures (`length_by_arm_3models.png`, `length_by_hint_3models.png`) show
the length *levels* (medians). The four results below are NOT in any figure
and should be reported in the text: items 1–2 in Results, items 3–4 as one
Methods sentence each.

| # | What | Where | Source file |
|---|---|---|---|
| 1 | Training effect on length (trained/base GMR) | Results | `length_dataset_level_model_comparison.md` |
| 2 | Hint effect on length (hinted/unhinted GMR) | Results (if discussed) | `length_dataset_hint_level_tables.md` (GMR matrix) |
| 3 | Truncation rates | Methods | step-08 `cot_token_stats.json` truncated flags |
| 4 | Skew (why median, not mean) | Methods | `length_dataset_hint_median_vs_mean.md` |

---

## 1. Training effect on length — paired trained/base GMR

**The headline length result: DPO training makes responses ~1.2–1.4× longer
per question, in both arms, for both training objectives.**

**What the GMR is.** For each question, collapse its runs to a median length
per model; take the ratio trained/base; then the geometric mean of those
per-question ratios. Every question counts equally, and the pairing is exact
because all three models ran the identical instances.

**Why not just divide the medians in the figures.** That would be the
*marginal* contrast (a property of the pooled distribution); the GMR is the
*paired* per-question effect. They differ (e.g. mmlu_pro hinted, harmless:
marginal 1.36× vs GMR 1.29×), so don't expect the figures' bars to reproduce
these numbers by division.

**Numbers** (dataset level, 8 datasets):

| Contrast | Hinted arm | Unhinted arm |
|---|---|---|
| Harmlessness DPO / Base | 1.18–1.32× | 1.16–1.32× |
| Anti-sycophancy DPO / Base | 1.24–1.42× | 1.19–1.41× |

Per-dataset (hinted arm, harmless / anti-syc): mmlu_pro 1.29/1.39 ·
gpqa 1.21/1.29 · lsat_ar 1.18/1.24 · lsat_lr 1.32/1.42 · medxpertqa 1.31/1.42 ·
lexam 1.29/1.40 · codemmlu 1.30/1.39 · wmdp 1.27/1.38.

**The sentence this supports.** The effect is proportional, not additive:
absolute median gaps vary ~7× across datasets (+200 to +1,600 tokens,
tracking each dataset's baseline length), while the ratio stays inside a
narrow 1.2–1.4× band — in both arms, for both objectives, with
anti-sycophancy ≥ harmlessness everywhere.

---

## 2. Hint effect on length — hinted/unhinted GMR (base model)

**Hints lengthen responses modestly — typically ~1.2× per question — with
clear hint-type and dataset structure.**

Same paired-GMR construction, but the contrast is between the two arms of the
base model: per question, median of its 8 hinted runs ÷ median of its 8
unhinted runs, geometric mean across questions. (Pooled cells are the GM of
their constituent dataset×hint cell GMRs, equal weight per cell.)

**Numbers:**

- Grand GM over all 24 dataset×hint cells: **1.18×**.
- By hint type: unethical **1.24×** > grader-hacking **1.19×** > professor
  **1.12×**.
- By dataset: from ~1.0× (medxpertqa 1.01×, wmdp 1.08×) up to ~1.3×
  (mmlu_pro 1.35×, lsat_ar 1.31×, codemmlu 1.31×). Individual cells span
  0.95–1.57×.

**The sentence this supports.** Hint-induced length inflation exists but is
modest and uneven: strongest under the unethical hint and in
mmlu_pro/lsat_ar/codemmlu, essentially absent in medxpertqa.

---

## 3. Truncation (Methods)

**Setup.** Generation is capped at 16,384 tokens. Truncated runs are INCLUDED
in all length statistics at the cap: truncation is caused by length, so
dropping the longest runs would bias length downward. The cap understates the
true tail, so all estimates are conservative.

**Rates** (share of all runs, hinted + unhinted, 51,136 runs per model):

| Model | Overall | wmdp (worst dataset) |
|---|---|---|
| Base | 1.0% (516) | 4.6% |
| Harmlessness DPO | 1.4% (726) | 5.0% |
| Anti-sycophancy DPO | 2.2% (1,113) | 5.9% |

Every dataset other than wmdp is ≤1.0% (base).

**Why this matters for the statistics.** Truncation rises with model
verbosity — the more verbose DPO models truncate more. A MEAN would therefore
be censored by different amounts across exactly the models being compared,
biasing the model contrast itself. The MEDIAN is immune: it is unaffected by
censoring as long as fewer than half the runs truncate, and the worst cell
here is ~6%.

---

## 4. Skew — why median, not mean (Methods)

**Token counts are strongly right-skewed, so the mean describes almost no
actual response.**

Across the base model's 48 condition pools (24 dataset×hint cells × 2 arms):

- Mean ÷ median: **1.30×** typical (median over pools), IQR 1.13–1.81×,
  up to 2.88×.
- SD ≈ mean (median SD/mean = 0.78), so a "mean ± SD" summary would imply
  impossible negative lengths.

Together with the truncation-censoring in (3), this is why all length
reporting is median-first: median [Q1–Q3] in the tables, medians in the
figures, and per-question medians inside the GMR of items 1–2.

---

# Appendix-ready tables

Three tables formatted for direct inclusion in the appendix. A1 is the one to
prioritise — it carries the two things no figure shows (IQRs and the paired
GMRs). A2 backs the hint-effect numbers of item 2; A3 backs the truncation
sentence of item 3.

## Table A1 — Response length by dataset and arm: medians with IQRs, and paired trained/base GMRs

Median [Q1–Q3] completion length in tokens per dataset and arm (hinted pools
the 3 hint types; unhinted is the single hint-independent arm). The last two
columns are the paired per-question GMRs vs base (see item 1; NOT the ratio
of the median columns). Truncated runs included at the 16,384-token cap.

| Dataset | Arm | Base | Harmlessness DPO | Anti-sycophancy DPO | HDPO/Base GMR | ASDPO/Base GMR |
|---|---|---|---|---|---|---|
| mmlu_pro | Hinted | 1563 [817–3510] | 2124 [1102–4538] | 2315 [1222–5000] | 1.29× | 1.39× |
| mmlu_pro | Unhinted | 1051 [684–2183] | 1363 [794–2786] | 1448 [804–2949] | 1.23× | 1.29× |
| gpqa_diamond | Hinted | 4751 [2705–7032] | 5754 [3520–8156] | 6066 [3699–8569] | 1.21× | 1.29× |
| gpqa_diamond | Unhinted | 4372 [2182–6784] | 5291 [2893–7627] | 5608 [3126–8054] | 1.21× | 1.30× |
| lsat_ar | Hinted | 6280 [4088–8932] | 7502 [4996–10399] | 7854 [5296–11008] | 1.18× | 1.24× |
| lsat_ar | Unhinted | 4717 [2838–7493] | 5448 [3415–8551] | 5494 [3406–8641] | 1.16× | 1.19× |
| lsat_lr | Hinted | 1174 [706–2747] | 1759 [855–3599] | 2094 [856–3990] | 1.32× | 1.42× |
| lsat_lr | Unhinted | 959 [647–1800] | 1180 [763–2354] | 1180 [741–2601] | 1.21× | 1.24× |
| medxpertqa | Hinted | 1442 [996–2043] | 1900 [1308–2707] | 2106 [1445–2992] | 1.31× | 1.42× |
| medxpertqa | Unhinted | 1408 [1002–1980] | 1907 [1317–2663] | 2068 [1416–2903] | 1.32× | 1.41× |
| lexam | Hinted | 1940 [1162–2887] | 2532 [1593–3560] | 2726 [1791–3806] | 1.29× | 1.40× |
| lexam | Unhinted | 1676 [876–3050] | 2247 [1151–3702] | 2376 [1222–3939] | 1.24× | 1.30× |
| codemmlu | Hinted | 1744 [703–3900] | 2370 [952–5169] | 2632 [1042–5708] | 1.30× | 1.39× |
| codemmlu | Unhinted | 1164 [541–2780] | 1499 [632–3897] | 1672 [662–4264] | 1.27× | 1.36× |
| wmdp | Hinted | 942 [632–1668] | 1298 [805–2261] | 1456 [855–2522] | 1.27× | 1.38× |
| wmdp | Unhinted | 838 [601–1405] | 1071 [706–1841] | 1154 [722–2134] | 1.23× | 1.30× |

Source: `length_dataset_level_model_comparison.md`.

## Table A2 — Hint effect on length: paired hinted/unhinted GMR, base model, by dataset and hint type

Paired per-question hinted/unhinted GMR for each (dataset, hint type) cell of
the base model. Margins are geometric means (last column: over the 3 hint
types; last row: over the 8 datasets; bottom-right: over all 24 cells).
Per question each cell contrasts the median of its 8 hinted runs (that hint
type) against the median of its 8 unhinted runs; the unhinted arm is
hint-independent and generated once per dataset, so the three cells in a row
share the same denominator — column differences within a row are therefore
pure hint-type effects, and the row margin uses the shared arm once per
question (no triple-counting).

| Dataset | Professor | Grader-hacking | Unethical | GM (dataset) |
|---|---|---|---|---|
| mmlu_pro | 1.21× | 1.32× | 1.53× | 1.35× |
| gpqa_diamond | 1.10× | 1.14× | 1.11× | 1.12× |
| lsat_ar | 1.16× | 1.30× | 1.50× | 1.31× |
| lsat_lr | 1.12× | 1.19× | 1.35× | 1.22× |
| medxpertqa | 1.02× | 1.07× | 0.95× | 1.01× |
| lexam | 1.09× | 1.18× | 1.03× | 1.10× |
| codemmlu | 1.17× | 1.23× | 1.57× | 1.31× |
| wmdp | 1.08× | 1.11× | 1.06× | 1.08× |
| **GM (hint type)** | **1.12×** | **1.19×** | **1.24×** | **1.18×** |

Source: `length_dataset_hint_level_tables.md` (GMR matrix).

## Table A3 — Truncation rate by dataset and model

Share of runs (hinted + unhinted, the unhinted arm counted once) hitting the
16,384-token generation cap. All length statistics include these runs at the
cap.

| Dataset | Base | Harmlessness DPO | Anti-sycophancy DPO |
|---|---|---|---|
| mmlu_pro | 0.3% | 0.5% | 1.1% |
| gpqa_diamond | 1.0% | 1.6% | 2.0% |
| lsat_ar | 1.0% | 2.6% | 3.7% |
| lsat_lr | 0.1% | 0.2% | 0.4% |
| medxpertqa | 0.0% | 0.1% | 0.3% |
| lexam | 0.1% | 0.2% | 1.5% |
| codemmlu | 1.0% | 1.2% | 2.5% |
| wmdp | 4.6% | 5.0% | 5.9% |
| **Overall** | **1.0%** | **1.4%** | **2.2%** |

Source: step-08 `cot_token_stats.json` per-run truncated flags.
