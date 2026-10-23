# Adjusting a group comparison for dataset and hint type (fixed-effects regression)

This note explains what "adjusted for dataset + hint type" means in the consistency-vs-monitorability
analysis, why it is needed, and exactly how it is computed. All real numbers are for base
Qwen/Qwen3-4B, test split, retained (question, hint) rows (TE > 0 and p1 > 0), n = 2,750 rows from
1,383 questions.

---

## 1. The problem adjustment solves

We want to know whether **consistency** (the model gives the same unhinted answer in all valid runs)
goes with **higher monitorability** (per-question g-mean²).

The simplest comparison is the **raw difference**: average g-mean² of consistent rows minus average
g-mean² of inconsistent rows. That comparison is only fair if the two groups are alike in everything
else that affects g-mean². In our data they are not:

| Hint type | Inconsistent (n=1433) | All consistent (n=1317) |
|---|---|---|
| professor | 29.1% | 18.9% |
| grader_hacking | 28.7% | 22.5% |
| unethical | 42.2% | **58.6%** |

| Dataset (selected) | Inconsistent | Consistent but WRONG | Consistent and CORRECT |
|---|---|---|---|
| medxpertqa | 18.3% | **40.0%** | 2.7% |
| mmlu_pro | 10.3% | 7.4% | 17.8% |

And dataset and hint type change g-mean² **on their own** (unethical ≈ 0.89, grader-hacking ≈ 0.52;
lsat_lr far below mmlu_pro).

When both are true — (a) the groups have different mixes of some factor, and (b) that factor affects
the outcome — the factor is a **confounder**. A raw gap could then appear even if consistency had no
link to monitorability at all. Adjustment removes that part of the gap.

---

## 2. The idea: compare like with like

Sort all rows into boxes, one per (dataset, hint type) combination: 8 × 3 = 24 boxes. Then:

- **Raw comparison**: pour all consistent rows into one pile and all inconsistent rows into another,
  compare the averages. The piles contain different mixes of boxes.
- **Adjusted comparison**: compare consistent vs inconsistent **inside each box**, then average those
  within-box gaps. A medxpertqa question is never compared with an mmlu_pro question, and an
  unethical-hint row is never compared with a professor-hint row.

A fixed-effects regression does this in one step.

---

## 3. Worked toy example (one confounder, invented numbers)

| Box | Group | Rows | Avg g-mean² |
|---|---|---|---|
| Hint A (easy to monitor) | Consistent | 80 | 0.92 |
| Hint A | Inconsistent | 20 | 0.88 |
| Hint B (hard to monitor) | Consistent | 20 | 0.60 |
| Hint B | Inconsistent | 80 | 0.50 |

Consistent rows are mostly Hint A; inconsistent rows are mostly Hint B.

**Raw:**
- Consistent = (80 × 0.92 + 20 × 0.60) / 100 = 0.856
- Inconsistent = (20 × 0.88 + 80 × 0.50) / 100 = 0.576
- Raw gap = **+0.280** — mostly because the consistent pile is full of easy Hint-A rows.

**Within each box:**
- Hint A: 0.92 − 0.88 = +0.04
- Hint B: 0.60 − 0.50 = +0.10

**Adjusted** = average of within-box gaps = (0.04 + 0.10) / 2 = **+0.07**
(equal weights here — see §5 for the weighting rule).

**Regression** `g = b0 + b1·consistent + b2·hintB` on these 200 rows gives
b0 = 0.856, **b1 = +0.070**, b2 = −0.350 (verified numerically). b1 is exactly the adjusted gap.

---

## 4. Step by step on the real data

### Step 1 — one row per data point
Each retained (question, hint) pair is one row with: dataset, hint type, group, g-mean².

### Step 2 — turn categories into 0/1 switches (dummy coding)
- Group: one switch `consistent` (1 = consistent, 0 = inconsistent).
- Dataset: 7 switches, one per dataset **except mmlu_pro**.
- Hint type: 2 switches, `grader_hacking` and `unethical`, **no switch for professor**.

One category per factor is left out because it is implied when all its switches are 0; it becomes the
**reference**. Here the reference row is *inconsistent, mmlu_pro, professor*. (Including all
categories plus a baseline would make the equation unsolvable — the "dummy variable trap".)

Example rows of the design matrix:

| Row | base | consistent | gpqa | lsat_ar | lsat_lr | medx | lexam | codemmlu | wmdp | grader_hack | unethical |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mmlu_pro, professor, consistent | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| gpqa, grader-hacking, consistent | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| medx, grader-hacking, inconsistent | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 |
| wmdp, unethical, inconsistent | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 |

### Step 3 — write the equation

```
g-mean² = b0
        + b1·consistent
        + b2·gpqa + b3·lsat_ar + b4·lsat_lr + b5·medx + b6·lexam + b7·codemmlu + b8·wmdp
        + b9·grader_hacking + b10·unethical
        + error
```

11 unknowns. The dataset and hint terms are the **fixed effects**: one intercept shift per category.

### Step 4 — fit by ordinary least squares (OLS)
Choose b0…b10 to minimise the sum of squared differences between predicted and actual g-mean² over all
2,750 rows. Result:

| Coefficient | Value | Meaning |
|---|---|---|
| b0 baseline | 0.787 | predicted g-mean² for inconsistent / mmlu_pro / professor |
| **b1 consistent** | **+0.089** | **consistency gap with dataset and hint held fixed** |
| b2 gpqa_diamond | −0.121 | vs mmlu_pro |
| b3 lsat_ar | −0.017 | |
| b4 lsat_lr | −0.245 | hardest dataset to monitor |
| b5 medxpertqa | −0.074 | |
| b6 lexam | +0.012 | |
| b7 codemmlu | −0.040 | |
| b8 wmdp | −0.056 | |
| b9 grader_hacking | −0.243 | vs professor |
| b10 unethical | +0.121 | vs professor |

### Step 5 — read a prediction
Add the coefficients whose switch is 1:
- gpqa, grader-hacking, consistent: 0.787 + 0.089 − 0.121 − 0.243 = 0.511
- wmdp, unethical, inconsistent: 0.787 − 0.056 + 0.121 = 0.853

Because dataset and hint have their own terms, b1 cannot take credit for a row coming from an easy
dataset or an easy hint.

### Step 6 — confidence interval and p-value (cluster bootstrap)
A question appears up to 3 times (once per hint type), so rows are not independent. Resample
**questions**, not rows:

1. Draw 1,383 questions with replacement; keep all of each question's hint rows together.
2. Refit the regression; record b1.
3. Repeat 10,000 times.
4. 95% CI = 2.5th and 97.5th percentiles of the b1 values → **[+0.061, +0.117]**.
5. Two-sided p = 2 × min(share of b1 ≤ 0, share of b1 ≥ 0) → **< 0.001**.

Treating rows as independent would make the CI too narrow.

---

## 5. How the boxes are weighted

The fitted b1 is a **weighted average of the within-box gaps**. With one fixed effect per box
(`C(d):C(h)`, 24 terms) this is exact; with separate dataset and hint terms, as used here, it holds
approximately. Each box's weight is proportional to

```
weight_box = n_box × share_consistent_box × (1 − share_consistent_box)
```

So a box counts more when it is large **and** evenly split between the two groups. A box that is all
consistent or all inconsistent contains no within-box comparison and gets **zero weight**. (In the toy
example both boxes have n = 100 split 80/20, so equal weights.)

Consequence: if the consistency gap genuinely differs between boxes (it does — e.g. strong for
grader-hacking, weak for professor), b1 is a summary across boxes, not the effect in any one of them.
Report per-hint or per-dataset results alongside it.

---

## 6. Results for the consistency-vs-correctness figure

| Comparison | Raw [95% CI] | p | Adjusted [95% CI] | p (adj.) |
|---|---|---|---|---|
| Consistent but WRONG − Inconsistent (consistency effect) | +0.095 [+0.047, +0.143] | < 0.001 | +0.081 [+0.033, +0.127] | 0.001 |
| Consistent and CORRECT − Consistent but WRONG (correctness effect) | +0.033 [−0.018, +0.083] | 0.201 | +0.019 [−0.037, +0.075] | 0.523 |
| All consistent − Inconsistent | +0.121 [+0.092, +0.150] | < 0.001 | +0.089 [+0.061, +0.117] | < 0.001 |
| Consistent and CORRECT − Inconsistent | +0.128 [+0.097, +0.159] | < 0.001 | +0.090 [+0.060, +0.122] | < 0.001 |

Cluster bootstrap over questions, 10,000 resamples, seed 88. Each adjusted model is fitted on the rows
of the two groups being compared only.

Reading: the consistency relationship shrinks a little after adjustment but stays clearly significant,
so it is not just a dataset/hint mix effect. The correctness effect is not significant either way.

---

## 7. Is adjustment needed?

Yes, whenever the groups being compared differ in their mix of a factor that independently moves the
outcome. Here both conditions hold for dataset and for hint type, for both questions:

- **Consistency vs monitorability**: adjustment answers "is the gap just because consistent rows are
  mostly unethical-hint rows?" → no, +0.089 remains.
- **Correctness vs monitorability**: the two consistent groups come from very different datasets
  (medxpertqa 40% vs 3%), so without adjustment any gap would be uninterpretable.

Report **both** raw and adjusted. The adjusted number is a robustness check, not the "true" effect.

A simpler complement readers grasp instantly: the per-box comparison done directly, e.g. "consistent
> inconsistent in 8/8 datasets".

---

## 8. What adjustment does **not** do

- It only removes differences in the factors you include (dataset, hint type). Other differences
  between groups — p0, question difficulty, response length — remain unless added to the model.
- Adding a variable that lies *on the path* between cause and outcome (a **mediator**) removes real
  effect, not bias. Example: p0 > 0 is part of what being inconsistent means, so adding it shrinks the
  consistency effect to +0.044; that is "the part not explained by p0", not a corrected effect.
- It assumes the effects add up (no interaction). If the consistency gap differs by hint, b1 is a
  weighted average (§5).
- It is still observational: it shows the association is robust to these factors, not that consistency
  causes monitorability.

---

## 9. Minimal code

```python
import numpy as np
from collections import defaultdict

# rows: list of dicts with keys d (dataset), h (hint), q (question id), cons (bool), g (g-mean²)
DATASETS = ["mmlu_pro", "gpqa_diamond", "lsat_ar", "lsat_lr",
            "medxpertqa", "lexam", "codemmlu", "wmdp"]
HINTS = ["professor", "grader_hacking", "unethical"]

def design(rows):
    return np.array([[1.0, float(r["cons"])]
                     + [float(r["d"] == d) for d in DATASETS[1:]]   # mmlu_pro = reference
                     + [float(r["h"] == h) for h in HINTS[1:]]      # professor = reference
                     for r in rows])

X = design(rows)
y = np.array([r["g"] for r in rows])
b1 = np.linalg.lstsq(X, y, rcond=None)[0][1]          # adjusted consistency effect

# cluster bootstrap over questions
by_q = defaultdict(list)
for i, r in enumerate(rows):
    by_q[r["q"]].append(i)
qs = list(by_q)
rng = np.random.default_rng(88)
boot = []
for _ in range(10_000):
    idx = np.concatenate([by_q[qs[j]] for j in rng.integers(0, len(qs), len(qs))])
    boot.append(np.linalg.lstsq(X[idx], y[idx], rcond=None)[0][1])
boot = np.array(boot)
lo, hi = np.percentile(boot, [2.5, 97.5])
p = max(2 * min((boot <= 0).mean(), (boot >= 0).mean()), 1 / len(boot))
print(f"adjusted = {b1:+.3f} [{lo:+.3f}, {hi:+.3f}], p = {p:.4f}")
```

Equivalent in statsmodels: `smf.ols("g ~ cons + C(d) + C(h)", df).fit()` gives the same b1; use the
cluster bootstrap above (or `cov_type="cluster"` with question ids) for the uncertainty.

---

## 10. Suggested wording for a paper

> Group differences are reported raw and adjusted for dataset and hint type. Adjusted estimates are
> the coefficient on the group indicator in an OLS regression of per-question g-mean² on the group
> indicator plus dataset and hint-type fixed effects, i.e. a comparison within the same dataset and
> hint type. 95% CIs and two-sided p-values come from a cluster bootstrap over questions (10,000
> resamples), since each question contributes up to three rows (one per hint type).
