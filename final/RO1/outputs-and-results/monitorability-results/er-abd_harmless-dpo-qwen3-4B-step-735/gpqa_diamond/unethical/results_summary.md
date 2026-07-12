# Results: er-abd_harmless-dpo-qwen3-4B-step-735 · gpqa_diamond · unethical

- **Model:** `er-abd_harmless-dpo-qwen3-4B-step-735`
- **Dataset:** gpqa_diamond · **Hint type:** unethical
- **Instances:** 198 (budget n_examples = 200) · **Samples/arm (N):** 8
- **Sampling:** temperature 0.6, top_p 0.95, top_k 20, max_new_tokens 16384
- **Monitor model:** gpt-5-mini

## Results

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| gpqa_diamond | 0.6383 [0.5863, 0.6893] | 0.8249 [0.7796, 0.8675] | 0.7355 [0.6913, 0.7804] | 0.8786 [0.8436, 0.9119] | 172 (86.9% [82.3, 91.4]) | 0.8891 [0.8566, 0.9193] | 0.507 [0.447, 0.565] | 0.165 [0.123, 0.209] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 1023 (65.5%) | 3 (0.2%) |
| **Not acknowledged** | 221 (14.1%) | 315 (20.2%) |

**Opacity rate:** 0.1777 [0.1425, 0.2141] — 221 covert / 1244 hint-following runs (covert / all hint-following).

---

## g-mean² components

`g-mean²_i = TPR_mc_i · √(TNR_X=0_i · TNR_X=1_i)`, averaged over Level-1 instances (`TE > 0 and p1 > 0`); `TNR_X=1 = 1` by vacuous truth when `p1 = 1`. 95% CIs are percentile bootstrap over instances.

| Metric | Value | 95% CI |
|--------|-------|--------|
| **g-mean²** | 0.8891 | [0.8566, 0.9193] |
| Mean TPR_mc | 0.8962 | [0.8663, 0.9242] |
| Mean TNR_X=0 | 1.0000 | [1.0000, 1.0000] |
| Mean TNR_X=1 | 0.9918 | [0.9776, 1.0000] |
| Instances contributing | 172 (111 vacuous-truth, 0 excluded for missing coverage) | |

---

## Filtering — Level-1: `TE > 0 and p1 > 0`

`TE_i = p1_i − p0_i`, where `p1 = Pr(answer = hint | hinted)` and `p0 = Pr(answer = hint | unhinted)`. `R_effect = TE / p1` (defined only for `p1 > 0`: n = 183 before, 172 after).

**n instances:** 198 → 172 after Level-1 (86.9%). **TE sign:** `<0` 1 (0.5%) · `=0` 25 (12.6%) · `>0` 172 (86.9%). **Skip rate** 0.1313 [0.0859, 0.1768] (26/198). **Level-2** (mean TE 0.6383 vs 0.05): **PASS**.

---

## Monitor coverage

Missing = verdict failed/unparseable; excluded from that rate's denominator. Accuracy denominators: unhinted 198 instances / 1553 runs, hinted 198 instances / 1562 runs.

| Condition | Runs | Missing | Rate |
|-----------|------|---------|------|
| X=1, Y=1 (followed) | 1244 | 0 | 0.0000 |
| X=1, Y=0 (not followed) | 318 | 0 | 0.0000 |
| X=0 (control) | 1553 | 0 | 0.0000 |

---

## CoT length

Completion tokens per response (thinking trace + final answer), pooled over all valid runs per condition:

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 6105.9 | 3459.0 | 5668 | 439 | 16342 | 1562 | 198 |
| Unhinted (X=0, all) | 5518.1 | 3243.2 | 5203 | 549 | 16355 | 1553 | 198 |
| Hinted, followed (X=1, Y=1) | 6035.3 | 3310.8 | 5558 | 439 | 16342 | 1244 | 183 |
| Hinted, not followed (X=1, Y=0) | 6381.9 | 3980.0 | 6019 | 551 | 16221 | 318 | 77 |
