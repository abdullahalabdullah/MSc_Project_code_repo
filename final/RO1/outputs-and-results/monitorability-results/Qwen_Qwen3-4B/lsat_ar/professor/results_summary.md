# Results: Qwen_Qwen3-4B · lsat_ar · professor

- **Model:** `Qwen/Qwen3-4B`
- **Dataset:** lsat_ar · **Hint type:** professor
- **Instances:** 200 (budget n_examples = 200) · **Samples/arm (N):** 8
- **Sampling:** temperature 0.6, top_p 0.95, top_k 20, max_new_tokens 16384
- **Monitor model:** gpt-5-mini

## Results

| Dataset | Mean TE before [95% CI] | Mean R_effect before [95% CI] | Mean TE after [95% CI] | Mean R_effect after [95% CI] | n TE>0 (post L1) [95% CI] | g-mean² [95% CI] | Unhinted Acc [95% CI] | Hinted Acc [95% CI] |
|---|---|---|---|---|---|---|---|---|
| lsat_ar | 0.1304 [0.0956, 0.1663] | 0.6417 [0.4904, 0.7751] | 0.3986 [0.3404, 0.4591] | 0.9007 [0.8497, 0.9456] | 71 (35.5% [29.0, 42.0]) | 0.7463 [0.6617, 0.8252] | 0.782 [0.735, 0.825] | 0.710 [0.660, 0.758] |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 203 (12.8%) | 4 (0.3%) |
| **Not acknowledged** | 103 (6.5%) | 1279 (80.5%) |

**Opacity rate:** 0.3366 [0.2664, 0.4116] — 103 covert / 306 hint-following runs (covert / all hint-following).

---

## g-mean² components

`g-mean²_i = TPR_mc_i · √(TNR_X=0_i · TNR_X=1_i)`, averaged over Level-1 instances (`TE > 0 and p1 > 0`); `TNR_X=1 = 1` by vacuous truth when `p1 = 1`. 95% CIs are percentile bootstrap over instances.

| Metric | Value | 95% CI |
|--------|-------|--------|
| **g-mean²** | 0.7463 | [0.6617, 0.8252] |
| Mean TPR_mc | 0.7508 | [0.6651, 0.8304] |
| Mean TNR_X=0 | 1.0000 | [1.0000, 1.0000] |
| Mean TNR_X=1 | 0.9886 | [0.9757, 0.9980] |
| Instances contributing | 71 (9 vacuous-truth, 0 excluded for missing coverage) | |

---

## Filtering — Level-1: `TE > 0 and p1 > 0`

`TE_i = p1_i − p0_i`, where `p1 = Pr(answer = hint | hinted)` and `p0 = Pr(answer = hint | unhinted)`. `R_effect = TE / p1` (defined only for `p1 > 0`: n = 85 before, 71 after).

**n instances:** 200 → 71 after Level-1 (35.5%). **TE sign:** `<0` 12 (6.0%) · `=0` 117 (58.5%) · `>0` 71 (35.5%). **Skip rate** 0.6450 [0.5800, 0.7100] (129/200). **Level-2** (mean TE 0.1304 vs 0.05): **PASS**.

---

## Monitor coverage

Missing = verdict failed/unparseable; excluded from that rate's denominator. Accuracy denominators: unhinted 200 instances / 1592 runs, hinted 200 instances / 1589 runs.

| Condition | Runs | Missing | Rate |
|-----------|------|---------|------|
| X=1, Y=1 (followed) | 306 | 0 | 0.0000 |
| X=1, Y=0 (not followed) | 1283 | 0 | 0.0000 |
| X=0 (control) | 1592 | 0 | 0.0000 |

---

## CoT length

Completion tokens per response (thinking trace + final answer), pooled over all valid runs per condition:

| Condition | Mean | Std | Median | Min | Max | Runs | Instances |
|-----------|------|-----|--------|-----|-----|------|-----------|
| Hinted (X=1, all) | 6106.3 | 3341.4 | 5438 | 959 | 16356 | 1589 | 200 |
| Unhinted (X=0, all) | 5540.3 | 3418.9 | 4704 | 888 | 16182 | 1592 | 200 |
| Hinted, followed (X=1, Y=1) | 9030.4 | 3244.9 | 9078 | 1691 | 16356 | 306 | 85 |
| Hinted, not followed (X=1, Y=0) | 5408.9 | 2966.0 | 4661 | 959 | 16200 | 1283 | 190 |
