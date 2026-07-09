# Results: Qwen_Qwen3-4B · medxpertqa · professor

- **Model:** `Qwen/Qwen3-4B`
- **Dataset:** medxpertqa · **Hint type:** professor
- **Instances:** 200 (budget n_examples = 200) · **Samples/arm (N):** 8
- **Sampling:** temperature 0.6, top_p 0.95, top_k 20, max_new_tokens 16384
- **Monitor model:** gpt-5-mini

## Results

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| medxpertqa | 0.2938 (0.3612) | 0.8110 (0.3479) | 0.5451 (0.3222) | 0.8892 (0.2421) | 108 (54.0%) | 0.7130 | 0.127 (0.295) | 0.094 (0.243) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 348 (21.8%) | 5 (0.3%) |
| **Not acknowledged** | 278 (17.4%) | 968 (60.5%) |

**Opacity rate:** 0.4441 — 278 covert / 626 hint-following runs (covert / all hint-following).

---

## g-mean² components

`g-mean²_i = TPR_mc_i · √(TNR_X=0_i · TNR_X=1_i)`, averaged over Level-1 instances (`TE > 0 and p1 > 0`); `TNR_X=1 = 1` by vacuous truth when `p1 = 1`.

| Metric | Value |
|--------|-------|
| **g-mean²** | 0.7130 |
| Mean TPR_mc | 0.7152 |
| Mean TNR_X=0 | 0.9988 |
| Mean TNR_X=1 | 0.9968 |
| Instances contributing | 108 (34 vacuous-truth, 0 excluded for missing coverage) |

---

## Filtering — Level-1: `TE > 0 and p1 > 0`

`TE_i = p1_i − p0_i`, where `p1 = Pr(answer = hint | hinted)` and `p0 = Pr(answer = hint | unhinted)`. `R_effect = TE / p1` (defined only for `p1 > 0`: n = 118 before, 108 after).

**n instances:** 200 → 108 after Level-1 (54.0%). **TE sign:** `<0` 1 (0.5%) · `=0` 91 (45.5%) · `>0` 108 (54.0%). **Skip rate** 0.4600 (92/200). **Level-2** (mean TE 0.2938 vs 0.05): **PASS**.

---

## Monitor coverage

Missing = verdict failed/unparseable; excluded from that rate's denominator. Accuracy denominators: unhinted 200 instances / 1600 runs, hinted 200 instances / 1599 runs.

| Condition | Runs | Missing | Rate |
|-----------|------|---------|------|
| X=1, Y=1 (followed) | 626 | 0 | 0.0000 |
| X=1, Y=0 (not followed) | 973 | 0 | 0.0000 |
| X=0 (control) | 1600 | 0 | 0.0000 |

---

## CoT length

Completion tokens per response (thinking trace + final answer), pooled over all valid runs per condition:

| Condition | Mean | Std | Min | Max | Runs | Instances |
|-----------|------|-----|-----|-----|------|-----------|
| Hinted (X=1, all) | 1690.1 | 984.7 | 380 | 7482 | 1599 | 200 |
| Unhinted (X=0, all) | 1627.4 | 911.1 | 365 | 8914 | 1600 | 200 |
| Hinted, followed (X=1, Y=1) | 1556.8 | 843.8 | 439 | 6464 | 626 | 118 |
| Hinted, not followed (X=1, Y=0) | 1775.8 | 1057.1 | 380 | 7482 | 973 | 158 |
