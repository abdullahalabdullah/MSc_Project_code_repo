# Results overall summary — per (dataset × hint type)

- **Model:** `Qwen/Qwen3-4B` (`Qwen_Qwen3-4B`)
- **Datasets:** mmlu_pro, medxpertqa, lexam
- **Hint types:** professor

## mmlu_pro × professor

| Dataset  | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
| -------- | -------------------- | -------------------------- | ------------------- | ------------------------- | ---------------- | ------- | ------------------ | ---------------- |
| mmlu_pro | 0.1800 (0.3236)      | 0.9688 (0.1250)            | 0.5625 (0.3354)     | 0.9688 (0.1250)           | 16 (32.0%)       | 0.6823  | 0.785 (0.378)      | 0.665 (0.424)    |

|                      | Followed   | Did not follow |
| -------------------- | ---------- | -------------- |
| **Acknowledged**     | 23 (11.6%) | 1 (0.5%)       |
| **Not acknowledged** | 12 (6.1%)  | 162 (81.8%)    |

**Opacity rate:** 0.3429 — 12 covert / 35 hint-following runs (covert / all hint-following).

## medxpertqa × professor

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| medxpertqa | 0.3950 (0.4043) | 0.9028 (0.2348) | 0.6583 (0.3113) | 0.9028 (0.2348) | 30 (60.0%) | 0.6639 | 0.125 (0.259) | 0.035 (0.124) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 50 (25.0%) | 1 (0.5%) |
| **Not acknowledged** | 39 (19.5%) | 110 (55.0%) |

**Opacity rate:** 0.4382 — 39 covert / 89 hint-following runs (covert / all hint-following).

## lexam × professor

| Dataset | Mean TE before (Std) | Mean R_effect before (Std) | Mean TE after (Std) | Mean R_effect after (Std) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
|---|---|---|---|---|---|---|---|---|
| lexam | 0.1900 (0.3095) | 0.6087 (0.4185) | 0.5278 (0.2957) | 0.7778 (0.2957) | 18 (36.0%) | 0.5972 | 0.655 (0.434) | 0.520 (0.454) |

| | Followed | Did not follow |
|---|---|---|
| **Acknowledged** | 33 (16.5%) | 5 (2.5%) |
| **Not acknowledged** | 39 (19.5%) | 123 (61.5%) |

**Opacity rate:** 0.5417 — 39 covert / 72 hint-following runs (covert / all hint-following).
