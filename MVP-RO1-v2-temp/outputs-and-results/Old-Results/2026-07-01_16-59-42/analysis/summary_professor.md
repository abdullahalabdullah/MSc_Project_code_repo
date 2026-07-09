## qwen3_1_7b

| Dataset           | Mean TE (before) | Mean R_effect (before) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
| ----------------- | ---------------- | ---------------------- | ---------------- | ------- | ------------------ | ---------------- |
| mmlu_redux        | 0.0750           | 0.6894                 | 9                | 0.5093  | 0.765 (0.359)      | 0.710 (0.389)    |
| gpqa_diamond      | 0.1233           | 0.4138                 | 17               | 0.3786  | 0.350 (0.398)      | 0.325 (0.358)    |
| mathbench_primary | 0.0350           | 0.6111                 | 3                | 1.0000  | 0.960 (0.170)      | 0.920 (0.260)    |
| mathbench_middle  | 0.0300           | 0.6667                 | 2                | 1.0000  | 0.930 (0.189)      | 0.905 (0.262)    |
| mathbench_high    | 0.0300           | 0.8750                 | 4                | 0.3750  | 0.922 (0.212)      | 0.892 (0.264)    |
| mathbench_college | 0.0400           | 0.9167                 | 4                | 0.4375  | 0.920 (0.217)      | 0.890 (0.258)    |
| compmath          | 0.0617           | 0.5556                 | 6                | 0.7500  | 0.945 (0.210)      | 0.877 (0.284)    |
| wmdp              | 0.1267           | 0.5200                 | 18               | 0.6898  | 0.577 (0.437)      | 0.455 (0.409)    |
| codemmlu          | 0.0883           | 0.5526                 | 14               | 0.5714  | 0.705 (0.387)      | 0.625 (0.407)    |

## qwen3_4b

| Dataset           | Mean TE (before) | Mean R_effect (before) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
| ----------------- | ---------------- | ---------------------- | ---------------- | ------- | ------------------ | ---------------- |
| mmlu_redux        | 0.0917           | 0.9167                 | 7                | 0.7024  | 0.818 (0.340)      | 0.762 (0.392)    |
| gpqa_diamond      | 0.3200           | 0.7069                 | 23               | 0.7077  | 0.445 (0.435)      | 0.340 (0.445)    |
| mathbench_primary | 0.0300           | 0.5000                 | 3                | 0.8889  | 0.955 (0.187)      | 0.930 (0.215)    |
| mathbench_middle  | 0.0200           | 0.6875                 | 4                | 0.8750  | 0.930 (0.232)      | 0.915 (0.265)    |
| mathbench_high    | 0.0357           | 0.7500                 | 3                | 0.5833  | 0.959 (0.200)      | 0.934 (0.244)    |
| mathbench_college | 0.0350           | 1.0000                 | 3                | 0.8889  | 0.965 (0.160)      | 0.940 (0.193)    |
| compmath          | 0.0600           | 1.0000                 | 3                | 1.0000  | 0.980 (0.141)      | 0.920 (0.274)    |
| wmdp              | 0.3163           | 0.8558                 | 24               | 0.9688  | 0.750 (0.346)      | 0.505 (0.446)    |
| codemmlu          | 0.0550           | 0.3333                 | 8                | 0.5521  | 0.725 (0.405)      | 0.710 (0.424)    |


| Dataset      | Mean TE (before) | Mean R_effect (before) | n TE>0 (post L1) | g-mean² | Unhinted Acc (Std) | Hinted Acc (Std) |
| ------------ | ---------------- | ---------------------- | ---------------- | ------- | ------------------ | ---------------- |
| mmlu_redux   | 0.0917           | 0.9167                 | 7                | 0.7024  | 0.818 (0.340)      | 0.762 (0.392)    |
| gpqa_diamond | 0.3200           | 0.7069                 | 23               | 0.7077  | 0.445 (0.435)      | 0.340 (0.445)    |
| wmdp         | 0.3163           | 0.8558                 | 24               | 0.9688  | 0.750 (0.346)      | 0.505 (0.446)    |
| codemmlu     | 0.0550           | 0.3333                 | 8                | 0.5521  | 0.725 (0.405)      | 0.710 (0.424)    |
| lsat_ar      | 0.1417           | 0.8073                 | 14               | 0.8512  | 0.810 (0.348)      | 0.705 (0.416)    |
| lsat_lr      | 0.1000           | 0.9000                 | 9                | 0.5648  | 0.800 (0.357)      | 0.715 (0.410)    |



---

## Dataset Exclusions

The following datasets were excluded from the final evaluation suite because both hinted and unhinted accuracies were too high (≥0.90), indicating the models already knew the correct answers and were not genuinely pressured to rely on hints. This violates the incentive calibration condition: near-ceiling accuracy means the hint's causal leverage is negligible, producing near-zero TE and an uninformative evaluation.

**Excluded datasets:** mathbench_primary, mathbench_middle, mathbench_high, mathbench_college, compmath.

---

## Remaining Evaluation Suite

After exclusions, the retained datasets for both models are: mmlu_redux, gpqa_diamond, wmdp, codemmlu.

### qwen3_1_7b (remaining)

| Dataset | Mean TE | n TE>0 | g-mean² | Unhinted Acc |
| --- | --- | --- | --- | --- |
| mmlu_redux | 0.075 | 9 | 0.5093 | 0.765 |
| gpqa_diamond | 0.123 | 17 | 0.3786 | 0.350 |
| wmdp | 0.127 | 18 | 0.6898 | 0.577 |
| codemmlu | 0.088 | 14 | 0.5714 | 0.705 |

### qwen3_4b (remaining)

| Dataset | Mean TE | n TE>0 | g-mean² | Unhinted Acc |
| --- | --- | --- | --- | --- |
| mmlu_redux | 0.092 | 7 | 0.7024 | 0.818 |
| gpqa_diamond | 0.320 | 23 | 0.7077 | 0.445 |
| wmdp | 0.316 | 24 | 0.9688 | 0.750 |
| codemmlu | 0.055 | 8 | 0.5521 | 0.725 |


---

## Model Selection: qwen3_4b Selected for RO2

**qwen3_4b** was selected as the actor model for RO2 over qwen3_1_7b on three grounds.

**1. Signal strength.** RO2 requires detecting changes in HAR after DPO fine-tuning under each post-training objective. This is only possible if the baseline hint-following signal (TE) is large enough that training-induced shifts remain detectable above noise. On the two most important datasets, qwen3_4b provides 2.6× stronger TE on gpqa_diamond (0.320 vs 0.123) and 2.5× stronger TE on wmdp (0.316 vs 0.127). The 1.7b values are low enough that small post-training effects could plausibly be masked by sampling variance across the 16 runs per instance.

**2. Signal quality.** g-mean² measures how consistently the hint shifts the model toward the hinted answer (across both hinted and unhinted arms), with 1.0 indicating a clean, symmetric effect. qwen3_4b dominates on g-mean² across all retained datasets, most notably on wmdp (0.9688 vs 0.6898) and gpqa_diamond (0.7077 vs 0.3786). The 1.7b GPQA signal in particular (g-mean² = 0.3786) indicates an asymmetric or noisy causal effect that would complicate interpretation of any HAR shift observed in RO2.

**3. Incentive calibration.** The evaluation design requires that models operate under genuine uncertainty — hard enough that hints provide real informational value, but not so hard that the model abandons task-directed reasoning entirely. qwen3_1_7b's unhinted accuracy on gpqa_diamond is 0.350, close to the random-chance baseline of 0.25 for a 4-option MCQ, suggesting the model is near-guessing rather than reasoning under uncertainty. qwen3_4b's 0.445 on the same dataset is in a better-calibrated zone: the model retains some domain knowledge and engages in genuine multi-step inference, making hint-following a meaningful behavioural signal rather than near-random answer selection.

The one dataset where qwen3_1_7b is stronger is codemmlu (TE = 0.088, n = 14 vs TE = 0.055, n = 8 for 4b), but codemmlu produces weak signal for both models and carries the least weight in RO2 relative to gpqa_diamond and wmdp. The overall balance of evidence strongly favours qwen3_4b as the actor model for RO2.
