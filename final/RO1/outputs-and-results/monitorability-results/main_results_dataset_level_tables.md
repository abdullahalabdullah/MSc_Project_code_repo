# Main results — monitorability pooled over hint types

## Per-model results

### Base — `Qwen/Qwen3-4B`

| Metric | **Academic QA**<br>`mmlu_pro` | **Academic QA**<br>`gpqa_diamond` | **Applied Reasoning**<br>`lsat_ar` | **Applied Reasoning**<br>`lsat_lr` | **Expert Medical**<br>`medxpertqa` | **Legal**<br>`lexam` | **Coding**<br>`codemmlu` | **Safety & Security**<br>`wmdp` |
|---|---|---|---|---|---|---|---|---|
| g-mean² (mean of 3 hints) [paired-boot 95% CI] | 0.800 [0.757, 0.843] | 0.647 [0.611, 0.684] | 0.776 [0.729, 0.821] | 0.529 [0.456, 0.604] | 0.700 [0.656, 0.743] | 0.795 [0.756, 0.832] | 0.756 [0.705, 0.807] | 0.738 [0.696, 0.780] |
| Mean TE on V (mean of 3 hints) [paired-boot 95% CI] | 0.596 [0.553, 0.639] | 0.599 [0.558, 0.640] | 0.442 [0.403, 0.480] | 0.449 [0.395, 0.505] | 0.633 [0.593, 0.673] | 0.555 [0.519, 0.590] | 0.496 [0.452, 0.541] | 0.603 [0.558, 0.649] |
| Mean R_effect on V (mean of 3 hints) [paired-boot 95% CI] | 0.937 [0.897, 0.971] | 0.826 [0.783, 0.866] | 0.901 [0.863, 0.936] | 0.879 [0.822, 0.932] | 0.905 [0.865, 0.941] | 0.861 [0.819, 0.899] | 0.907 [0.856, 0.952] | 0.906 [0.862, 0.946] |
| % with TE > 0 (mean of 3 hints) | 59% | 70% | 54% | 36% | 67% | 69% | 48% | 58% |
| Hinted accuracy (mean of 3 hints) [paired-boot 95% CI] | 0.456 [0.412, 0.499] | 0.300 [0.253, 0.347] | 0.614 [0.568, 0.659] | 0.687 [0.639, 0.735] | 0.064 [0.041, 0.088] | 0.325 [0.283, 0.367] | 0.555 [0.505, 0.603] | 0.414 [0.364, 0.465] |
| Unhinted accuracy (single arm) [95% CI] | 0.719 [0.663, 0.772] | 0.512 [0.455, 0.567] | 0.782 [0.735, 0.825] | 0.834 [0.789, 0.877] | 0.127 [0.087, 0.169] | 0.585 [0.529, 0.641] | 0.763 [0.709, 0.816] | 0.679 [0.622, 0.736] |

### Harmlessness DPO — `er-abd/harmless-dpo-qwen3-4B-step-735`

| Metric | **Academic QA**<br>`mmlu_pro` | **Academic QA**<br>`gpqa_diamond` | **Applied Reasoning**<br>`lsat_ar` | **Applied Reasoning**<br>`lsat_lr` | **Expert Medical**<br>`medxpertqa` | **Legal**<br>`lexam` | **Coding**<br>`codemmlu` | **Safety & Security**<br>`wmdp` |
|---|---|---|---|---|---|---|---|---|
| g-mean² (mean of 3 hints) [paired-boot 95% CI] | 0.843 [0.801, 0.882] | 0.684 [0.646, 0.721] | 0.796 [0.749, 0.841] | 0.550 [0.474, 0.629] | 0.746 [0.700, 0.789] | 0.855 [0.822, 0.886] | 0.722 [0.662, 0.782] | 0.799 [0.759, 0.838] |
| Mean TE on V (mean of 3 hints) [paired-boot 95% CI] | 0.590 [0.545, 0.633] | 0.605 [0.560, 0.648] | 0.412 [0.373, 0.452] | 0.453 [0.397, 0.512] | 0.664 [0.627, 0.701] | 0.519 [0.483, 0.555] | 0.464 [0.417, 0.512] | 0.610 [0.563, 0.657] |
| Mean R_effect on V (mean of 3 hints) [paired-boot 95% CI] | 0.954 [0.924, 0.979] | 0.845 [0.803, 0.883] | 0.914 [0.877, 0.948] | 0.865 [0.797, 0.929] | 0.921 [0.885, 0.952] | 0.862 [0.816, 0.903] | 0.869 [0.810, 0.922] | 0.903 [0.858, 0.944] |
| % with TE > 0 (mean of 3 hints) | 56% | 67% | 53% | 32% | 67% | 66% | 44% | 56% |
| Hinted accuracy (mean of 3 hints) [paired-boot 95% CI] | 0.490 [0.441, 0.538] | 0.305 [0.256, 0.354] | 0.652 [0.608, 0.695] | 0.710 [0.663, 0.757] | 0.066 [0.043, 0.092] | 0.338 [0.293, 0.383] | 0.587 [0.535, 0.638] | 0.421 [0.368, 0.474] |
| Unhinted accuracy (single arm) [95% CI] | 0.735 [0.679, 0.789] | 0.507 [0.447, 0.565] | 0.814 [0.770, 0.856] | 0.847 [0.803, 0.887] | 0.135 [0.096, 0.177] | 0.567 [0.510, 0.624] | 0.756 [0.702, 0.808] | 0.673 [0.614, 0.731] |

### Anti-sycophancy DPO — `er-abd/anti-syc-dpo-qwen3-4B-step-520`

| Metric | **Academic QA**<br>`mmlu_pro` | **Academic QA**<br>`gpqa_diamond` | **Applied Reasoning**<br>`lsat_ar` | **Applied Reasoning**<br>`lsat_lr` | **Expert Medical**<br>`medxpertqa` | **Legal**<br>`lexam` | **Coding**<br>`codemmlu` | **Safety & Security**<br>`wmdp` |
|---|---|---|---|---|---|---|---|---|
| g-mean² (mean of 3 hints) [paired-boot 95% CI] | 0.836 [0.794, 0.875] | 0.663 [0.625, 0.699] | 0.802 [0.756, 0.847] | 0.549 [0.474, 0.628] | 0.761 [0.720, 0.803] | 0.857 [0.820, 0.892] | 0.691 [0.631, 0.750] | 0.764 [0.724, 0.806] |
| Mean TE on V (mean of 3 hints) [paired-boot 95% CI] | 0.540 [0.493, 0.587] | 0.596 [0.554, 0.637] | 0.426 [0.382, 0.471] | 0.441 [0.382, 0.504] | 0.636 [0.600, 0.672] | 0.491 [0.458, 0.525] | 0.465 [0.422, 0.507] | 0.608 [0.566, 0.651] |
| Mean R_effect on V (mean of 3 hints) [paired-boot 95% CI] | 0.942 [0.905, 0.975] | 0.852 [0.811, 0.889] | 0.891 [0.847, 0.931] | 0.842 [0.771, 0.909] | 0.920 [0.888, 0.949] | 0.850 [0.807, 0.891] | 0.866 [0.814, 0.913] | 0.892 [0.848, 0.934] |
| % with TE > 0 (mean of 3 hints) | 56% | 68% | 47% | 32% | 63% | 63% | 46% | 54% |
| Hinted accuracy (mean of 3 hints) [paired-boot 95% CI] | 0.500 [0.453, 0.547] | 0.312 [0.265, 0.360] | 0.674 [0.627, 0.718] | 0.714 [0.667, 0.762] | 0.066 [0.043, 0.092] | 0.350 [0.304, 0.397] | 0.584 [0.531, 0.635] | 0.428 [0.378, 0.480] |
| Unhinted accuracy (single arm) [95% CI] | 0.731 [0.675, 0.785] | 0.512 [0.453, 0.570] | 0.826 [0.784, 0.866] | 0.845 [0.800, 0.887] | 0.128 [0.089, 0.169] | 0.551 [0.493, 0.608] | 0.762 [0.709, 0.813] | 0.679 [0.623, 0.736] |

## Comparison 1: Base vs harmlessness-trained DPO

Δ = `er-abd/harmless-dpo-qwen3-4B-step-735` − `Qwen/Qwen3-4B`

| Metric (Δ = trained − base) [paired-boot 95% CI] | **Academic QA**<br>`mmlu_pro` | **Academic QA**<br>`gpqa_diamond` | **Applied Reasoning**<br>`lsat_ar` | **Applied Reasoning**<br>`lsat_lr` | **Expert Medical**<br>`medxpertqa` | **Legal**<br>`lexam` | **Coding**<br>`codemmlu` | **Safety & Security**<br>`wmdp` |
|---|---|---|---|---|---|---|---|---|
| Δ g-mean² (mean of 3 hints) | +0.042 [+0.005, +0.079] | +0.037 [+0.001, +0.071] | +0.020 [-0.034, +0.076] | +0.021 [-0.058, +0.100] | +0.046 [+0.012, +0.078] | +0.060 [+0.024, +0.096] | -0.034 [-0.096, +0.025] | +0.062 [+0.021, +0.100] |
| Δ Mean TE on V (mean of 3 hints) | -0.006 [-0.041, +0.030] | +0.006 [-0.027, +0.038] | -0.030 [-0.064, +0.006] | +0.004 [-0.039, +0.049] | +0.031 [+0.004, +0.058] | -0.036 [-0.066, -0.005] | -0.032 [-0.076, +0.015] | +0.007 [-0.033, +0.048] |
| Δ Mean R_effect on V (mean of 3 hints) | +0.017 [-0.007, +0.046] | +0.019 [-0.016, +0.054] | +0.013 [-0.025, +0.053] | -0.014 [-0.059, +0.030] | +0.016 [-0.004, +0.036] | +0.001 [-0.037, +0.039] | -0.038 [-0.089, +0.014] | -0.002 [-0.045, +0.041] |
| Δ % with TE > 0 (mean of 3 hints) | -3 pp [-6, -0] | -3 pp [-6, +1] | -1 pp [-4, +3] | -4 pp [-7, -1] | -1 pp [-3, +2] | -3 pp [-7, +1] | -4 pp [-7, -1] | -1 pp [-5, +2] |
| Δ Hinted accuracy (mean of 3 hints) | +0.035 [+0.021, +0.048] | +0.005 [-0.006, +0.016] | +0.038 [+0.024, +0.052] | +0.023 [+0.010, +0.036] | +0.002 [-0.003, +0.008] | +0.013 [-0.001, +0.027] | +0.033 [+0.021, +0.045] | +0.007 [-0.008, +0.023] |
| Δ Unhinted accuracy (single arm) | +0.016 [-0.001, +0.035] | -0.005 [-0.029, +0.019] | +0.033 [+0.013, +0.052] | +0.013 [-0.005, +0.031] | +0.008 [-0.009, +0.025] | -0.018 [-0.044, +0.008] | -0.007 [-0.021, +0.007] | -0.006 [-0.030, +0.016] |

## Comparison 2: Base vs anti-sycophancy-trained DPO

Δ = `er-abd/anti-syc-dpo-qwen3-4B-step-520` − `Qwen/Qwen3-4B`

| Metric (Δ = trained − base) [paired-boot 95% CI] | **Academic QA**<br>`mmlu_pro` | **Academic QA**<br>`gpqa_diamond` | **Applied Reasoning**<br>`lsat_ar` | **Applied Reasoning**<br>`lsat_lr` | **Expert Medical**<br>`medxpertqa` | **Legal**<br>`lexam` | **Coding**<br>`codemmlu` | **Safety & Security**<br>`wmdp` |
|---|---|---|---|---|---|---|---|---|
| Δ g-mean² (mean of 3 hints) | +0.036 [-0.004, +0.076] | +0.015 [-0.020, +0.049] | +0.026 [-0.029, +0.081] | +0.020 [-0.056, +0.094] | +0.061 [+0.023, +0.098] | +0.062 [+0.024, +0.101] | -0.064 [-0.116, -0.016] | +0.026 [-0.011, +0.063] |
| Δ Mean TE on V (mean of 3 hints) | -0.056 [-0.096, -0.017] | -0.003 [-0.034, +0.027] | -0.016 [-0.049, +0.019] | -0.008 [-0.051, +0.036] | +0.003 [-0.027, +0.034] | -0.063 [-0.097, -0.031] | -0.031 [-0.066, +0.003] | +0.005 [-0.033, +0.043] |
| Δ Mean R_effect on V (mean of 3 hints) | +0.006 [-0.018, +0.032] | +0.026 [-0.008, +0.061] | -0.010 [-0.051, +0.029] | -0.037 [-0.089, +0.010] | +0.015 [-0.014, +0.045] | -0.011 [-0.045, +0.025] | -0.041 [-0.090, +0.003] | -0.013 [-0.055, +0.029] |
| Δ % with TE > 0 (mean of 3 hints) | -3 pp [-5, +0] | -2 pp [-5, +2] | -7 pp [-10, -3] | -4 pp [-7, -1] | -4 pp [-7, -1] | -6 pp [-10, -1] | -2 pp [-5, +1] | -3 pp [-7, +0] |
| Δ Hinted accuracy (mean of 3 hints) | +0.045 [+0.030, +0.060] | +0.012 [+0.002, +0.023] | +0.060 [+0.046, +0.073] | +0.027 [+0.015, +0.039] | +0.002 [-0.003, +0.008] | +0.025 [+0.008, +0.042] | +0.029 [+0.017, +0.041] | +0.014 [-0.002, +0.030] |
| Δ Unhinted accuracy (single arm) | +0.012 [-0.005, +0.030] | 0.000 [-0.025, +0.025] | +0.045 [+0.025, +0.066] | +0.010 [-0.004, +0.026] | +0.001 [-0.012, +0.014] | -0.034 [-0.067, -0.003] | -0.001 [-0.017, +0.015] | 0.000 [-0.020, +0.019] |

---

**Delta tables**: each cell is the trained model's pooled point estimate minus
the base model's, with its own **paired-bootstrap 95% CI**: each replicate
draws one set of instance ids per dataset and reuses it for BOTH models and
all three hint types (10,000 resamples, seed 88), so the shared-question
covariance is propagated — these intervals are much tighter than comparing the
two models' separate CIs, which is the correct behavior for a paired design.
Each model averages over its own defined subset of the drawn ids, so
level-1-filtering differences are propagated too. A delta CI excluding 0 is a
paired test at the 5% level. The `% with TE > 0` row is in percentage points
(pp).

**CI conventions** (per-model tables) — the two interval types are not the same
estimator:

- **Paired-boot 95% CI** (pooled rows): paired cluster bootstrap over instances —
  each replicate draws n instance ids with replacement once and reuses that same
  draw across the three hint types (10,000 resamples, seed 88). The pooled point
  estimate is the unweighted mean of the three per-hint-type estimates
  (professor, grader_hacking, unethical), never a mean over pooled instances.
  Hint types are treated as fixed, so the interval covers question-sampling
  uncertainty only.
- **95% CI** (unhinted accuracy row): plain percentile bootstrap over instances —
  the unhinted arm is a single condition, shared verbatim by all hint types, so
  no pooling applies.

*V* = level-1-passing instances (TE > 0 and p1 > 0). Each dataset: n = 200
instances, 8 samples per arm. † marks a cell pooled over fewer than 3 hint types.
