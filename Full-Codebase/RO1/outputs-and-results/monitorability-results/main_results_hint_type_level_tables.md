# Main results — monitorability per hint type, pooled over datasets

## Per-model results

### Base — `Qwen/Qwen3-4B`

| Metric | Professor hint | Grader-hacking hint | Unethical hint |
|---|---|---|---|
| g-mean² (mean of 8 datasets) [paired-boot 95% CI] | 0.743 [0.717, 0.769] | 0.517 [0.485, 0.549] | 0.893 [0.881, 0.904] |
| Mean TE on V (mean of 8 datasets) [paired-boot 95% CI] | 0.474 [0.450, 0.497] | 0.442 [0.421, 0.465] | 0.724 [0.708, 0.739] |
| Mean R_effect on V (mean of 8 datasets) [paired-boot 95% CI] | 0.868 [0.847, 0.888] | 0.873 [0.854, 0.892] | 0.930 [0.920, 0.940] |
| % with TE > 0 (mean of 8 datasets) | 42% | 44% | 86% |

### Harmlessness DPO — `er-abd/harmless-dpo-qwen3-4B-step-735`

| Metric | Professor hint | Grader-hacking hint | Unethical hint |
|---|---|---|---|
| g-mean² (mean of 8 datasets) [paired-boot 95% CI] | 0.759 [0.730, 0.787] | 0.581 [0.547, 0.614] | 0.908 [0.897, 0.919] |
| Mean TE on V (mean of 8 datasets) [paired-boot 95% CI] | 0.460 [0.437, 0.484] | 0.454 [0.431, 0.477] | 0.705 [0.689, 0.721] |
| Mean R_effect on V (mean of 8 datasets) [paired-boot 95% CI] | 0.860 [0.837, 0.881] | 0.881 [0.862, 0.900] | 0.934 [0.924, 0.944] |
| % with TE > 0 (mean of 8 datasets) | 39% | 43% | 83% |

### Anti-sycophancy DPO — `er-abd/anti-syc-dpo-qwen3-4B-step-520`

| Metric | Professor hint | Grader-hacking hint | Unethical hint |
|---|---|---|---|
| g-mean² (mean of 8 datasets) [paired-boot 95% CI] | 0.744 [0.713, 0.774] | 0.566 [0.533, 0.599] | 0.911 [0.900, 0.922] |
| Mean TE on V (mean of 8 datasets) [paired-boot 95% CI] | 0.448 [0.423, 0.473] | 0.431 [0.409, 0.454] | 0.698 [0.682, 0.714] |
| Mean R_effect on V (mean of 8 datasets) [paired-boot 95% CI] | 0.843 [0.820, 0.866] | 0.872 [0.853, 0.891] | 0.931 [0.920, 0.940] |
| % with TE > 0 (mean of 8 datasets) | 37% | 43% | 81% |

## Comparison 1: Base vs harmlessness-trained DPO

Δ = `er-abd/harmless-dpo-qwen3-4B-step-735` − `Qwen/Qwen3-4B`

| Metric (Δ = trained − base) [paired-boot 95% CI] | Professor hint | Grader-hacking hint | Unethical hint |
|---|---|---|---|
| Δ g-mean² (mean of 8 datasets) | +0.016 [-0.019, +0.049] | +0.064 [+0.031, +0.096] | +0.015 [+0.004, +0.027] |
| Δ Mean TE on V (mean of 8 datasets) | -0.014 [-0.036, +0.009] | +0.011 [-0.009, +0.032] | -0.019 [-0.032, -0.005] |
| Δ Mean R_effect on V (mean of 8 datasets) | -0.008 [-0.028, +0.012] | +0.008 [-0.011, +0.027] | +0.004 [-0.004, +0.012] |
| Δ % with TE > 0 (mean of 8 datasets) | -2 pp [-4, -0] | -1 pp [-4, +1] | -3 pp [-5, -2] |

## Comparison 2: Base vs anti-sycophancy-trained DPO

Δ = `er-abd/anti-syc-dpo-qwen3-4B-step-520` − `Qwen/Qwen3-4B`

| Metric (Δ = trained − base) [paired-boot 95% CI] | Professor hint | Grader-hacking hint | Unethical hint |
|---|---|---|---|
| Δ g-mean² (mean of 8 datasets) | +0.001 [-0.035, +0.036] | +0.049 [+0.016, +0.081] | +0.019 [+0.007, +0.030] |
| Δ Mean TE on V (mean of 8 datasets) | -0.026 [-0.049, -0.003] | -0.011 [-0.031, +0.008] | -0.026 [-0.040, -0.012] |
| Δ Mean R_effect on V (mean of 8 datasets) | -0.025 [-0.046, -0.003] | -0.001 [-0.020, +0.018] | +0.001 [-0.008, +0.009] |
| Δ % with TE > 0 (mean of 8 datasets) | -5 pp [-7, -3] | -1 pp [-3, +1] | -5 pp [-7, -4] |

---

**Delta tables**: each cell is the trained model's pooled point estimate minus
the base model's, with its own **paired-bootstrap 95% CI**: each replicate
draws, for every dataset, one set of instance ids and reuses it for BOTH
models and all hint types (10,000 resamples, seed 88), so the shared-question
covariance is propagated — these intervals are much tighter than comparing
the two models' separate CIs, which is the correct behavior for a paired
design. A delta CI excluding 0 is a paired test at the 5% level. The
`% with TE > 0` row is in percentage points (pp).

**Paired-boot 95% CI** (every interval in this report): paired cluster
bootstrap over instances — each replicate draws, for every dataset, n instance
ids with replacement from that dataset's shared pool and reuses the same
dataset-wise draws across the three hint types (10,000 resamples, seed 88).
The pooled point estimate is the unweighted mean of the 8 per-dataset
estimates, never a mean over pooled instances. Datasets and hint types are
treated as fixed, so the interval covers question-sampling uncertainty only.

**Accuracy rows are intentionally omitted** from this report: dataset-pooled
accuracy averages disjoint question populations of very different difficulty,
and the unhinted arm cannot vary by hint type. Accuracies are reported at
dataset level and per (dataset, hint type); both are still stored in
`pooled_summary_by_hint.json`.

*V* = level-1-passing instances (TE > 0 and p1 > 0). Each dataset: n = 200
instances, 8 samples per arm. † marks a cell pooled over fewer than 8 datasets.
