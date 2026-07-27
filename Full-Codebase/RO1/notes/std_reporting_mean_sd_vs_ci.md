# Why the reported stds are so large — and how to report uncertainty

Question raised (2026-07-10): the `mean (std)` cells in the results tables
look broken — e.g. `mmlu_pro × grader_hacking` has TE = 0.198 (0.305) and
unhinted acc = 0.735 (0.402), stds bigger than or comparable to the means.
Is the std computation wrong?

## Verdict: the computation is correct

The std is `np.std(vals, ddof=1)` (unbiased sample std) over per-instance
values (`gmean2_metrics.py`, `_sample_std`). Verified numerically: the
reported summary stds reproduce exactly from `results_per_instance.json`.
Nothing is miscomputed — the size is a property of the data, not a bug.

## Why the numbers are inherently this big

The per-instance distributions are strongly bimodal, and the std is a
descriptive spread over that bimodal population:

- **TE_before** (mmlu_pro × grader_hacking): 115/200 instances have TE
  exactly 0; the other 85 spread over 0.125–1.0. A spike at zero plus a
  positive tail forces std ≈ 0.3.
- **Unhinted accuracy**: 128/200 instances are 8/8 correct (1.0), 34 are
  0/8 (0.0), few in between — essentially Bernoulli(0.735), whose
  theoretical std ceiling is √(0.735 × 0.265) ≈ 0.44. The observed 0.40 is
  where it *has* to be; it cannot be small.

Two structural causes:

1. Each per-instance value is a proportion out of only **N=8 samples** —
   quantized to multiples of 1/8, carrying binomial noise up to
   √(p(1−p)/8) ≈ 0.18 per arm on top of true between-instance variation.
2. Instances genuinely bifurcate: the model mostly either always follows
   the hint on a question or never does, so values pile up at the ends of
   [0, 1].

## Std vs SEM vs CI — the three numbers

Using TE_before = 0.198, std = 0.305, n = 200:

- **std = 0.305** — spread of *individual instances*: "pick one question
  at random; how far from 0.198 is its TE likely to be?"
- **SEM = std/√n ≈ 0.022** — uncertainty of the *mean itself*: "rerun with
  a fresh 200 questions; how much does the 0.198 wobble?"
- **95% CI half-width ≈ 1.96 × SEM ≈ 0.04** — so TE = 0.198 ± 0.04, i.e.
  the mean is measured precisely even though instances vary wildly.

The tables print `mean (std)`, which readers parse as uncertainty on the
mean — by that (wrong) reading 0.198 (0.305) "spans zero". The actual
estimate precision is ±0.02.

## Reporting convention for this kind of research

Both conventions exist, but for per-instance rate metrics on a benchmark
(accuracy, TE, R_effect, g-mean²) the field norm — including the CoT
faithfulness literature this builds on — is **mean with 95% CI (usually
bootstrapped over instances)**, not mean (std). Two reasons std is
essentially never shown for these:

1. **Uninformative**: for near-binary per-instance values the std is
   almost a deterministic function of the mean (≈ √(p(1−p))) — 0.735
   (0.402) says nothing beyond 0.735 alone.
2. **Misleading**: readers take the parenthesis as measurement error and
   conclude the estimate is noise when it is actually tight.

`mean (std)` remains appropriate where the spread itself is the finding —
the CoT token-length tables (e.g. 3303 ± 3185 tokens is a real observation
about length variability), and mean ± std over training seeds/runs.

## Recommendation for the write-up

- Report `mean [95% CI]` for TE, R_effect, accuracies, g-mean². Bootstrap
  over instances is cleanest (values are bounded and bimodal); a normal
  ±1.96 × SEM is defensible at n=200.
- Keep `mean (std)` only for CoT token-length tables.
- If the instance-level heterogeneity is worth putting on record (the TE
  bimodality is itself interesting), state it once in prose or show a
  histogram — not a std column.
- If tables keep an SD column, label it explicitly "SD across instances"
  so it can't be read as an error bar.

Code note: per-instance values are already saved in
`results_per_instance.json`, so adding bootstrap CIs to `reporting.py` is
a computation-only change — no re-inference or re-monitoring needed.
