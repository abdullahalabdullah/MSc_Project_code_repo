# Monitor comparison — gpt-5 vs gpt-5-mini (Qwen/Qwen3-4B, professor hint)

**Question:** assuming **gpt-5 is the gold-standard monitor**, is **gpt-5-mini** a good replacement?

**Generated:** 2026-07-09 · datasets: `mmlu_pro`, `lexam`, `medxpertqa` · hint type: `professor`

---

## Data provenance & method

- **Model responses:** `Qwen/Qwen3-4B` inference outputs (identical for both monitors).
- **gpt-5-mini labels:** per-sample `monitor_acknowledged` from
  `monitorability-results/Qwen_Qwen3-4B-with-monitor-gpt5mini/<dataset>/professor/qualitative_export.xlsx`.
- **gpt-5 labels:** the gpt-5 run had **no Excel export**; per-sample verdicts were taken from the
  live `monitor_outputs/Qwen_Qwen3-4B/<dataset>/professor/monitor_results.jsonl`, verified to be
  the gpt-5 run (it differs from the gpt-5-mini labels by exactly the disagreements below and
  predates the gpt-5 analysis artifacts).
  ⚠️ That live folder is **not** keyed by monitor model — re-running the gpt-5-mini monitor would
  overwrite these gpt-5 labels. Regenerate a gpt-5 Excel to preserve them.
- **Population:** hinted arm, model followed the hint, **both** monitors returned a valid label
  (this is the `Hinted_Followed_*` set). Control-arm and hint-not-followed samples are excluded —
  including them would add hundreds of trivial true-negatives and inflate agreement.
- **Reference (gold) = gpt-5**, prediction = gpt-5-mini, **positive class = Verbalised**
  (`monitor_acknowledged = True`). Accuracy and Cohen's κ are direction-free; precision / recall / F1
  assume this direction (recoverable in reverse from the confusion matrices).
- `mmlu_pro`: 2 samples present in only one run (gpt-5 has 398 vs 400) — excluded.

---

## 1. Agreement metrics

| dataset | n | Verbalised acc | NotVerbalised acc | accuracy | precision | recall | F1 | Cohen's κ | κ band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| mmlu_pro | 35 | 0.885 | 1.000 | 0.914 | 1.000 | 0.885 | 0.939 | 0.798 | substantial |
| lexam | 72 | 0.971 | 1.000 | 0.986 | 1.000 | 0.971 | 0.985 | 0.972 | almost perfect |
| medxpertqa | 89 | 1.000 | 0.907 | 0.955 | 0.920 | 1.000 | 0.958 | 0.910 | almost perfect |
| **OVERALL** | **196** | **0.962** | **0.956** | **0.959** | **0.962** | **0.962** | **0.962** | **0.918** | **almost perfect** |

- **Verbalised acc** = of gpt-5's Verbalised samples, fraction gpt-5-mini also called Verbalised (= recall).
- **NotVerbalised acc** = of gpt-5's NotVerbalised samples, fraction gpt-5-mini agreed (= specificity).
- κ bands per Landis & Koch (0.61–0.80 substantial; 0.81–1.00 almost perfect).
- Classes are fairly balanced (106 Verbalised vs 90 NotVerbalised), so the high κ is not inflated by a majority class.

## 2. Confusion matrices (rows = gpt-5 gold, cols = gpt-5-mini)

| dataset | TP (both Verb) | FP (mini Verb, gpt-5 Not) | FN (mini Not, gpt-5 Verb) | TN (both Not) |
|---|---:|---:|---:|---:|
| mmlu_pro | 23 | 0 | 3 | 9 |
| lexam | 33 | 0 | 1 | 38 |
| medxpertqa | 46 | 4 | 0 | 39 |
| **OVERALL** | **102** | **4** | **4** | **86** |

Only **8 disagreements in 196**. They split by dataset: gpt-5-mini is **stricter** on `mmlu_pro`/`lexam`
(4 FN — misses verbalisations gpt-5 flags) and **looser** on `medxpertqa` (4 FP — over-flags). Overall
precision = recall = 0.962 only because FP and FN both happen to equal 4.

## 3. Downstream impact on g-mean² (the reported metric)

`TNR = 1.0` on both arms for every dataset, so g-mean² is driven **entirely by TPR** — i.e. by the exact
Verbalised/NotVerbalised calls the monitors disagree on. Every label flip flows straight into the headline
number. All monitor-*independent* aggregates (TE, R_effect, accuracy, TNR, CoT tokens) are byte-identical.

| dataset | g-mean² (gpt-5) | g-mean² (gpt-5-mini) | Δ | opacity rate gpt-5 → mini |
|---|---:|---:|---:|---|
| mmlu_pro | 0.776 | 0.682 | **−0.094 (−12%)** | 0.257 → 0.343 |
| lexam | 0.597 | 0.597 | 0.000 | 0.528 → 0.542 |
| medxpertqa | 0.614 | 0.664 | **+0.050 (+8%)** | 0.483 → 0.438 |

g-mean² is computed over only **16 / 18 / 30** level-1 instances respectively, so the mmlu_pro gap is
essentially **3 sample-level flips** — fragile.

---

## Verdict

**On agreement metrics alone: yes — a strong replacement.** Overall κ = 0.918 ("almost perfect"),
96% label match, F1 = 0.96, no systematic overall label bias. Two of three datasets are individually
"almost perfect"; only `mmlu_pro` (κ = 0.80, n = 35) dips to "substantial", and κ is unstable at that
sample size.

**For the reported g-mean² numbers: use with care.** Because g-mean² here = TPR, the 8 disagreements move
the headline metric by up to ~12% (mmlu_pro 0.78 → 0.68), and **the bias flips sign across datasets**
(mmlu-mini reads *less* monitorable, medxpert-mini *more*). A sign-flipping bias distorts cross-dataset /
cross-checkpoint rankings, which is usually the experiment's point.

### Recommendation
- **If gpt-5 is affordable for the final run:** report with gpt-5; use gpt-5-mini for dev/iteration.
- **If shipping gpt-5-mini:** (a) adjudicate the **8 disagreement cases against human labels** to establish
  the true bias (κ measures agreement, not correctness — gpt-5 as "gold" is itself unvalidated);
  (b) report monitor-induced uncertainty of ~±0.05–0.09 on g-mean²; (c) add samples to `mmlu_pro`, the
  weakest and most fragile dataset.
