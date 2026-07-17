# Length-confound bucket analysis — Base vs Harmless-DPO (step 735)

**Question.** The harmlessness-DPO model (`er-abd_harmless-dpo-qwen3-4B-step-735`)
generates longer CoTs than base (`Qwen_Qwen3-4B`). Longer CoTs tend to be more
monitorable. So if the two models differ in monitorability (g-mean²), is that a
**genuine training effect** or just an artifact of **response length**?

**Approach.** Take every per-instance g-mean² that *enters* the metric (Level-1
pass: `TE_i > 0 and p1_i > 0`, and `gmean2_i` defined). For each instance use the
mean token length of its 8 **hinted** responses (`mean_cot_tokens_hinted`) and
bucket by length. Within a bucket both models sit at ~the same length, so a
within-bucket difference is closer to a pure training effect. Pooled over **all
24 `(dataset, hint)` slices** (8 datasets × 3 hint types: `grader_hacking`,
`professor`, `unethical`) — all present in both models, none skipped.

- Script: `monitorability-eval/scripts/10_length_bucket_gmean2.py`
- Command: `python3 monitorability-eval/scripts/10_length_bucket_gmean2.py 500`
- Bucket width: **500 tokens**
- Entering instances: **BASE n = 2750**, **HARM n = 2451**

## Results (pooled, 500-token buckets)

| bucket (tok) | BASE n | BASE g² | HARM n | HARM g² |
|---|---|---|---|---|
| 0–500 | 11 | 0.830 | 0 | — |
| 500–1000 | 330 | 0.753 | 105 | 0.840 |
| 1000–1500 | 419 | 0.764 | 263 | 0.744 |
| 1500–2000 | 350 | 0.779 | 254 | 0.829 |
| 2000–2500 | 256 | 0.797 | 271 | 0.816 |
| 2500–3000 | 177 | 0.797 | 221 | 0.817 |
| 3000–3500 | 145 | 0.760 | 156 | 0.833 |
| 3500–4000 | 127 | 0.779 | 133 | 0.835 |
| 4000–4500 | 90 | 0.742 | 119 | 0.767 |
| 4500–5000 | 98 | 0.730 | 80 | 0.808 |
| 5000–5500 | 90 | 0.718 | 88 | 0.697 |
| 5500–6000 | 84 | 0.737 | 68 | 0.782 |
| 6000–6500 | 85 | 0.774 | 66 | 0.782 |
| 6500–7000 | 76 | 0.704 | 73 | 0.688 |
| 7000–7500 | 45 | 0.860 | 63 | 0.793 |
| 7500–8000 | 58 | 0.777 | 61 | 0.834 |
| 8000–8500 | 51 | 0.801 | 53 | 0.718 |
| 8500–9000 | 39 | 0.814 | 52 | 0.815 |
| 9000–9500 | 39 | 0.772 | 48 | 0.809 |
| 9500–10000 | 31 | 0.795 | 39 | 0.766 |
| 10000–10500 | 27 | 0.817 | 42 | 0.784 |
| 10500–11000 | 26 | 0.784 | 42 | 0.744 |
| 11000–11500 | 22 | 0.713 | 35 | 0.659 |
| 11500–12000 | 19 | 0.736 | 31 | 0.769 |
| 12000–12500 | 20 | 0.684 | 22 | 0.779 |
| 12500–13000 | 11 | 0.624 | 22 | 0.626 |
| 13000–13500 | 10 | 0.860 | 13 | 0.717 |
| 13500–14000 | 4 | 0.719 | 12 | 0.756 |
| 14000–14500 | 6 | 0.773 | 8 | 0.715 |
| 14500–15000 | 3 | 0.667 | 5 | 0.600 |
| 15000–15500 | 0 | — | 5 | 1.000 |
| 15500–16000 | 1 | 1.000 | 1 | 1.000 |

**Overall:** BASE n=2750, median len=2524, g²=**0.7679** · HARM n=2451, median
len=3352, g²=**0.7893**

## Interpretation

1. **The length shift is real and substantial.** Median hinted length rises from
   **2524 → 3352 tokens** (+~33%), and the whole HARM distribution is shifted
   right (e.g. base has 330 instances in 500–1000 tok vs HARM's 105; HARM has
   more mass in every bucket above ~2000 tok). This confirms the original
   observation: harmlessness-DPO training makes the model write longer.

2. **The monitorability difference is small.** Overall g-mean² is **0.768 (base)
   vs 0.789 (harmless)** — a gap of only **+0.021**. So even before controlling
   for length, the harmless model is not dramatically different in monitorability.

3. **The gap is NOT explained by length.** Within matched-length buckets the two
   models track each other closely and the sign is mixed — HARM is a little
   higher in many mid-range buckets (500–1000, 1500–2000, 3000–4000, 4500–5000,
   7500–8000) and a little lower in others (1000–1500, 6500–7000, 8000–8500,
   11000–11500). There is **no systematic within-length gap**. If the raw
   difference were purely a length artifact, we would expect HARM to look better
   *only because* it has more long (more-monitorable) responses, and to be
   **equal to base within every bucket** — but instead HARM is often *at least as
   good as* base at the *same* length. So the small overall advantage is not
   manufactured by length composition.

4. **Consistent with the single-slice deep dive.** On `mmlu_pro / grader_hacking`
   alone (the earlier detailed analysis), length-standardization — reweighting
   HARM onto base's length distribution — moved its g-mean² by only **−0.001**
   (0.779 → 0.780 vs base 0.669), i.e. length explained essentially none of that
   slice's gap. The pooled bucket table here shows the same qualitative pattern
   at scale.

### What this confirms

- **Harmlessness-DPO does inflate CoT length** (median +~33%). ✅
- **That length inflation is *not* what is driving the monitorability numbers.**
  Controlling for length (bucketing) leaves at most a small, non-systematic
  monitorability difference. ✅
- **Any monitorability difference between the two models is a (small) property of
  the training itself, not a byproduct of longer responses.** The confound is
  ruled out as the explanation. ✅

## Caveats

- **Pooling mixes heterogeneous slices.** Datasets/hint-types have different
  baseline g² and different length distributions, so a single pooled bucket can
  blend composition across slices. The BASE-vs-HARM comparison is still fair
  (both pooled identically over the same 24 slices), but a pooled bucket mean is
  not a clean "monitorability at length L" for any one task.
- **No CIs in this table.** Bucket means are point estimates. On the one slice
  where CIs were computed, the length-driven component was tightly around zero
  (95% CI [−0.068, 0.059]) while the *existence* of an overall gap was borderline
  — a power limitation, not a confound. For a quotable headline number, use the
  length-standardization + bootstrap-CI approach.
- **Tail buckets (>13,500 tok) are thin** (single digits per model) and noisy;
  read them loosely.
