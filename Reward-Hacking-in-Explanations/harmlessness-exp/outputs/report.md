# Harmlessness preference-pair pipeline — report

- Sampled prompts (denominator): **300**
- Responses per prompt (N): 8, max_tokens: 8192

## Filtering funnel
- filter-1 (safety-label diversity): dropped **275 / 300** (91.7%) → 25 enter scoring
- filter-3 (chosen=Safe & rejected=Unsafe): dropped **7 / 25** (28.0%)
- **End-to-end survival rate: 18 / 300 = 6.0%**

## Cost-spread distribution (rejected − chosen)
- mean: 8.485  median: 6.893
- min: 4.578  max: 14.938
- q25: 5.958  q75: 10.617
- histogram: `/root/MSc_Project_code_repo/Reward-Hacking-in-Explanations/harmlessness-exp/outputs/cost_spread.png`

## Sign-convention sanity check
- mean Beaver cost of **Unsafe** answers: 2.1554 (n=90)
- mean Beaver cost of **Safe**   answers: -2.9607 (n=110)
- Unsafe > Safe (expected)? **PASS**

## Artifacts
- final pairs: `/root/MSc_Project_code_repo/Reward-Hacking-in-Explanations/harmlessness-exp/outputs/final_pairs.joblib` (18 pairs)
- sample pairs: `/root/MSc_Project_code_repo/Reward-Hacking-in-Explanations/harmlessness-exp/outputs/sample_pairs.json` (10 pairs)


-----

- Sign-convention check PASSES with a clean margin: Unsafe mean cost +2.16 vs Safe −2.96 — a ~5-point separation. Beaver and Qwen3Guard agree, and the "higher cost = more harmful" convention is confirmed. The earlier guard-template worry is fully resolved: the guard is emitting a real mix (90 Unsafe / 110 Safe among survivors), not silently returning None.
- Cost spread is healthy: chosen↔rejected gap averages 8.5 (median 6.9, up to 14.9). Where pairs form, they're genuine safe-vs-harmful contrasts, not noise.
- So the 18 pairs are high quality. Eyeball sample_pairs.json to confirm, but the stats say they're legit.

The real finding: low yield, bottlenecked at filter-1

91.7% of prompts (275/300) were dropped at filter-1 — meaning all 8 of Qwen3-4B's responses got the same safety label. Given the model is safety-tuned and the sanity check shows Safe dominating, that overwhelmingly means all-8-Safe: on ~92% of these prompts, Qwen3-4B simply never produces an unsafe answer, so there's nothing to contrast against.
                                                                                                                                                                   That's the suitability signal: PKU-SafeRLHF-30K prompts are mostlyd Qwen3-4B. Only ~8% sit in the sweet spot where it sometimescomplies unsafely and sometimes refuses.                                                                                                                          
filter-3 then drops 7/25 where Beaver's cheapest/most-costly pick didn't match the guard's safe/unsafe — normal judge disagreement, correctly filtered. Net: 6%    end-to-end yield.
                                                                                                                                                                   What this implies for using the dataset
                                                                                                                                                                   - At 6%, the ~6,975 unique prompts in PKU-SafeRLHF-30K cap you at this self-generation setup. Fine for a small/eval set; thin if youwanted thousands for DPO.                                                                                                                                          - Levers to raise yield, if you need more:
  - Bump N (8 → 16/32): more samples = more chances to catch a rare unsafe generation. Helps, with diminishing returns — prompts the model never fails on won't be rescued.
  - Pre-select harder prompts (adversarial categories) instead of a random 300.
  - Elicit more unsafe responses: higher temperature, or a less-al
- Before concluding, check two numbers from the stage 02/03 logs: the FINAL truncated: X% line and stage 03's Invalid (no visible answer): X%. If truncation is
nontrivial, some of the 275 were dropped for having <2 valid respoment) — and raising max_tokens would recover them. Givenmax_tokens=8192 with thinking mode, this is worth confirming.

Bottom line

The pipeline works and produces clean pairs — but PKU-SafeRLHF-30K + self-generation with a safety-tuned Qwen3-4B is a low-yield source (~6%, ~420 pairs max). Whether that's "suitable" depends on how many pairs you need: goodity set, too thin for large-scale DPO without raising N orpre-filtering to harder prompts.    