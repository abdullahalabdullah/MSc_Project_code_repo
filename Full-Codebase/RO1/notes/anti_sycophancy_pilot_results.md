# Anti-Sycophancy DPO Curation — Pilot Run Results (X=500)

Observed results from the pilot curation run. Facts only — numbers as reported by the
pipeline, no interpretation or recommendations.

**Run:** X=500, N=10 (Filter-0) / 10 (Filter-1), PILOT=True, generation model Qwen/Qwen3-4B
(temp 1.0, thinking mode), grader gpt-5-mini (temperature omitted).
Generated 2026-07-17, git `767053411d3ac7741ceb36515ee63d4be443a8b0`.
Source: `RO2/anti-syc-DPO-train-dataset-curation/outputs/pilot/{qa_report.md, pilot_gates.json, manifest.json}`.

## Data acquisition (Stage A/B)

- TriviaQA `rc.nocontext` train: 138,384 rows → **76,523 unique questions** after dedup.
- Training pool: **500**; eval slice (frozen): **1,500**.
- OOD eval (SycophancyEval "Are you sure?"): **4,888 gradable records**, all kept
  (aqua_mc 254, math_mc_cot 1000, mmlu_mc_cot 1000, trivia_qa 1000, truthful_qa_mc 817,
  truthful_qa 817).

## Funnel

| Stage boundary | Count |
|---|---|
| sampled_pool | 500 |
| filter0_pass | 248 |
| instances | 496 |
| filter1_pass | 140 |
| pairs_selected | 140 |
| pairs_post_balance | **136** |

## Filter-0 (knowledge filter)

- **Pass rate: 248/500 = 49.6%.**
- Consistency distribution of the 248 survivors: 70–80: 16, 80–90: 15, 90–99: 21,
  100%: 196. (Failed below 70%: 252.)

## Filter-1 (contrast sampling)

- **Pass: 140/496 instances.**
  - wrong_opinion: **72/248 = 29.0%**
  - right_opinion: **68/248 = 27.4%**
- Per-stratum Filter-1 survival (by Filter-0 consistency stratum):

  | Stratum | Instances | Pass | Survival |
  |---|---|---|---|
  | 70–80 | 32 | 13 | 40.6% |
  | 80–90 | 30 | 10 | 33.3% |
  | 90–99 | 42 | 16 | 38.1% |
  | 100 | 392 | 101 | 25.8% |

## Baseline sycophancy (syc_fraction, wrong-opinion instances)

- Defined for 247/248 wrong-opinion instances.
- **Mean 0.2, median 0.0.** (syc_fraction = fraction of classified samples that flipped to
  the user's wrong answer `w`.)

## Per-template stats

| Template | Instances | Survival | Wrong-opinion rejected-class rate | Mean syc_fraction |
|---|---|---|---|---|
| T1 | 54 | 22.2% | 28.28% | 0.3 |
| T2 | 61 | 16.4% | 21.10% | 0.2 |
| T3 | 66 | 22.7% | 25.86% | 0.3 |
| T4 | 57 | 33.3% | 29.80% | 0.3 |
| T5 | 61 | 36.1% | 26.82% | 0.2 |
| T6 | 67 | 16.4% | 12.11% | 0.1 |
| T7 | 62 | 27.4% | 48.65% | 0.5 |
| T8 | 68 | 50.0% | 8.70% | 0.1 |

(All 8 templates cleared the 2% rejected-class-rate threshold; none dropped.)

## Parse health

- **UNPARSEABLE: 2/9,960 samples = 0.02%** (both from truncation at max_new_tokens; 0 from
  missing final-answer line; 0 from invalid classifier output).

## Completion length (tokens)

- chosen: mean 590.4, median 418.0
- rejected: mean 591.1, median 420.0
- (chosen/rejected mean gap < 0.2% — no >20% length-confound flag.)

## Final dataset

- **136 pairs post-balance** — wrong_opinion 68 / right_opinion 68 (exact 50/50).
- Export: **train = 122, valid = 14** (`outputs/pilot/dpo_dataset`).

## Pilot gates

| Gate | Status | Detail |
|---|---|---|
| 1 — grader audit | **PENDING** | requires 100 hand-labeled grades (≥95% agreement) |
| 2 — yield projection | **PASS** | 136 pilot pairs → projected 5,440 at X=20,000 (≥ 3,000) |
| 3 — template health | **PASS** | all 8 templates ≥ 2% wrong-opinion rejected-class rate; none dropped |
| 4 — parse health | **PASS** | UNPARSEABLE 0.02% (< 10%) |

## Call volume

- Classifier (gpt-5-mini) calls: 9,958.
- Wrong-answer generator unique calls: 1,748.

## Observed timing (80 GB GPU / local API)

- Phase 1 (GPU, Filter-0 gen): ~10–15 min (5,000 completions + ~32 s engine init).
- Phase 2 (local API): Filter-0 grading 5,000 @ ~40 call/s (~2 min); wrong answers
  248 pool + 1,500 eval @ ~9 call/s (~3 min); template assignment instant.
- Phase 3 (GPU, Filter-1 gen): 4,960 completions in 8:55 (~4,350 tok/s), 1 truncated.
- Phase 4 (local API): Filter-1 grading 4,959 @ ~26 call/s (~3 min).
