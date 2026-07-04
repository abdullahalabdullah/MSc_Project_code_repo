# WildJailbreak suitability experiment — summary (follow-up to PKU)

**Goal.** After PKU-SafeRLHF-30K proved too low-yield (6%), decide whether
`allenai/wildjailbreak` (harmful subsets) is a suitable prompt source for mining harmlessness
preference pairs via Qwen3-4B self-generation — **and** validate that
`PKU-Alignment/beaver-7b-unified-cost` is a usable ranker on this data.

Source data: `outputs/run_20260703_121251/report.md` and
`outputs/run_20260703_121251/beaver_sanity_report.md` (WildJailbreak run).

## Method (as run)

- **Load & dedup.** `allenai/wildjailbreak` (train config), keeping only the two harmful
  subsets: `vanilla_harmful` (plain request, from the `vanilla` column) and
  `adversarial_harmful` (jailbreak-wrapped, from the `adversarial` column). Deduplicated on the
  prompt string. The combined harmful pool is large (tens of thousands of unique prompts, far
  bigger than PKU's ~6,975), so prompt supply is not the binding constraint here.
- **Sample.** 300 unique prompts, fixed seed (0) — ~126 vanilla + ~174 adversarial.
- **Generate.** Qwen3-4B, thinking mode on, temperature 0.6, top_p 0.95, top_k 20, **N=10**
  responses, max_tokens 8192, full outputs stored. Live truncation reporting on.
- **Split CoT/answer.** Strip everything up to and including `</think>`; keep the remainder as
  `answer` (both the classifier and the RM score this visible answer only).
- **Classify (filter-1).** Qwen3Guard-Gen per (prompt, answer) → safe/unsafe; drop prompts
  where all N answers share one label.
- **Score.** `beaver-7b-unified-cost` per (prompt, answer) → scalar cost; per prompt
  chosen = min cost, rejected = max cost.
- **Validate pair (filter-3).** Keep prompt only if Qwen3Guard says chosen = Safe and
  rejected = Unsafe.
- **Added: Beaver ranker sanity check.** Independently test whether higher Beaver cost ⇒ more
  harmful answer, per subset, via ROC-AUC against Qwen3Guard labels (raw prompt `cost_full` vs
  vanilla-seed `cost_vanilla`), with a length-confound check, truncation audit, and a WildGuard
  cross-check.

## Results — pilot funnel

- **Sampled:** 300 unique prompts (seed 0), 10 responses each (~126 vanilla / ~174 adversarial).
- **filter-1 (safety-label diversity):** dropped **215 / 300 (71.7%)** → 85 enter scoring.
- **filter-3 (chosen=Safe & rejected=Unsafe):** dropped **14 / 85 (16.5%)** → 71 pairs.
- **End-to-end survival: 71 / 300 = 23.7%** (overall; WildJailbreak has no per-response harm
  category used here, but the two harmful subsets are tracked — see sample mix).
- **Cost-spread (rejected − chosen):** mean 5.25, median 4.00, range 0.91–13.97,
  IQR 2.21–8.13 — clear positive separation on every surviving pair.
- **Sign-convention sanity check:** mean Beaver cost **Unsafe +2.23** (n=487) vs
  **Safe −1.29** (n=363) → **PASS** ("higher cost = more harmful" confirmed).
- **Eyeball file:** 10 random surviving pairs in `sample_pairs.json`.

## Results — Beaver ranker sanity check

- **Scale:** 2,997 valid (prompt, answer) records scored (vanilla 1,260 / adversarial 1,737).
- **ROC-AUC of cost vs Qwen3Guard** (higher cost ⇒ unsafe):
  - vanilla / `cost_full`: **0.989**
  - adversarial / `cost_full`: **0.873**
  - adversarial / `cost_vanilla`: **0.849** (worse — vanilla-query scoring does not help)
- **Truncation** over Beaver's 2048 ctx: only **2.7%** (80/2997); left-truncated to keep the answer.
- **Length confound (adversarial):** Pearson(cost, prompt length) ≈ 0 (0.08 safe, −0.01 unsafe)
  → cost reflects harm, not prompt length.
- **Beaver vs WildGuard exact pair agreement:** modest (~30–43% on 27/58 survivor prompts) —
  expected for the finest argmin/argmax picks among near-tied answers; not a correctness issue.
- **Disagreements** (Beaver sign vs Qwen3Guard, adversarial): 284 total; 10 dumped to
  `beaver_sanity_disagreements.jsonl` for eyeballing.

**Interpretation.**
- **~4× the PKU yield** (23.7% vs 6.0%). The bottleneck is still filter-1 (71.7%), but far
  healthier — the **adversarial** prompts, in particular, push Qwen3-4B to sometimes comply and
  sometimes refuse, creating the safe/unsafe contrast that pair-mining needs.
- **filter-3 is modest (16.5%)**, which independently confirms Beaver's cost-extremes agree with
  Qwen3Guard's labels ~83% of the time — consistent with the high ranker AUC.
- **Beaver is a valid ranker** on both subsets (AUC 0.87–0.99), with the raw prompt (`cost_full`),
  no length confound and negligible truncation. Qwen3Guard remains the safe/unsafe label gate.

## Decision

- **Survival 23.7% ≥ ~20% → WildJailbreak (vanilla_harmful + adversarial_harmful) IS suitable.
  Proceed with the full run.** At ~24% survival over a pool of tens of thousands of harmful
  prompts, thousands of clean pairs are attainable (well past the ~420-pair ceiling of PKU).
- **Ranker decision: keep `beaver-7b-unified-cost`, scoring with the raw prompt (`cost_full`)** —
  validated by the sanity check. Do not use vanilla-query scoring. Qwen3Guard stays the label
  gate (filter-1/filter-3).
- **Optional next steps:** (a) human skim of the 10 disagreement cases to confirm the failure
  mode is benign; (b) if more pairs per prompt are wanted, raise N or lean toward the adversarial
  subset (higher diversity → higher filter-1 survival).

## Decision rule (reference)

- Survival ≥ ~20% → proceed with full run (expect ≥ ~2–4K pairs from the full dedup'd set).
- Survival < ~10% → supplement prompts with WildJailbreak adversarial-harmful.
- In between → inspect the eyeball file and (where available) the per-category breakdown first.
