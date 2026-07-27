# PKU-SafeRLHF-30K suitability experiment — summary

**Goal.** Decide whether `PKU-Alignment/PKU-SafeRLHF-30K` is a suitable prompt source for
mining harmlessness preference pairs via Qwen3-4B self-generation.

Source data: `outputs/report.md` (PKU-SafeRLHF-30K run).

## Method (as run)

- **Load & dedup.** `PKU-Alignment/PKU-SafeRLHF-30K` (train), deduplicated on the prompt column.
- **Sample.** 300 unique prompts, fixed seed (0). *(Spec said 350; the run used 300.)*
- **Generate.** Qwen3-4B, thinking mode on, temperature 0.6, top_p 0.95, N=8 responses,
  max_tokens 8192, full outputs stored. *(Spec said N=10; the run used N=8.)* Live truncation
  reporting on.
- **Split CoT/answer.** Strip everything up to and including `</think>`; keep the remainder as
  `answer`.
- **Classify (filter-1).** Qwen3Guard-Gen per (prompt, answer) → safe/unsafe; drop prompts where
  all N answers share one label.
- **Score.** `beaver-7b-unified-cost` per (prompt, answer) → scalar cost; per prompt
  chosen = min cost, rejected = max cost.
- **Validate pair (filter-3).** Keep prompt only if Qwen3Guard says chosen = Safe and
  rejected = Unsafe.
- *Note: "by harm category" was not produced — PKU-SafeRLHF-30K has no harm-category field, so
  survival is reported overall only.*

## Results

- **Unique prompts after dedup:** 6,975.
- **Sampled:** 300 unique prompts (seed 0), 8 responses each.
- **filter-1 (safety-label diversity):** dropped **275 / 300 (91.7%)** → 25 enter scoring.
  (~92% of prompts had all responses on the same label — overwhelmingly all-Safe.)
- **filter-3 (chosen=Safe & rejected=Unsafe):** dropped **7 / 25 (28.0%)** → 18 pairs.
- **End-to-end survival: 18 / 300 = 6.0%** (overall only; no per-category breakdown).
- **Cost-spread (rejected − chosen):** mean 8.49, median 6.89, range 4.58–14.94,
  IQR 5.96–10.62 — healthy separation.
- **Sign-convention sanity check:** mean Beaver cost **Unsafe +2.16** (n=90) vs
  **Safe −2.96** (n=110) → **PASS** (clean ~5-point margin; "higher cost = more harmful"
  confirmed; guard emits a real Safe/Unsafe mix, not None).
- **Eyeball file:** 10 random surviving pairs dumped to `sample_pairs.json`.

**Interpretation.** The pipeline works and the pairs that form are high-quality; the bottleneck
is **filter-1** — a safety-tuned Qwen3-4B refuses ~92% of these prompts on *every* sample, so
there is no safe/unsafe contrast to mine. At 6%, the ~6,975 unique prompts cap out around
~420 pairs — too thin for large-scale training.

## Decision

- **Survival 6.0% < ~10% → PKU-SafeRLHF-30K is NOT suitable on its own** for self-generation
  pair mining. Per the decision rule, **supplement / replace prompts with WildJailbreak
  adversarial-harmful.**
- **Follow-up (confirms the decision):** the WildJailbreak run (`vanilla_harmful` +
  `adversarial_harmful`, 300 prompts, N=10) yielded **71 / 300 = 23.7%** survival (≥20% →
  proceed), ~4× the PKU rate — driven by the adversarial prompts producing far more
  safe/unsafe diversity. See `documentation.md` and the WildJailbreak run's `report.md` /
  `beaver_sanity_report.md`.

## Decision rule (reference)

- Survival ≥ ~20% → proceed with full run (expect ≥ ~2–4K pairs from the full dedup'd set).
- Survival < ~10% → supplement prompts with WildJailbreak adversarial-harmful.
- In between → inspect the eyeball file and (where available) the per-category breakdown first.
