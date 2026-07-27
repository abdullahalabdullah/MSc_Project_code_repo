# Repository structure — what each folder was used for

This repo accumulated one folder per iteration of the MSc project rather than
rewriting in place, so most top-level directories are **historical snapshots**.
The only directory that holds the current, submitted work is **`Full-Codebase/`**.
Everything else is kept for provenance.

Dates below are first/last commit touching that folder.

---

## Development timeline — how the project evolved folder by folder

The story the folders tell, in order. Each phase notes **what was built, what
was learned, and the decision that led to the next phase** — this is the
skeleton of the methodology section.

### Phase 1 — parallel MVPs *(Jun 9 – Jun 22)*
Evaluation and training started side by side.
- **Eval:** `RO1-old-attempts/MVP-RO1/` — first monitorability pipeline: one
  task (MMLU), one hint type (Stanford professor), four Tulu-3-8B checkpoints,
  metrics TE / ack / HAR / |S|. Proved the hint-based eval concept worked, but
  one-script-per-model didn't scale and the metrics were not yet g-mean².
- **Training:** `RO2-old-attempts/MVP-RO2-v1/RO2-SFT-Qwen3-4B/` — harmlessness
  **SFT** of Qwen3-4B on PKU-SafeRLHF safe responses.
- **Finding that changed the project:** answer-only SFT **collapsed Qwen3's
  CoT to 0%** — the trained checkpoints could not be used for a
  CoT-monitorability study at all.

### Phase 2 — failed workaround, then pivot decision *(Jun 24)*
`RO2-old-attempts/MVP-RO2-v1/RO2-SFT-llama/` — the same SFT pipeline ported to
Llama-3.2-3B on hh-rlhf. But Llama-3.2 has no thinking mode, so the port had
to drop the CoT objective entirely — a dead end for the research question.
**Decision: abandon SFT as the main training method; move to DPO**, which
optimizes preferences without retraining the response format (and later proved
not to suppress the CoT).

### Phase 3 — learning DPO from a reference implementation *(Jun 27 – Jul 4)*
`RO2-old-attempts/MVP-RO2-v2/Reward-Hacking-in-Explanations/` — the Ferreira
et al. (*Truthful or Fabricated*) codebase, run locally as a working reference
for DPO training + reward-model evaluation. No thesis code descends from it
directly, but the DPO pipeline design in Phase 5 was informed by it. (An
earlier one-day DPO sketch, `MVP-RO2-old`, predates this and led nowhere —
deleted, see below.)

### Phase 4 — eval rewrite + monitor validation *(Jun 30 – Jul 9)*
- `RO1-old-attempts/MVP-RO1-v2/` — full rewrite of the eval: **6 datasets**,
  raw HF model IDs, two positional drivers, the `monitorability-eval/` /
  `harmlessness-eval/` split, and **g-mean² becomes the headline metric**.
  Direct ancestor of `Full-Codebase/RO1/`.
- Monitor choice: the GPT-5 vs GPT-5-mini comparison
  (`Full-Codebase/RO1/notes/monitor_comparison_gpt5_vs_gpt5mini.md`) supported using the
  cheaper **gpt-5-mini as the API monitor**. (Two other monitor studies were
  deleted 2026-07-26 as not write-up-relevant: a planned-but-never-performed
  human hand-check, and an inter-monitor agreement study from the
  qwen3-1.7B/old-dataset era — both recoverable from git history. Limitation
  to state in the write-up: no human ground-truth check of the monitor was
  done.)

### Phase 5 — DPO pipeline: prototype → pilot *(Jul 4 – Jul 9)*
- `Full-Codebase/RO2/DPO-data-&-RM-suitability-check/` — 5-stage prototype of the
  harmlessness curation pipeline (WildJailbreak → Qwen3-4B samples →
  Qwen3Guard labels → Beaver cost model → pairs), used to verify the reward
  model was fit for purpose and measure pair-survival rate **before** paying
  for a full run.
- `RO2-old-attempts/MVP-RO2-v3/` — the harmlessness DPO pipeline proper,
  scaling the prototype up (`DPO-train-dataset-curation/` + `train_dpo.py`).
  `PILOT_TRAINING_ANALYSIS.md` records the pilot that validated the setup
  (clean training signal, no CoT collapse). **Decision: this becomes the
  methodology; carried into `Full-Codebase/RO2/` (`train_dpo.py` is
  byte-identical).**

### Phase 6 — consolidation and the full study *(Jul 12 – Jul 26)*
Everything moved into `Full-Codebase/` and the actual experiments ran:
- **Full harmlessness DPO run** (checkpoint `…step-735`) + the three RO1 eval
  suites over base vs trained models.
- **Problem found & fixed:** DPO inflated response lengths (unnormalized
  sum-of-log-probs objective) — investigated in
  `Full-Codebase/RO1/notes/harmless_dpo_response_length_inflation.md` and
  `…RO2/harmlessness-DPO-train-dataset-curation/dpo_length_inflation_summary.md`,
  fixed by adding the length-normalized trainer `train_dpo_norm.py`; the
  length-confound analyses (`notes/length-confound/`) rule length out as the
  driver of the monitorability result.
- **Second training objective added:** the anti-sycophancy arm — TriviaQA
  preference-pair curation, DPO run (`…step-520`), and its own eval suite —
  to test whether the monitorability effect is objective-specific or general.
- **SFT ablation arm added** (`train_sft.py`, checkpoint `…step-735`) to
  separate "DPO the method" from "preference data the signal".
- **Mechanistic follow-up** — a sub-study asking *why* DPO raised monitorability
  (weight- vs activation-space overlap between the two DPO arms). Its code and
  results were **deleted 2026-07-27**; see *Removed reference material* below.

**One-line arc for the write-up:** SFT destroyed the very signal being studied
→ pivot to DPO via a paper replication → prototype and pilot de-risked the
curation pipeline → full runs on two independent DPO objectives + an SFT
ablation, with the monitor and eval scale each validated by dedicated
side-experiments.

---

## Current work

### `Full-Codebase/` — the submitted pipeline *(2026-07-12 → 2026-07-26)*
The two research objectives, cleaned up and consolidated. Everything referenced
in the write-up comes from here.

#### `Full-Codebase/RO1/` — evaluation side
Three self-contained eval suites, all driven by
the single shared config `Full-Codebase/RO1/config/mvp_config.yaml`. Entry point for
docs: `Full-Codebase/RO1/documentation.md`.

- **`monitorability-eval/`** — CoT monitorability under hint pressure; headline
  metric **g-mean²**. `run_inference.sh` (GPU) → `run_monitor_and_results.sh`
  (monitor ensemble + analysis).
  - `data/` — the eval datasets: `codemmlu`, `gpqa_diamond`, `lexam`,
    `lsat_ar`, `lsat_lr`, `medxpertqa`, `mmlu_pro`, `wmdp`.
  - `scripts/` — **50 numbered analysis scripts** (`01_…`–`49_…`). The early
    ones are the core pipeline (`04_compute_gmean2.py`, `05_analyze_gmean2.py`,
    `14_pooled_gmean2.py`); everything after ~`10_` generates the plots and
    tables in `outputs-and-results/` — length-standardized/bucketed g-mean²,
    dumbbell/forest plots, accuracy-slice scatters, length-quantile cuts by
    hint/dataset. The number prefix ≈ the order they were written; filenames
    say what each computes.
- **`harmlessness-eval/`** — safety on the WildJailbreak eval set, judged by
  WildGuard (Safety Rate + Refusal Rate). `run_harmlessness.sh`, plus
  `resume_from_wildguard.sh` to restart a run at the judging stage.
- **`anti-sycophancy-eval/`** — sycophancy metrics M1/M2 (frozen TriviaQA
  slice) and M3 (SycophancyEval "Are you sure?"), graded by gpt-5-mini.
  Numbered stages `01_prepare_eval_data.py` → `06_generate_reports.py`
  (inference → grading → metrics → paired base-vs-trained comparison →
  reports), with the logic in `src/`.
- **`outputs-and-results/`** — all generated artefacts. Result folders are
  namespaced per model: `Qwen_Qwen3-4B` (base),
  `er-abd_harmless-dpo-qwen3-4B-step-735`,
  `er-abd_anti-syc-dpo-qwen3-4B-step-520`,
  `er-abd_harmless-sft-qwen3-4B-step-735` (and `…harmless-dpo…step-147` in
  the monitorability results only).
  - `inference-outputs/`, `monitor_outputs/` — raw generations and monitor
    verdicts.
  - `monitorability-results/` — per-model g-mean² results, plus
    `length_*_tables.md` / `length_*_model_comparison.md` — the CoT-length
    tables at dataset, dataset×hint, and hint-type level.
  - `harmlessness-results/`, `anti-sycophancy-results/` — per-model results
    for the other two suites.
  - `plots/` — per-model main plots.
  - `extra-plots/` — the cross-cutting figure sets:
    `cot-length-vs-monitorability/`, `cross-model-comparison/`,
    `task-performance-vs-monitorability/`.
- **`notes/`** — analysis write-ups that aren't code. Highlights:
  - `paired_bootstrap_pooled_gmean2.md` — the paired-bootstrap methodology
    behind the headline comparisons; `std_reporting_mean_sd_vs_ci.md`.
  - `eval_set_scale_n_vs_N.md` — n=100/N=8 eval-set scaling rationale;
    `monitor_cost_estimate.md`.
  - `harmless_dpo_response_length_inflation.md` + the step-147 / dpo_norm
    step-438 length checks — the response-length-inflation investigation that
    motivated `train_dpo_norm.py`.
  - `length-confound/` — bucketed, cumulative, standardized and within-slice
    length analyses for base vs harmless-DPO (step 735) and base vs
    anti-syc-DPO (step 520).
  - `anti_sycophancy_pilot_results.md`, `…_pipeline_design_notes.md`,
    `…_qa_transcript.md`, `antisyc_dpo_monitorability_vs_base.md` — the
    anti-sycophancy arm's design and results notes.
- `logs/` — timestamped run logs.

#### `Full-Codebase/RO2/` — training side
Two DPO dataset-curation pipelines feeding one shared LoRA trainer. Entry
point for docs: `Full-Codebase/RO2/DOCUMENTATION.md`.

- **`harmlessness-DPO-train-dataset-curation/`** — WildJailbreak harmful
  prompts → Qwen3-4B generations → Qwen3Guard labels → Beaver cost model →
  preference pairs. Numbered stages `01_prepare_prompts.py` →
  `06_export_dpo_dataset.py`, each resumable via `.joblib`; `run_all.sh`,
  `run_from_03.sh`, `run_pilot_and_sanity.sh`, `merge_runs.py` (merges
  multiple sampling runs to hit the pair target). Write-up material lives
  alongside the code: `documentation.md`, `labeler_comparison.md`,
  `NOTES_length-confound_and_benign-axis.md`,
  `dpo_length_inflation_summary.md`.
- **`anti-syc-DPO-train-dataset-curation/`** — TriviaQA → sycophancy
  preference pairs, plus the frozen eval files RO1 consumes. Numbered stages
  `01_sample_pool.py` → `12_export_dpo_dataset.py` (no `10_`), run in four
  phases: `run_phase1_gpu.sh` … `run_phase4_api.sh` (or `run_all.sh`).
  Extras: `audit_grader_sample.py`, `fix_yesno_grades.py`,
  `backup_run_to_hf.py`. The curation spec this implements is here too:
  `anti_sycophancy_dpo_dataset_curation_spec.md` (moved in from `Full-Codebase/`).
- **Trainers** (shared, repo-root of RO2):
  - `train_dpo.py` / `train_dpo_norm.py` — LoRA DPO and length-normalized DPO
    (open-instruct `dpo_norm`, average log-probs; added after the
    response-length inflation problem was found).
  - `train_sft.py` + `convert_dpo_to_sft.py` + `verify_sft_tokenization.py` —
    the SFT ablation arm.
  - `train.sh` (harmlessness defaults, `METHOD=dpo|dpo_norm|sft`),
    `train_antisyc.sh` (anti-sycophancy wrapper), `train_sft.sh`.
- `checkpoints/qwen3-4B/` — `harmless_dpo`, `anti_syc_dpo`, `smoke_sft`
  (full checkpoints are on the HF Hub under `er-abd/…`; these are local
  copies/smoke artefacts).
- `PILOT_TRAINING_ANALYSIS.md`, `SFT_Arm_Developer_Spec.md`,
  `SMOKE_RUN_LOG-template.md` — specs and run records.

---

## RO1 lineage (evaluation) — superseded

Both superseded RO1 folders live under **`RO1-old-attempts/`** (moved there
2026-07-26).

### `RO1-old-attempts/MVP-RO1/` *(2026-06-09 → 2026-07-08)*
First working RO1 pipeline: **one task (MMLU), one hint type (Stanford
professor), four Tulu-3-8B checkpoints**. Metrics were TE / ack / HAR / |S|
before g-mean² became the headline. Contains one `run_*.sh` per model
(`run_qwen.sh`, `run_mistral.sh`, `run_deepseek.sh`, `run_llama32_3b.sh`, …) and
their logs — this per-model-script sprawl is what v2 replaced with positional
drivers. Spec: `MVP_RO1.md`.

### `RO1-old-attempts/MVP-RO1-v2/` *(2026-06-30 → 2026-07-08)*
Rewrite of the above: **6 datasets**, raw HF model IDs (no model registry),
two positional drivers (`run_inference.sh`, `run_monitor_and_results.sh`), and
the `monitorability-eval/` + `harmlessness-eval/` split. This is the direct
ancestor of `Full-Codebase/RO1/`, which adds `anti-sycophancy-eval/`.

### `MVP-RO1-v2-temp/` — **deleted 2026-07-26**
Was a one-day (2026-07-09) copy of `MVP-RO1-v2` used to run the **GPT-5 vs
GPT-5-mini monitor comparison** without disturbing the main tree. Diff vs
`MVP-RO1-v2` was just two extra scripts (`run_gpt5_monitor.sh`,
`run_results_analysis.sh`) and the side-by-side result dirs (`..._trimmed`,
`...-with-monitor-gpt5`, `...-with-monitor-gpt5mini`). The conclusion survives
as `Full-Codebase/RO1/notes/monitor_comparison_gpt5_vs_gpt5mini.md`; everything else is
recoverable from git history (`git show 804711c:MVP-RO1-v2-temp/...`).

---

## RO2 lineage (training) — superseded

All superseded RO2 folders live under **`RO2-old-attempts/`**, renumbered
`MVP-RO2-v1` → `v2` → `v3` in chronological order (restructured 2026-07-26).
v1 is the abandoned SFT strand, v2 the external reference codebase, v3 the
DPO pipeline that became `Full-Codebase/RO2/`.

> **Naming caution for git archaeology:** in history *before* 2026-07-26,
> `MVP-RO2-v1` was a (deleted, redundant) snapshot of the paper-replication
> code and `MVP-RO2-v2` was what is now `v3`. The pre-restructure names were:
> `MVP-RO2` → `v1/RO2-SFT-Qwen3-4B`, `MVP-RO2-llama` → `v1/RO2-SFT-llama`,
> `Reward-Hacking-in-Explanations` → `v2/…`, `MVP-RO2-v2` → `v3`.

### `RO2-old-attempts/MVP-RO2-v1/` — the SFT strand *(2026-06-09 → 2026-06-24)*
The original RO2 experiment: harmlessness **SFT**. Dead-ended on the CoT
collapse, which forced the pivot to DPO.

- **`RO2-SFT-Qwen3-4B/`** *(2026-06-09 → 2026-06-22, was `MVP-RO2`)* — LoRA SFT
  of `Qwen/Qwen3-4B` on the safe (chosen) responses of PKU-SafeRLHF conflict
  cases, one adapter per epoch pushed to the Hub for the RO1 eval pipeline.
  This is the arm where **answer-only SFT collapsed Qwen3's CoT to 0%**.
  Spec: `sft_mvp_ro2_experiment_design.md`.
- **`RO2-SFT-llama/`** *(2026-06-24, was `MVP-RO2-llama`)* — port of the above
  to `meta-llama/Llama-3.2-3B-Instruct` on hh-rlhf `harmless-base` single-turn,
  4 epochs → 4 checkpoints (plain PEFT/TRL, no Unsloth). All `<think>`-block
  machinery removed (Llama-3.2 has no thinking mode), so the CoT-monitorability
  objective is dropped here — which is why this strand ends.

### `RO2-old-attempts/MVP-RO2-v2/` — paper replication *(2026-06-27 → 2026-07-04)*
`Reward-Hacking-in-Explanations/` — external codebase (Ferreira et al.,
*Truthful or Fabricated*) brought in and run locally as a reference
implementation for DPO + reward-model evaluation. Its checked-in virtualenv
(`my_env/`) was deleted from disk 2026-07-26; recreate from its
`requirements.txt` if you ever need to run it again.

### `RO2-old-attempts/MVP-RO2-v3/` — harmlessness DPO pipeline *(2026-07-05 → 2026-07-09, was `MVP-RO2-v2`)*
The harmlessness DPO pipeline proper: `DPO-train-dataset-curation/` (WildJailbreak
→ preference pairs) + `train_dpo.py` + `train.sh`. This is the direct ancestor of
`Full-Codebase/RO2/` (`train_dpo.py` is byte-identical between the two), which adds
the anti-sycophancy curation module, the length-normalized trainer, and the SFT
arm. `PILOT_TRAINING_ANALYSIS.md` records the pilot that validated the setup.

### Deleted RO2 folders *(both 2026-07-26, recoverable from git history)*
- `MVP-RO2-old/` — a one-day (2026-06-17) first DPO sketch: a bare 680-line
  `train_dpo.py` + sanity scripts, no dataset curation. Nothing downstream
  built on it (`git show 804711c:MVP-RO2-old/...`).
- `MVP-RO2-v1/` *(old naming)* — a snapshot copy of
  `Reward-Hacking-in-Explanations` taken 2026-07-02 as the fork point; a strict
  subset with zero differing files (`git show 804711c:MVP-RO2-v1/...`).

---

## Supporting studies and reference material

### `Experiments/` — **dissolved 2026-07-26**
Held standalone validation experiments; its two useful items were moved to
where they're relevant and the folder deleted:
- `DPO-data-&-RM-suitability-check/` → **`Full-Codebase/RO2/`** — the 5-stage
  prototype of the harmlessness curation pipeline (WildJailbreak → Qwen3-4B
  N=10 → Qwen3Guard → Beaver cost model → pair report). Its `outputs/` hold
  the never-repeated decision studies: PKU vs WildJailbreak dataset
  suitability, WildGuard vs Qwen3Guard labeler comparison, and the
  pair-survival measurement.
- `monitor_comparison_gpt5_vs_gpt5mini.md` → **`Full-Codebase/RO1/notes/`** —
  the GPT-5 vs GPT-5-mini comparison; the basis for choosing gpt-5-mini as
  the API monitor.

Two other studies were deleted outright (recoverable from git history under
`Experiments/`):
- `monitor-validation/` — despite the name, not validation but a three-way
  *agreement* study (gpt-5-mini vs gemma-4-E4B vs qwen3-8B, 2,164 records)
  from the qwen3-1.7B/old-dataset era — judged not relevant to the final
  write-up.
- `Human-Validation/` — a *planned* human hand-check of the gpt-5-mini
  monitor. Only raw qualitative exports were ever generated (400-row xlsx per
  dataset, no human judgments recorded), so no human-validation claim can be
  made in the write-up.

### Removed sub-study — mechanistic follow-up *(deleted 2026-07-27)*
`Full-Codebase/RO1/monitorability-eval/mechinterp/` (code: `lora_diff.py`,
`act_extract.py`, `analyse_deltas.py`, `cross_steer.py`, `length_confound.py`,
`README.md`) and its results dir `Full-Codebase/RO1/outputs-and-results/mechinterp/`
(`lora_diff_results.json`, `length_confound.md`) were deleted from the working
tree. It compared the harmless-DPO and anti-syc-DPO LoRA adapters in weight and
activation space, asking whether both raised monitorability along a shared
direction; the causal steering arm (`cross_steer.py`) was never finished. Nothing
in the eval pipeline imported it — it was read-only w.r.t. the three suites — so
removal breaks nothing. Recover with
`git show 804711c:final/RO1/monitorability-eval/mechinterp/README.md` (the
pre-rename path is `final/`, not `Full-Codebase/`). The activation vectors
(`act_full/delta_vectors.npz`) were never committed and would need a re-run.

### Removed reference material
`papers-and-notes/` (reading notes on Tulu3/Qwen3 and the g-mean² metric
definition files) and the root-level project proposal
(`Impact_of_Post-Training_on_CoT_Monitorability.md`) were **deleted from the
working tree on 2026-07-26** — recover them from git history if needed
(`git show 804711c:papers-and-notes/metrics/gmean2_evaluation_metric.md`, etc.).
The g-mean² *implementation* remains in
`Full-Codebase/RO1/monitorability-eval/scripts/04_compute_gmean2.py`.

### Root-level files
- `README.md` — stub.
- `REPO_STRUCTURE.md` — this file. Per-folder histories for the superseded
  trees live in `RO1-old-attempts/documentation.md` and
  `RO2-old-attempts/documentation.md`.
- `hh_rlhf_stats.py` — **deleted 2026-07-27**. One-off script counting
  single- vs multi-turn examples in hh-rlhf `harmless-base`/`helpful-base`,
  written while scoping the Llama SFT port; it is the source of the 12,254
  single-turn figure still cited in
  `RO2-old-attempts/MVP-RO2-v1/RO2-SFT-llama/` (`sft/data.py`,
  `configs/sft_harmlessness.yaml`, `documentation.md`). Nothing in
  `Full-Codebase/` uses hh-rlhf. Recover with
  `git show d3dd459:hh_rlhf_stats.py`.

---

## Quick "which folder do I open?"

| I want to… | Go to |
|---|---|
| Write the methodology section | the **Development timeline** at the top of this file, then the per-phase evidence files it points to |
| Run any evaluation | `Full-Codebase/RO1/` (`documentation.md`) |
| Train a model or curate a preference dataset | `Full-Codebase/RO2/` (`DOCUMENTATION.md`) |
| Find a result, plot, or table used in the write-up | `Full-Codebase/RO1/outputs-and-results/` |
| Find the methodology notes (bootstrap, length confound, scaling) | `Full-Codebase/RO1/notes/` |
| See how g-mean² is computed | `Full-Codebase/RO1/monitorability-eval/scripts/04_compute_gmean2.py` |
| See the dataset/labeler suitability studies | `Full-Codebase/RO2/DPO-data-&-RM-suitability-check/outputs/` |
| Read the curation specs | `Full-Codebase/RO2/anti-syc-DPO-train-dataset-curation/anti_sycophancy_dpo_dataset_curation_spec.md`, `Full-Codebase/RO2/harmlessness-DPO-train-dataset-curation/documentation.md` |
| Check how the monitor was chosen | `Full-Codebase/RO1/notes/monitor_comparison_gpt5_vs_gpt5mini.md` |
| See an earlier version of an eval | `RO1-old-attempts/MVP-RO1-v2/` → `RO1-old-attempts/MVP-RO1/` |
| See an earlier version of training (DPO) | `RO2-old-attempts/MVP-RO2-v3/` |
| See the abandoned SFT training arm | `RO2-old-attempts/MVP-RO2-v1/` |
| See the paper-replication reference code | `RO2-old-attempts/MVP-RO2-v2/` |
