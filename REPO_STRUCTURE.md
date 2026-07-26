# Repository structure — what each folder was used for

This repo accumulated one folder per iteration of the MSc project rather than
rewriting in place, so most top-level directories are **historical snapshots**.
The only directory that holds the current, submitted work is **`final/`**.
Everything else is kept for provenance.

Dates below are first/last commit touching that folder.

---

## Current work

### `final/` — the submitted pipeline *(2026-07-12 → 2026-07-24)*
The two research objectives, cleaned up and consolidated. Everything referenced
in the write-up comes from here.

- **`final/RO1/`** — evaluation side. Three self-contained eval suites plus a
  mechanistic follow-up. See `final/RO1/documentation.md`.
  - `monitorability-eval/` — CoT monitorability under hint pressure; headline
    metric **g-mean²**. `run_inference.sh` (GPU) → `run_monitor_and_results.sh`
    (monitor ensemble + analysis). Contains `mechinterp/` — the weight- and
    activation-space sub-study (`lora_diff.py`, `act_extract.py`,
    `analyse_deltas.py`, `cross_steer.py`, `length_confound.py`) asking whether
    harmless-DPO and anti-syc-DPO raised monitorability along a shared direction.
  - `harmlessness-eval/` — safety on the WildJailbreak eval set, judged by
    WildGuard (Safety Rate + Refusal Rate). `run_harmlessness.sh`.
  - `anti-sycophancy-eval/` — sycophancy metrics M1/M2 (frozen TriviaQA slice)
    and M3 (SycophancyEval "Are you sure?"), graded by gpt-5-mini, with a paired
    base-vs-trained comparison.
  - `config/mvp_config.yaml` — the single shared config for all three suites.
  - `outputs-and-results/` — all generated artefacts, namespaced per model
    (`Qwen_Qwen3-4B`, `er-abd_harmless-dpo-...`, `er-abd_anti-syc-dpo-...`,
    `er-abd_harmless-sft-...`): `inference-outputs/`, `monitor_outputs/`,
    `monitorability-results/`, `harmlessness-results/`,
    `anti-sycophancy-results/`, `plots/`, `extra-plots/` (CoT-length,
    cross-model, task-performance cuts), `mechinterp/`.
  - `notes/` — analysis write-ups that aren't code: length-confound checks,
    paired-bootstrap methodology, n-vs-N eval-set scaling, pilot results,
    cost estimates.
  - `logs/` — timestamped run logs (gitignored in earlier versions).

- **`final/RO2/`** — training side. Two DPO dataset-curation pipelines feeding
  one shared LoRA trainer. See `final/RO2/DOCUMENTATION.md`.
  - `harmlessness-DPO-train-dataset-curation/` — WildJailbreak harmful prompts →
    Qwen3-4B generations → Qwen3Guard labels → Beaver cost model → preference
    pairs (numbered `01_`–`06_` stages, each resumable via `.joblib`).
  - `anti-syc-DPO-train-dataset-curation/` — TriviaQA → sycophancy preference
    pairs (`01_`–`12_` stages), plus the frozen eval files RO1 consumes.
  - `train_dpo.py` / `train_dpo_norm.py` — LoRA DPO and length-normalized DPO
    (open-instruct `dpo_norm`, average log-probs; added after the response-length
    inflation problem was found).
  - `train_sft.py` + `convert_dpo_to_sft.py` + `verify_sft_tokenization.py` — the
    SFT ablation arm.
  - `train.sh` (harmlessness defaults, `METHOD=dpo|dpo_norm`), `train_antisyc.sh`
    (anti-sycophancy wrapper), `train_sft.sh`.
  - `checkpoints/qwen3-4B/` — `harmless_dpo`, `anti_syc_dpo`, `smoke_sft`.
  - `PILOT_TRAINING_ANALYSIS.md`, `SFT_Arm_Developer_Spec.md`,
    `SMOKE_RUN_LOG-template.md` — specs and run records.

- `anti_sycophancy_dpo_dataset_curation_spec.md` / `.pdf` — the curation spec
  that `final/RO2/anti-syc-DPO-train-dataset-curation/` implements.

---

## RO1 lineage (evaluation) — superseded

### `MVP-RO1/` *(2026-06-09 → 2026-07-08)*
First working RO1 pipeline: **one task (MMLU), one hint type (Stanford
professor), four Tulu-3-8B checkpoints**. Metrics were TE / ack / HAR / |S|
before g-mean² became the headline. Contains one `run_*.sh` per model
(`run_qwen.sh`, `run_mistral.sh`, `run_deepseek.sh`, `run_llama32_3b.sh`, …) and
their logs — this per-model-script sprawl is what v2 replaced with positional
drivers. Spec: `MVP_RO1.md`.

### `MVP-RO1-v2/` *(2026-06-30 → 2026-07-08)*
Rewrite of the above: **6 datasets**, raw HF model IDs (no model registry),
two positional drivers (`run_inference.sh`, `run_monitor_and_results.sh`), and
the `monitorability-eval/` + `harmlessness-eval/` split. This is the direct
ancestor of `final/RO1/`, which adds `anti-sycophancy-eval/` and `mechinterp/`.

### `MVP-RO1-v2-temp/` *(2026-07-09)*
A one-day copy of `MVP-RO1-v2` used to run the **GPT-5 vs GPT-5-mini monitor
comparison** without disturbing the main tree. Diff vs `MVP-RO1-v2` is just two
extra scripts (`run_gpt5_monitor.sh`, `run_results_analysis.sh`) and the
side-by-side result dirs (`..._trimmed`, `...-with-monitor-gpt5`,
`...-with-monitor-gpt5mini`, `monitor_comparison_gpt5_vs_gpt5mini.md`).
Disposable.

---

## RO2 lineage (training) — superseded

### `MVP-RO2-old/` *(2026-06-17)*
Earliest RO2 attempt — a bare `train_dpo.py` + `train_harmlessness.sh` +
sanity scripts, no dataset-curation module. Spec: `MVP_RO2.md`.

### `MVP-RO2/` *(2026-06-09 → 2026-06-22)*
The **SFT MVP**: LoRA SFT of `Qwen/Qwen3-4B` on the safe (chosen) responses of
PKU-SafeRLHF conflict cases, one adapter per epoch pushed to the Hub for the
RO1 eval pipeline. This is the arm where **answer-only SFT collapsed Qwen3's CoT
to 0%**. Spec: `sft_mvp_ro2_experiment_design.md`.

### `MVP-RO2-llama/` *(2026-06-24)*
Port of `MVP-RO2` to `meta-llama/Llama-3.2-3B-Instruct` on hh-rlhf
`harmless-base` single-turn, 4 epochs → 4 checkpoints. All `<think>`-block
machinery removed (Llama-3.2 has no thinking mode), so the CoT-monitorability
objective is dropped here. Kept the old Qwen/PKU spec file for history.

### `Reward-Hacking-in-Explanations/` *(2026-06-27 → 2026-07-04)*
External codebase (Ferreira et al., *Truthful or Fabricated*) brought in and run
locally as a reference implementation for DPO + reward-model evaluation. Note it
contains a checked-in virtualenv (`my_env/`).

### `MVP-RO2-v1/` *(2026-07-02)*
A byte-identical copy of `Reward-Hacking-in-Explanations` (only
`.claude/settings.local.json` and some venv binaries differ) — the fork point
taken before adapting it. Effectively redundant with the folder above.

### `MVP-RO2-v2/` *(2026-07-05 → 2026-07-09)*
The harmlessness DPO pipeline proper: `DPO-train-dataset-curation/` (WildJailbreak
→ preference pairs) + `train_dpo.py` + `train.sh`. This is the direct ancestor of
`final/RO2/`, which adds the anti-sycophancy curation module, the length-normalized
trainer, and the SFT arm. `PILOT_TRAINING_ANALYSIS.md` records the pilot that
validated the setup.

---

## Supporting studies and reference material

### `Experiments/` *(2026-07-04 → 2026-07-09)*
Standalone validation experiments that fed decisions in the main pipelines.
- `DPO-data-&-RM-suitability-check/` — the 5-stage prototype of the harmlessness
  curation pipeline (WildJailbreak → Qwen3-4B N=10 → Qwen3Guard → Beaver cost
  model → pair report), used to check the reward model was fit for purpose and to
  measure pair-survival rate before committing to a full run. Its structure was
  lifted into `MVP-RO2-v2` / `final/RO2`.
- `monitor-validation/` — does **gpt-5-mini** monitor as well as gpt-4o?
  `compare_monitors.py` + two monitor-output sets + `monitor_comparison.md`.
- `Human-Validation/` — hand-check of the gpt-5-mini monitor: 30 hinted
  (15 followed the hint, 15 resisted) + 15 unhinted Qwen3-4B samples across
  4 datasets, drawn at random from a larger run.
- `monitor_comparison_gpt5_vs_gpt5mini.md` — the GPT-5 vs GPT-5-mini result from
  the `MVP-RO1-v2-temp` run.

### `papers-and-notes/` *(2026-06-30)*
Reading notes and metric definitions: `Tulu3.md`, `qwen3.md`, and
`metrics/` (`gmean2_evaluation_metric.md`, `g-mean² Filtering Levels.md`) — the
definition and filtering rules behind the headline metric.

### Root-level files
- `Impact_of_Post-Training_on_CoT_Monitorability.md` — the project proposal /
  research plan the whole repo implements.
- `hh_rlhf_stats.py` — one-off dataset statistics for hh-rlhf, written while
  scoping the Llama port.
- `README.md` — stub.

---

## Quick "which folder do I open?"

| I want to… | Go to |
|---|---|
| Run any evaluation | `final/RO1/` (`documentation.md`) |
| Train a model or curate a preference dataset | `final/RO2/` (`DOCUMENTATION.md`) |
| Find a result, plot, or table used in the write-up | `final/RO1/outputs-and-results/` |
| Understand the g-mean² metric | `papers-and-notes/metrics/` |
| Check how the monitor was validated | `Experiments/monitor-validation/`, `Experiments/Human-Validation/` |
| See an earlier version of an eval | `MVP-RO1-v2/` → `MVP-RO1/` |
| See an earlier version of training | `MVP-RO2-v2/` → `MVP-RO2/` → `MVP-RO2-old/` |
