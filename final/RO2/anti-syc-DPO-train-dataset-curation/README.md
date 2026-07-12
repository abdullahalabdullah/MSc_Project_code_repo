# Anti-Sycophancy DPO — training-dataset curation

Self-contained module implementing the frozen curation spec
(`../../anti_sycophancy_dpo_dataset_curation_spec.md`, Stages A–J): TriviaQA questions →
knowledge filter (Filter-0) → LLM-generated wrong answers → opinionated templates →
contrast sampling (Filter-1) → one (chosen, rejected) pair per surviving prompt →
50/50 direction balance → sham control → QA report — ending in the same training-ready
`DatasetDict({train,valid})` with `prompt`/`chosen`/`rejected` columns that the
harmlessness module exports, so `../train_dpo.py` consumes it **unchanged** (use the
`../train_antisyc.sh` / `../train_antisyc_sham.sh` wrappers).

It also produces the **two frozen eval files** (`reserved/`) that the RO1
`anti-sycophancy-eval` consumes — import them there once with its
`scripts/01_prepare_eval_data.py --source-dir <run>/reserved`.

## Stage map (spec stage → script → what runs it)

| # | Script | Spec stage | Hardware | Output (per run dir) |
|---|---|---|---|---|
| 01 | `01_sample_pool.py` | A — sampling (eval slice FIRST, seed 7) | CPU + network | `raw/training_pool.jsonl`, `reserved/eval_id_triviaqa.partial.jsonl` |
| 02 | `02_build_eval_ood.py` | B — OOD eval file (SycophancyEval "Are you sure?") | CPU + network | `reserved/eval_ood_sycophancyeval.jsonl` + provenance |
| 03 | `03_filter0_generate.py` | C — bare-prompt sampling, N=10 | **GPU (vLLM)** | `filter0/generations.jsonl` |
| 04 | `04_filter0_grade.py` | C — grading + pass band (≥9/10 parseable, ≥70% correct) | **API (gpt-5-mini)** | `filter0/{grades,summary}.jsonl` |
| 05 | `05_wrong_answers.py` | D — wrong answer `w` per survivor + eval slice freeze | **API** | `filter0/wrong_answers.jsonl`, `reserved/eval_id_triviaqa.jsonl` |
| 06 | `06_templates.py` | E — one template per (question, direction), seed 1234 | CPU | `filter1/instances.jsonl` |
| 07 | `07_filter1_generate.py` | F — contrast sampling, N_FILTER1=10 | **GPU (vLLM)** | `filter1/generations.jsonl` |
| 08 | `08_filter1_grade.py` | F — grading, classes, syc_fraction | **API** | `filter1/{grades,summary}.jsonl` |
| 09 | `09_pairs.py` | G+H — pair selection (seed 1701) + 50/50 balance (seed 2003) | CPU | `pairs/anti_syc_dpo.jsonl` |
| 10 | `10_sham.py` | I — sham control (p=0.5 swap, seed 69) | CPU | `pairs/anti_syc_dpo_sham.jsonl` |
| 11 | `11_qa_report.py` | J — QA report, manifest, pilot gates | CPU | `qa_report.md`, `manifest.json`, `pilot_gates.json` |
| 12 | `12_export_dpo_dataset.py` | (training handoff) | CPU | `dpo_dataset/`, `dpo_dataset_sham/` |

Support files: `config.py` (single source of truth — spec constants hardcoded, run knobs
env-overridable), `prompts.py` (spec-verbatim prompt strings; the RO1 eval carries a twin),
`grader.py` + `grading.py` (gpt-5-mini wrappers + shared grading core), `generation.py`
(shared vLLM core), `utils.py` (stage cache, checkpoints, concurrent API driver),
`audit_grader_sample.py` (pilot gate 1 tooling), `backup_run_to_hf.py` (move run dirs
between boxes via a private HF dataset repo).

## Drivers — GPU/API separation

Only stages 03 and 07 need a GPU; every grading/analysis stage runs on any laptop with
`OPENAI_API_KEY`. Two GPU rental windows per run are structurally unavoidable (Filter-1's
prompts depend on API grading of Filter-0).

```bash
# ONE box (GPU + API key) — smoke test or single-rental run:
X=5 RUN_OUTPUT_DIR=outputs/smoke bash run_all.sh

# Split execution:
X=500 bash run_phase1_gpu.sh outputs/pilot            # 01, 02, 03           (GPU)
X=500 bash run_phase2_api.sh outputs/pilot            # 04, 05, 06           (no GPU)
X=500 bash run_phase3_gpu.sh outputs/pilot            # 07                   (GPU)
X=500 PILOT=1 X_FULL=20000 bash run_phase4_api.sh outputs/pilot   # 08, 09, 11 (no GPU)
# full run: same four commands with X=20000, no PILOT (adds 10 + 12)

# Move the run dir between boxes (heavy JSONLs are gitignored):
python backup_run_to_hf.py --push outputs/pilot       # on the box you're leaving
python backup_run_to_hf.py --pull outputs/pilot       # on the box you're joining
```

Env knobs (see `config.py` header): `X` (pool size; 5 smoke / 500 pilot / 20000 full),
`RUN_OUTPUT_DIR`, `N_FILTER1` (owner-approved escalation to 16), `DROPPED_TEMPLATES`
(pilot gate 3), `PILOT`, `X_FULL`, `VALID_FRACTION`, `GEN_SEED`, `FORCE=1`.

## Caching & resume (spec §2)

- **Stage level:** each stage writes `stage_markers/<stage>.done.json` with a hash of the
  config it consumed (chained with its upstream stage's hash). Re-running skips completed
  stages; a config change invalidates and re-runs; `--force` / `FORCE=1` always re-runs.
- **Mid-stage:** GPU stages checkpoint per generate-batch (`*.ckpt.jsonl` +
  fingerprint sidecar); API stages resume from `grade_progress.json` done-keys.
- **Call level:** every classifier call is cached in `grader_cache.jsonl` keyed
  `(question_id, prompt_type, sha1(extracted))`; wrong answers in
  `wrong_answer_cache.jsonl` keyed `question_id`. These survive `--force` on purpose.

## Pilot workflow (blocking gates, spec §10)

1. Run the four phases with `X=500 PILOT=1 X_FULL=20000` → `pilot_gates.json`.
2. Gate 1 (grader audit): `python audit_grader_sample.py --make outputs/pilot`, hand-label
   the 100-row CSV, `--score`, then re-run `11_qa_report.py` — required agreement ≥ 95%.
3. Gate 2 (yield): projected pairs ≥ 3,000. On failure STOP and report (escalation:
   `N_FILTER1=16`, owner decision). Gate 3: templates with wrong-opinion rejected-class
   rate < 2% → set `DROPPED_TEMPLATES=...` for the full run. Gate 4: UNPARSEABLE < 10%.

## Known deviations from the spec (recorded in `manifest.json`)

- **gpt-5-mini temperature**: spec says 0; the model rejects non-default temperature, so
  it is omitted (API default) — same as the existing RO1 monitorability monitor.
- **vLLM sampling seed** (`GEN_SEED`, default 0) is not in the spec's seed table.
- **OOD loader**: SycophancyEval records' assistant-prefill turns (`"The answer is ("`)
  are dropped (incompatible with thinking-mode + Final-answer grading); records without
  an objectively gradable gold are dropped and counted (see `eval_ood_provenance.json`).

## Pair schema (`pairs/*.jsonl`)

Spec §9: `pair_id`, `question_id`, `source`, `direction`, `template_id`, `question`,
`gold`, `gold_aliases`, `asserted_answer`, `prompt`, `chosen`/`rejected`
(`{full_text, final_answer, grade, num_tokens}` — `full_text` is the verbatim completion
incl. `<think>…</think>`, never trimmed), `filter0_consistency`, `syc_fraction`,
`n_chosen_class`, `n_rejected_class`. The sham file adds `sham_swapped`.
