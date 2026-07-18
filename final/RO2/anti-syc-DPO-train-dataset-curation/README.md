# Anti-Sycophancy DPO — training-dataset curation

Self-contained module implementing the frozen curation spec
(`../../anti_sycophancy_dpo_dataset_curation_spec.md`; the spec's Stage I sham control
was dropped by owner decision — not implemented): TriviaQA questions →
knowledge filter (Filter-0) → LLM-generated wrong answers → opinionated templates →
contrast sampling (Filter-1) → one (chosen, rejected) pair per surviving prompt →
50/50 direction balance → QA report — ending in the same training-ready
`DatasetDict({train,valid})` with `prompt`/`chosen`/`rejected` columns that the
harmlessness module exports, so `../train_dpo.py` consumes it **unchanged** (use the
`../train_antisyc.sh` wrapper).

It also produces the **two frozen eval files** (`reserved/`) that the RO1
`anti-sycophancy-eval` consumes — import them there once with its
`scripts/01_prepare_eval_data.py --source-dir <run>/reserved`.

## Stage map (spec stage → script → what runs it)

| # | Script | Spec stage | Hardware | Output (per run dir) |
|---|---|---|---|---|
| 01 | `01_sample_pool.py` | A — sampling (eval slice FIRST, seed 7; dedups TriviaQA's per-document rows to unique questions) | CPU + network | `raw/training_pool.jsonl`, `reserved/eval_id_triviaqa.partial.jsonl` |
| 02 | `02_build_eval_ood.py` | B — OOD eval file (SycophancyEval "Are you sure?") | CPU + network | `reserved/eval_ood_sycophancyeval.jsonl` + provenance |
| 03 | `03_filter0_generate.py` | C — bare-prompt sampling, N=10 | **GPU (vLLM)** | `filter0/generations.jsonl` |
| 04 | `04_filter0_grade.py` | C — grading + pass band (≥9/10 parseable, ≥70% correct) | **API (gpt-5-mini)** | `filter0/{grades,summary}.jsonl` |
| 05 | `05_wrong_answers.py` | D — wrong answer `w` per survivor + eval slice freeze | **API** | `filter0/wrong_answers.jsonl`, `reserved/eval_id_triviaqa.jsonl` |
| 06 | `06_templates.py` | E — one template per (question, direction), seed 1234 | CPU | `filter1/instances.jsonl` |
| 07 | `07_filter1_generate.py` | F — contrast sampling, N_FILTER1=10 | **GPU (vLLM)** | `filter1/generations.jsonl` |
| 08 | `08_filter1_grade.py` | F — grading, classes, syc_fraction | **API** | `filter1/{grades,summary}.jsonl` |
| 09 | `09_pairs.py` | G+H — pair selection (seed 1701) + direction balance (seed 2003; 50/50 default, `DIRECTION_BALANCE=off|N:1` relaxes) | CPU | `pairs/anti_syc_dpo.jsonl` |
| 11 | `11_qa_report.py` | J — QA report, manifest, pilot gates | CPU | `qa_report.md`, `manifest.json`, `pilot_gates.json` |
| 12 | `12_export_dpo_dataset.py` | (training handoff) | CPU | `dpo_dataset/` |

(There is no stage 10: the spec's Stage I sham control was dropped by owner decision;
the script numbering keeps the gap rather than renumbering.)

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
(pilot gate 3), `PILOT`, `X_FULL`, `VALID_FRACTION`, `GEN_SEED`, `FORCE=1`,
`DIRECTION_BALANCE` (stage 09: `50_50` default; `off` keeps all selected pairs at their
natural wrong/right ratio; `N:1` (e.g. `6:1`) keeps all minority pairs and caps the
majority at N x minority — the owner-approved deviation for the corrected full run).

Post-hoc grade correction: `fix_yesno_grades.py` deterministically relabels Filter-1
grades whose final answer is a bare "Yes/No/True/False" (the confirm-template ambiguity
the gpt-5-mini grader mishandles) or a literal `<your answer>` placeholder echo, then
rebuilds `filter1/summary.jsonl`. Run it after stage 08 and re-run 09/11/12 with
`--force`; do NOT re-run stage 08 afterwards (the grader cache holds the old labels).

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

## Split-machine pilot runbook (GPU pod ↔ local, git transport)

Concrete end-to-end runbook for running the pilot with the **GPU phases on rented pods
and the API phases on a local machine**, transporting the run dir via git. Because the
phases alternate GPU → API → GPU → API → GPU, this uses **three GPU pod sessions**
(phase 1, phase 3, training), each deletable after its `git push`. A fresh pod just
`git clone`s the repo and picks up the committed run dir.

**Two hard rules:**
- **`git add -f`** is mandatory — the heavy files (`filter0/generations.jsonl`,
  `filter1/generations.jsonl`, grades, `dpo_dataset/`) are gitignored; a plain `git add`
  silently drops them and the next machine's stage fails with a missing-file error.
- **Always `git push` (and confirm it landed) BEFORE deleting a pod.** Deleting a pod
  wipes its disk; an un-pushed phase output is lost and must be regenerated.
- git works at pilot scale (files < 100 MB); the **full run's generation files are
  1–2 GB → git will reject them**, so use `backup_run_to_hf.py --push/--pull` or `scp`
  for the full run instead.

Recurring per-machine setup: on every **pod** run `unset HF_TOKEN` (so the real write
token from `.env` is used, not a stale shell value); on **local** run
`export HF_MODELS_CACHE_DIR="$HOME/.cache/huggingface"` (the committed `.env` points HF's
cache at the pod's `/root` path, unwritable on a Mac) and ensure `OPENAI_API_KEY` is set.
`git pull --rebase` before every `git push` (both machines commit to `main`).

```bash
# ── POD #1 (GPU) — phase 1 ──────────────────────────────────────────────────
cd ~/MSc_Project_code_repo/final && git pull        # or git clone <url> if brand-new
conda activate qwen_env                              # env with vllm etc.
unset HF_TOKEN
cd RO2/anti-syc-DPO-train-dataset-curation
X=500 bash run_phase1_gpu.sh outputs/pilot           # [GPU] stages 01,02,03
cd ~/MSc_Project_code_repo/final
git add -f RO2/anti-syc-DPO-train-dataset-curation/outputs/pilot
git commit -m "pilot: after phase1" && git pull --rebase && git push
#   → confirm push landed, then delete the pod

# ── LOCAL — phase 2 ─────────────────────────────────────────────────────────
cd ~/Desktop/MSc_Project/MSc_Project_code_repo/final && git pull
export HF_MODELS_CACHE_DIR="$HOME/.cache/huggingface"
export OPENAI_API_KEY=sk-...                          # or rely on local .env
cd RO2/anti-syc-DPO-train-dataset-curation
X=500 bash run_phase2_api.sh outputs/pilot           # [no GPU] stages 04,05,06
cd ~/Desktop/MSc_Project/MSc_Project_code_repo/final
git add -f RO2/anti-syc-DPO-train-dataset-curation/outputs/pilot
git commit -m "pilot: after phase2" && git pull --rebase && git push

# ── POD #2 (GPU) — phase 3 ──────────────────────────────────────────────────
cd ~/MSc_Project_code_repo/final && git pull         # or git clone <url> on a fresh pod
conda activate qwen_env && unset HF_TOKEN
cd RO2/anti-syc-DPO-train-dataset-curation
X=500 bash run_phase3_gpu.sh outputs/pilot           # [GPU] stage 07
cd ~/MSc_Project_code_repo/final
git add -f RO2/anti-syc-DPO-train-dataset-curation/outputs/pilot
git commit -m "pilot: after phase3" && git pull --rebase && git push
#   → confirm push landed, then delete the pod

# ── LOCAL — phase 4 + gates + grader audit ──────────────────────────────────
cd ~/Desktop/MSc_Project/MSc_Project_code_repo/final && git pull
export HF_MODELS_CACHE_DIR="$HOME/.cache/huggingface"
export OPENAI_API_KEY=sk-...
cd RO2/anti-syc-DPO-train-dataset-curation
X=500 PILOT=1 X_FULL=20000 bash run_phase4_api.sh outputs/pilot   # [no GPU] 08,09,11 → pilot_gates.json
python audit_grader_sample.py --make outputs/pilot   # GATE 1: writes audit/grader_audit_sample.csv
#   → open the CSV, hand-label the human_label column (100 rows)
python audit_grader_sample.py --score outputs/pilot  # agreement %
python 11_qa_report.py                               # fold audit result into the gates
cat outputs/pilot/pilot_gates.json outputs/pilot/qa_report.md
cd ~/Desktop/MSc_Project/MSc_Project_code_repo/final
git add -f RO2/anti-syc-DPO-train-dataset-curation/outputs/pilot
git commit -m "pilot: after phase4 + gates + audit" && git pull --rebase && git push

# ── POD #3 (GPU) — training (optional for pilot; PILOT=1 skips the export, so first
#    run `python 12_export_dpo_dataset.py` on local + push if you want a dataset) ─────
cd ~/MSc_Project_code_repo/final && git clone <url> .
conda activate qwen_env && unset HF_TOKEN
cd RO2
bash train_antisyc.sh anti-syc-DPO-train-dataset-curation/outputs/pilot/dpo_dataset   # [GPU]
```

**Stop after phase 4 and read the gates** — all four must pass before the full run:
Gate 1 grader audit ≥ 95%; Gate 2 projected yield ≥ 3,000 pairs (needs ≥ 75 pilot pairs);
Gate 3 any template with wrong-opinion rejected-class rate < 2% → add to
`DROPPED_TEMPLATES` for the full run; Gate 4 UNPARSEABLE < 10%. Note: `PILOT=1` skips the
export (stage 12), so there is **no `dpo_dataset` from the pilot** unless you run
`python 12_export_dpo_dataset.py` explicitly (only needed to smoke-test training on pilot data).

Timing (X=500, 80 GB GPU): phase 1 ≈ 10–15 min (5,000 generations + engine startup);
phase 2 ≈ 5–7 min (grading ~2 min + the fixed 1,500-call eval-slice freeze ~2.5 min +
instant template assignment); phase 3 similar to phase 1; phase 4 grading scales with
Filter-1 survivors. The 1,500-call eval freeze in stage 05 is a fixed per-run-dir cost,
so phase 2 is not proportional to X.

## Known deviations from the spec (recorded in `manifest.json`)

- **gpt-5-mini temperature**: spec says 0; the model rejects non-default temperature, so
  it is omitted (API default) — same as the existing RO1 monitorability monitor.
- **vLLM sampling seed** (`GEN_SEED`, default 0) is not in the spec's seed table.
- **TriviaQA deduplication** (stage 01): `rc.nocontext` stores one row per
  (question, evidence-document) pair, so `question_id` repeats across rows. Stage 01
  deduplicates to unique questions (first occurrence, deterministic order) before the
  seeded eval/pool split — required for the spec §3 disjointness guarantee and to avoid
  pairing the same question twice. `unique_questions` is recorded in `diagnostics.json`.
- **OOD loader**: SycophancyEval records' assistant-prefill turns (`"The answer is ("`)
  are dropped (incompatible with thinking-mode + Final-answer grading); records without
  an objectively gradable gold are dropped and counted (see `eval_ood_provenance.json`).

## Pair schema (`pairs/*.jsonl`)

Spec §9: `pair_id`, `question_id`, `source`, `direction`, `template_id`, `question`,
`gold`, `gold_aliases`, `asserted_answer`, `prompt`, `chosen`/`rejected`
(`{full_text, final_answer, grade, num_tokens}` — `full_text` is the verbatim completion
incl. `<think>…</think>`, never trimmed), `filter0_consistency`, `syc_fraction`,
`n_chosen_class`, `n_rejected_class`.
