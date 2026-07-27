# `RO2-old-attempts/` — superseded training pipelines

RO2 is the **training** side of the project: post-train a model (safety /
anti-sycophancy) so that RO1 can measure what happened to its CoT
monitorability. This folder holds everything that came before the submitted
training code — the abandoned SFT strand, the external codebase used to learn
DPO, and the DPO pipeline that was eventually promoted. Nothing here is run any
more; the current training code is [`Full-Codebase/RO2/`](../Full-Codebase/RO2/)
(docs: `DOCUMENTATION.md` there). Repo-wide story:
[`../REPO_STRUCTURE.md`](../REPO_STRUCTURE.md).

Chronological order: **`MVP-RO2-v1` (SFT, dead end) → `MVP-RO2-v2` (paper
replication) → `MVP-RO2-v3` (DPO pipeline) → `Full-Codebase/RO2`.**

> **Git-archaeology warning.** These folders were renumbered on 2026-07-26. In
> history before that date, `MVP-RO2` = today's `v1/RO2-SFT-Qwen3-4B`,
> `MVP-RO2-llama` = `v1/RO2-SFT-llama`, `Reward-Hacking-in-Explanations` = `v2/…`,
> and the old `MVP-RO2-v2` = today's `v3`. The old top-level `MVP-RO2-v1` and
> `MVP-RO2-old` were deleted (recoverable via `git show 804711c:MVP-RO2-old/...`).

---

## `MVP-RO2-v1/` — the SFT strand *(2026-06-09 → 2026-06-24)*

The project's original RO2 plan: get a harmlessness dose into the model with
**supervised fine-tuning** on safe responses. It dead-ended, and that dead end
is what redirected the whole project to DPO — so this folder is the evidence
behind a methodology decision, not just an old draft.

### `RO2-SFT-Qwen3-4B/` *(2026-06-09 → 2026-06-22)*
LoRA SFT of `Qwen/Qwen3-4B` on the safe (chosen) responses of **PKU-SafeRLHF**
conflict cases (10,813 cases at the revision used), 3 epochs → one adapter per
epoch pushed to the Hub
(`er-abd/qwen3-4b-sft-harmlessness-pku-adapter-epoch{1,2,3}`) so the RO1 eval
could load them. Layout: `sft/` package + numbered `scripts/` (00 seq-len
preflight → 01 train → 02 push → 03 generation check → 05 think-token eval).
Spec: `sft_mvp_ro2_experiment_design.md`.

**The finding that changed the project:** answer-only SFT **collapsed Qwen3's
chain-of-thought to zero**. Measured over 200 eval prompts (greedy): base
Qwen3-4B **413.5** mean think tokens vs **0.0** for every epoch-1/2/3 adapter —
collapsed by epoch 1, `frac_non_empty_think = 0.000`. Masking the injected empty
`<think></think>` scaffold was necessary but not sufficient: training on bare
direct answers teaches the model Qwen3's *non-thinking* format. A model with no
CoT cannot be used in a **CoT-monitorability** study, so these checkpoints are
usable only as a negative result.

**Results:** `dev-notes/cot-suppression-investigation.md` (2026-06-20) — the
full investigation with the think-token table and root-cause analysis.
`documentation.md` describes the pipeline.

### `RO2-SFT-llama/` *(2026-06-24)*
Attempted workaround: port the same SFT pipeline to
`meta-llama/Llama-3.2-3B-Instruct` on Anthropic `hh-rlhf` `harmless-base`
single-turn examples (12,254 of them), 4 epochs → 4 evenly-spaced adapters,
plain PEFT/TRL. Because **Llama-3.2 has no thinking mode**, all `<think>`
machinery was deleted (`sft/generate.py`, `sft/think_stats.py`, the
generation/think-token scripts).

**Outcome:** that is exactly why the strand ends — removing the CoT removes the
thing being measured. **Decision: abandon SFT as the main training method and
move to DPO**, which shapes preferences without retraining the response format.
(SFT returns much later, but only as a controlled *ablation arm* inside
`Full-Codebase/RO2/train_sft.py`, trained on the same DPO data to separate "DPO
the method" from "preference data the signal".) No results were kept here — the
port was never taken past training setup.

---

## `MVP-RO2-v2/` — paper replication, used as a DPO reference *(2026-06-27 → 2026-07-04)*

`Reward-Hacking-in-Explanations/` is an **external codebase** (Ferreira et al.,
*Truthful or Fabricated?*) brought in and run locally after the SFT pivot, to
learn a working DPO + reward-model evaluation setup before writing one. See
`info.txt`, `CODEBASE.md`, `TruthfulOrFabricated - Ferreira et al.md`,
`RUN_ON_POD.md`.

**What was actually run:** a small-scale DPO replication on **AQuA with
Qwen3-1.7B**, pushing adapters as
`er-abd/aqua-dpo-qwen3-1.7B-sk-llama_original-seed-0`, plus the paper's
reward-model / best-of-N analyses.

**Results:** `results/qwen_main_qwen3-1.7B.md`,
`results/qwen_compare_qwen3-1.7B.md`, `results/qwen_finegrained_qwen3-1.7B.md`,
figure `figures/qwen_main_qwen3-1.7B.{png,pdf}`, raw generations under
`outputs/aqua/`, analysis code in `analysis/`.

**Link to the rest of the project:** no thesis code descends from it directly,
but the design of the DPO pipeline in `v3` was informed by it, and the DPO'd
Qwen3-1.7B adapter it produced was then evaluated through the old eval pipeline
via `../RO1-old-attempts/MVP-RO1/run_aqua_dpo_qwen3_1_7b_s0.sh` — the only
crossover between the two old strands. Its checked-in virtualenv (`my_env/`) was
deleted 2026-07-26; recreate from `requirements.txt` if it ever needs to run.

---

## `MVP-RO2-v3/` — the harmlessness DPO pipeline *(2026-07-05 → 2026-07-09)*

The real RO2 pipeline, and the **direct ancestor of `Full-Codebase/RO2/`**
(`train_dpo.py` is byte-identical between the two — verified by checksum).

- `DPO-train-dataset-curation/` — WildJailbreak harmful prompts → Qwen3-4B
  generations (N=10) → Qwen3Guard safety labels → Beaver cost model →
  preference pairs. Numbered resumable stages `01_prepare_prompts.py` →
  `06_export_dpo_dataset.py`, plus `run_all.sh`, `run_from_03.sh`,
  `run_pilot_and_sanity.sh`, and `merge_runs.py` (merge several sampling runs to
  hit an exact pair target).
- `train_dpo.py` + `train.sh` — the LoRA DPO trainer.

It scales up the 5-stage prototype that lived in
`Experiments/DPO-data-&-RM-suitability-check/`, now at
`Full-Codebase/RO2/DPO-data-&-RM-suitability-check/`, which is where the
"is this reward model / dataset fit for purpose?" studies were done before
paying for a full run.

**Results kept here.**
- `DPO-train-dataset-curation/outputs/pilot/` — the pilot curation:
  **867 pairs from SAMPLE=4500 (19.3% end-to-end survival)**, with the filtering
  funnel, per-subset mix and Beaver-vs-Qwen3Guard labeler agreement
  (85.6% / 88.3%, point-biserial r=0.509) in `report.md`; also
  `final_pairs_margins.csv`, `cost_spread.png`, `sample_pairs.json`, and the
  resumable `.joblib` state.
- `DPO-train-dataset-curation/outputs/main/` — the scaled-up curation run
  (same artefact set); `smoke8/`, `smoke_batchfix/` are throughput smoke tests.
- `PILOT_TRAINING_ANALYSIS.md` — the pilot **training-signal** analysis that
  validated the hyperparameters actually used for every later run
  (LR 5e-6, cosine, β 0.1, 3 epochs, batch 1×16, MAX_LENGTH 12288; best
  checkpoint step 147, reward-accuracy 0.874, peak 75.2/85 GB → needs an 80GB
  card). Note the scope caveat stated in that file: these are optimization-health
  signals, not behavioural results — the behavioural verdict came from RO1
  (`../RO1-old-attempts/MVP-RO1-v2/notes/2026-07-07_pilot-analysis-and-scaling-decisions.md`).
- `checkpoints/qwen3-4B/harmless_dpo/` — local adapter/eval artefacts
  (`eval_metrics.jsonl`, `train_log_history.json`, `best_checkpoint.txt`).
- Write-up material next to the code: `documentation.md`, `README.md`,
  `labeler_comparison.md` (WildGuard vs Qwen3Guard),
  `NOTES_length-confound_and_benign-axis.md`.

**What the successor added:** the anti-sycophancy curation module, the
length-normalized trainer `train_dpo_norm.py` (after DPO was found to inflate
response length), the SFT ablation arm, and the full-scale runs whose
checkpoints (`…harmless-dpo…step-735`, `…anti-syc-dpo…step-520`,
`…harmless-sft…step-735`) are what the write-up reports. All final training
artefacts and results are under `Full-Codebase/RO2/` and
`Full-Codebase/RO1/outputs-and-results/`.
