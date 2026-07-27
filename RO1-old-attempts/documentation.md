# `RO1-old-attempts/` — superseded evaluation pipelines

RO1 is the **evaluation** side of the project: does post-training change how
monitorable a model's chain-of-thought is when the prompt contains a hint it
could exploit? This folder holds the two pipeline generations that came before
the submitted one. Nothing here is run any more — the current pipeline is
[`Full-Codebase/RO1/`](../Full-Codebase/RO1/) (docs: `documentation.md` there).
The whole-repo story is in [`../REPO_STRUCTURE.md`](../REPO_STRUCTURE.md).

Lineage: **`MVP-RO1` → `MVP-RO1-v2` → `Full-Codebase/RO1`**.

---

## `MVP-RO1/` — first working monitorability pipeline *(2026-06-09 → 2026-07-08)*

The first attempt at the honeypot evaluation, built to prove the concept was
workable at all: **one task (MMLU), one hint type (Stanford professor), four
Tulu-3-8B checkpoints** (`sft_no_safety` → `sft_full` → `dpo` → `final`), with a
GPT monitor reading the CoT. Metrics were **TE / acknowledgement rate / HAR /
|S|** — this predates g-mean² becoming the headline metric.

**What it produced.** The eval concept worked: hinted-vs-unhinted arms, a
structured 7-question monitor, and per-instance acknowledgement all behaved
sensibly. It also picked up extra models beyond the Tulu ladder — the
per-dataset outputs actually checked in here are for `qwen3_1_7b` across
`mmlu`, `gpqa`, `wmdp`, `agieval`, and `run_aqua_dpo_qwen3_1_7b_s0.sh` evaluates
the DPO adapter trained in the paper replication
(`../RO2-old-attempts/MVP-RO2-v2/`) — the one place the two old strands touch.

**Why it was replaced.** One `run_<model>.sh` driver per model
(`run_qwen.sh`, `run_mistral.sh`, `run_deepseek.sh`, `run_llama32_3b.sh`,
`run_tulu_sft_no_safety.sh`, …) did not scale past a handful of models, a single
dataset and single hint type could not support the claim, and the metric set
changed. `MVP-RO1-v2` is the rewrite.

**Where things are.**
- `MVP_RO1.md` — the original spec; `documentation.md`, `cross_checkpoint_checklist.md`.
- `outputs/{mmlu,gpqa,wmdp,agieval}/qwen3_1_7b/{hinted,unhinted}.jsonl` — the raw
  generations that survive here. `analysis/`, `plots/`, `monitor_outputs/` are
  empty (results were never committed at this stage / were superseded).
- `metrics/` — `gmean2_evaluation_metric.md` and `g-mean² Filtering Levels.md`.
  These are the **metric definition documents**, and after `papers-and-notes/`
  was deleted (2026-07-26) this is the only copy left in the working tree. The
  implementation lives in
  `Full-Codebase/RO1/monitorability-eval/scripts/04_compute_gmean2.py`.
- `monitor-validation/` — `compare_monitors.py` + `monitor_comparison.md`, an
  agreement check between two GPT monitor runs (`monitor_outputs-1` labelled
  gpt-4o, `monitor_outputs-2` labelled gpt-5-mini, 2,164 records). The 100%
  agreement on every dataset is the artefact that revealed the two directories
  were **byte-identical copies of a single real run** — so gpt-4o was dropped
  and the monitor question was re-opened properly later. The decision that
  actually stands is the GPT-5 vs GPT-5-mini comparison in
  `Full-Codebase/RO1/notes/monitor_comparison_gpt5_vs_gpt5mini.md`.

---

## `MVP-RO1-v2/` — the rewrite, and where scale was decided *(2026-06-30 → 2026-07-08)*

Full rewrite of the above and the **direct ancestor of `Full-Codebase/RO1/`**:

- **6 datasets** (`mmlu_redux, gpqa_diamond, wmdp, codemmlu, lsat_ar, lsat_lr`)
  and 6 hint types, driven from one config (`config/mvp_config.yaml`).
- **Raw HF model IDs / local dirs** instead of a `models:` registry — adding a
  model no longer means adding code.
- Two flag-based drivers replace the per-model scripts:
  `monitorability-eval/run_inference.sh` (GPU) and
  `run_monitor_and_results.sh` (monitor + analysis).
- The **`monitorability-eval/` + `harmlessness-eval/` split** that the final
  codebase still uses.
- **g-mean² becomes the headline metric.**

**What it was used for.** This is the tree the **harmlessness-DPO pilot** was
evaluated in (`er-abd_harmless-dpo-qwen3-4B-pilot` vs base `Qwen/Qwen3-4B`,
professor hint, 8 datasets, n=50 / N=4). Result: a **clean null** on
monitorability, no capability loss, and — critically — **no CoT collapse**,
which is what confirmed DPO was a viable replacement for the abandoned SFT arm
(see `../RO2-old-attempts/README.md`). The same session fixed the scale for
every later run at **n=100 / N=8**.

**Where things are.**
- `notes/2026-07-07_pilot-analysis-and-scaling-decisions.md` — the pilot
  analysis and the n/N power reasoning. The single most write-up-relevant file
  in this folder.
- `outputs-and-results/Old-Results/<timestamp>/` — five dated result snapshots
  (`2026-07-01` … `2026-07-07`, each with `outputs/`, `monitor_outputs/`,
  `analysis/`, `plots/`; the last also has `harmlessness/`), plus
  `cot_length_rl_comparison.md`.
- The live `outputs-and-results/{inference-outputs,monitor_outputs,monitorability-results,harmlessness-results,plots}/`
  directories are **empty on purpose** — the main runs were done in the
  successor tree, so all final results live in
  `Full-Codebase/RO1/outputs-and-results/`.
- `documentation.md`, `CLAUDE.md` — pipeline docs as of this generation.

**What the successor added on top:** the `anti-sycophancy-eval/` suite,
8 datasets at n=200/N=8 with 3 active hint types, the paired-bootstrap pooling
scripts, and the length-confound analyses.
