# Documentation — end-to-end runbook

Two curation pipelines and three training objectives sharing one wrapper:
**curate a preference dataset → train the model**. (Behavioural evaluation of the pushed
checkpoints lives in the sibling [`../RO1/`](../RO1/) evaluation suite.)

```
harmlessness-DPO-train-dataset-curation/   # harmlessness curation (WildJailbreak → preference pairs)
anti-syc-DPO-train-dataset-curation/       # anti-sycophancy curation (TriviaQA → preference pairs
                                           #   + the RO1 eval's frozen files; includes the frozen
                                           #   curation spec anti_sycophancy_dpo_dataset_curation_spec.md)
DPO-data-&-RM-suitability-check/           # FROZEN precursor study (PKU/WildJailbreak data + reward-model
                                           #   suitability checks) — historical, not part of the runbook;
                                           #   see its own README.md
train_dpo.py                  # LoRA DPO trainer (self-contained; consumes either module's export)
train_dpo_norm.py             # LoRA length-normalized DPO trainer (open-instruct dpo_norm; avg log-probs)
train_sft.py                  # LoRA SFT-on-chosen trainer (TRL SFTTrainer) — DPO ablation arm
convert_dpo_to_sft.py         # dpo_dataset -> sft_dataset (drop `rejected`; records provenance SHAs)
verify_sft_tokenization.py    # go/no-go gate: proves SFT tokenization == DPO chosen pathway
train.sh                      # one-command training wrapper (harmlessness defaults; METHOD=dpo|dpo_norm|sft)
train_sft.sh                  # SFT arm driver: convert -> verify -> METHOD=sft train.sh (idempotent prep)
train_antisyc.sh              # anti-sycophancy wrapper (pins output dir / HF repo / W&B names)
requirements.txt              # unified environment
SFT_Arm_Developer_Spec.md     # rationale + frozen spec for the SFT ablation arm
PILOT_TRAINING_ANALYSIS.md    # analysis of the pilot DPO training run
SMOKE_RUN_LOG-template.md     # fill-in log for the small-scale HP-validation run
```

## 0. Setup (once)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The Hugging Face token (write access, `er-abd` account; needed for the gated
WildJailbreak/WildGuard and for checkpoint pushes) ships in the **committed `.env`** — this
repo is private, so a fresh pod works straight after `git clone` with no token setup. If the
repo is ever made public: rotate the token and re-add `.env` to `.gitignore` first (see the
note inside `.env`). On a pod, uncomment `HF_MODELS_CACHE_DIR=/workspace/hf_cache` in `.env`
so model caches land on the volume disk.

Requires a CUDA GPU for harmlessness curation and for training. The anti-sycophancy
module needs the GPU only for its two generation stages; its grading stages need
`OPENAI_API_KEY` (gpt-5-mini) in `.env` instead and run on any machine.

## 1. Curate the DPO dataset — one command

```bash
bash harmlessness-DPO-train-dataset-curation/run_all.sh
```

Runs stages 01→06 (prepare prompts → generate Qwen3-4B responses → Qwen3Guard safety labels
(Qwen3Guard-Gen-4B; WildGuard is held out as the later verifier) → Beaver cost scoring →
filter/report → **export**). By default it curates **ALL** unique
WildJailbreak harmful prompts (`vanilla_harmful` + `adversarial_harmful`, train config);
set `SAMPLE=<int>` for a smaller seeded pilot. Writes a fresh
`harmlessness-DPO-train-dataset-curation/outputs/run_<timestamp>/` containing, among the diagnostics:

- **`dpo_dataset/`** — the handoff: a HF `DatasetDict({train,valid})` with columns
  `prompt` / `chosen` / `rejected`. This is what training consumes.
- **`run_all.log`** — the run's complete console output (all stages, stdout+stderr),
  appended live, so the log survives closed terminals and ships with the run folder.

Useful knobs (env vars, all optional):

```bash
SAMPLE=300 bash harmlessness-DPO-train-dataset-curation/run_all.sh                      # pilot sample instead of ALL prompts
SEED=1 bash harmlessness-DPO-train-dataset-curation/run_all.sh                          # single seed for every stage (default 0; same var train.sh uses)
RUN_OUTPUT_DIR=outputs/my_run bash harmlessness-DPO-train-dataset-curation/run_all.sh   # pin the output folder
VALID_FRACTION=0.1  bash harmlessness-DPO-train-dataset-curation/run_all.sh             # valid split size (seeded random sample of the final pairs)
STRATIFY_VALID=true bash harmlessness-DPO-train-dataset-curation/run_all.sh             # 50/50 vanilla/adversarial valid mix instead of random
EXPORT_FIELD=answer bash harmlessness-DPO-train-dataset-curation/run_all.sh             # export visible answer only (default: full, with CoT)
EXCLUDE_RUNS=outputs/pilot bash harmlessness-DPO-train-dataset-curation/run_all.sh      # skip prompts a previous run already used
```

**Two-phase curation** (pilot for a training smoke test, then a top-up of *new* prompts,
merged into one dataset):

```bash
cd harmlessness-DPO-train-dataset-curation
SAMPLE=4500 RUN_OUTPUT_DIR=outputs/pilot bash run_all.sh
SAMPLE=24000 EXCLUDE_RUNS=outputs/pilot RUN_OUTPUT_DIR=outputs/main bash run_all.sh
TARGET_PAIRS=5000 python merge_runs.py outputs/pilot outputs/main --out outputs/merged
cd .. && bash train.sh harmlessness-DPO-train-dataset-curation/outputs/merged/dpo_dataset
```

`TARGET_PAIRS` (optional) caps the merged pool at an exact size via a seeded random
downsample before the export — the run oversamples prompts for safety (filter survival is
~19% and noisy) and the merge trims the surplus, so the final dataset lands on exactly the
target. Unset = keep every merged pair.

This works from a fresh clone on a new pod: the two files the workflow reads —
`01_prompts_meta.joblib` (per-run consumed prompts, for `EXCLUDE_RUNS`) and
`final_pairs.joblib` (per-run final pairs, for `merge_runs.py`) — are git-tracked via
narrow `.gitignore` re-includes (the other, heavy stage joblibs stay ignored).

If a run's stages 01/02 (prompt prep + the expensive vLLM generation) already finished,
resume from classification onward instead of regenerating:
`RUN_OUTPUT_DIR=outputs/main bash harmlessness-DPO-train-dataset-curation/run_from_03.sh`
(stages 03/04 are checkpointed, so re-running after an interrupt continues where it stopped).

Smoke test first (`SAMPLE=8 bash harmlessness-DPO-train-dataset-curation/run_all.sh`) to confirm the
whole chain produces `dpo_dataset/` before a full run. `SAMPLE` is the only smoke-test
knob — `N` and `MAX_TOKENS` are fixed across all runs so runs stay mergeable/comparable.

## 1b. Curate the anti-sycophancy DPO dataset

Implements the frozen spec
`anti-syc-DPO-train-dataset-curation/anti_sycophancy_dpo_dataset_curation_spec.md` (Stages A–J).
Full details, stage map, caching semantics, and pilot-gate workflow:
**[anti-syc-DPO-train-dataset-curation/README.md](anti-syc-DPO-train-dataset-curation/README.md)**.

Key differences from the harmlessness module: grading and wrong-answer generation use
**gpt-5-mini** (needs `OPENAI_API_KEY` in `.env`; no local judge model), only the two
generation stages need a GPU, and the pipeline is split into four phase drivers so API
stages run off-GPU:

```bash
# smoke (one box: GPU + API key) — verifies the whole chain end to end:
X=5 RUN_OUTPUT_DIR=outputs/smoke bash anti-syc-DPO-train-dataset-curation/run_all.sh

# pilot (X=500, blocking gates) and full run (X=20000) — split GPU/API phases:
cd anti-syc-DPO-train-dataset-curation
X=500 bash run_phase1_gpu.sh outputs/pilot                        # GPU: sample + Filter-0 gen
X=500 bash run_phase2_api.sh outputs/pilot                        # API: grade + wrong answers + templates
X=500 bash run_phase3_gpu.sh outputs/pilot                        # GPU: Filter-1 gen
X=500 PILOT=1 X_FULL=20000 bash run_phase4_api.sh outputs/pilot   # API: grade + pairs + gates
python audit_grader_sample.py --make outputs/pilot                # gate 1: hand-label 100 → --score
# full run: same four phases with X=20000 (no PILOT), optionally DROPPED_TEMPLATES=<gate-3 list>
```

Handoff: `outputs/<run>/dpo_dataset/` — the identical `prompt`/`chosen`/`rejected`
DatasetDict contract, so training needs no changes. (The spec's Stage I sham control was
dropped by owner decision; there is no stage 10 script and no sham export.) The run dir also carries
`reserved/eval_id_triviaqa.jsonl` + `reserved/eval_ood_sycophancyeval.jsonl` — the frozen
eval files the RO1 `anti-sycophancy-eval` imports once (from `../RO1/`:
`python anti-sycophancy-eval/scripts/01_prepare_eval_data.py --source-dir <run>/reserved`).
Use `python backup_run_to_hf.py --push/--pull <run_dir>` to move run dirs between the GPU
box and the API box (heavy JSONLs are gitignored).

## 2. Train — one command

```bash
bash train.sh                                            # harmlessness, standard DPO
METHOD=dpo_norm bash train.sh <path/to/dpo_dataset>      # harmlessness, length-normalized DPO
METHOD=sft      bash train.sh <path/to/sft_dataset>      # harmlessness, SFT-on-chosen ablation
bash train_antisyc.sh <path/to/dpo_dataset>              # anti-sycophancy
```

### Objective: DPO vs length-normalized DPO (`METHOD`)

`train.sh` runs two interchangeable preference objectives on the **same** exported dataset,
selected by `METHOD`:

- **`METHOD=dpo`** (default) — standard DPO (`train_dpo.py`). Unchanged.
- **`METHOD=dpo_norm`** — length-normalized DPO (`train_dpo_norm.py`). Standard DPO scores a
  completion by the **sum** of its per-token log-probs, which mechanically rewards longer
  generations and drives the post-training response-length inflation documented in
  `../RO1/notes/harmless_dpo_response_length_inflation.md`. Length-normalized DPO scores by the
  **average** instead (sum ÷ #completion-tokens), removing that length bias while keeping DPO's
  reference-based sigmoid loss. Length normalization for DPO was noted in the original DPO work
  (Rafailov et al.) and established as a strong, well-tuned variant by Meng et al. 2024 (SimPO,
  which reports it as a baseline). The averaging follows AllenAI **open-instruct**'s `dpo_norm`
  ([`open_instruct/dpo_utils.py`](https://github.com/allenai/open-instruct/blob/main/open_instruct/dpo_utils.py),
  `_get_batch_logps(average_log_prob=True)`), applied here to TRL's sigmoid DPO via a thin
  `DPOTrainer` subclass (`LengthNormalizedDPOTrainer`) that divides the summed completion
  log-probs by completion length on both the policy and reference passes. It reuses all of
  `train_dpo.py`'s machinery (LoRA, eval/checkpoint schedule, per-step Hub push, best-checkpoint
  selection, metric logging, dataset formatting) so the two are directly comparable —
  `train_dpo.py` is not modified.

`METHOD=dpo_norm` changes the default `OUTPUT_DIR` (`checkpoints/qwen3-4B/harmless_dpo_norm`),
`HUB_MODEL_ID` (`er-abd/harmless-dpo-norm-qwen3-4B`), and W&B project so the two methods'
artifacts never collide. Loss knob (env): `DPO_NORM_BETA` (default 5 — larger than standard
DPO's `DPO_BETA`=0.1 because averaged per-token log-probs are ~1/|y| smaller). Shared:
`LEARNING_RATE` (default 5.0e-6). Example:

```bash
METHOD=dpo_norm DPO_NORM_BETA=5 bash train.sh path/to/dpo_dataset
```

Note `dpo_norm` stays **reference-based** (`loss_type="sigmoid"`), so — like standard DPO with
PEFT — the reference is the base model with the LoRA adapter disabled (no separate model copy),
and the length normalization is applied identically to the reference log-probs.

### Objective: SFT on the chosen responses (`METHOD=sft`)

The SFT arm is a **controlled ablation of DPO** (full rationale: `SFT_Arm_Developer_Spec.md`).
DPO's gradient simultaneously pushes up the chosen response and pushes down the rejected one —
and the rejected response contains a full CoT, so DPO applies a negative gradient to reasoning
tokens no judge ever evaluated. `METHOD=sft` trains `train_sft.py` (TRL `SFTTrainer`) on the
**chosen responses only** — DPO with the negative term removed (equivalently:
rejection-sampling fine-tuning, RAFT/RFT). Everything else is held identical to the DPO run:
base model, LoRA config (r16/α32/dropout 0.05, same target modules), max lengths, seed policy,
optimizer/schedule (adamw, cosine, warmup 0.1, wd 0.01, bs 1 × grad-accum 16, 3 epochs), and
the eval/checkpoint schedule — so with the same data the SFT run produces checkpoints at the
same steps as the DPO run.

**One-command driver** — `train_sft.sh` runs the whole workflow and skips anything already
done (conversion is skipped when the `sft_dataset` already exists; verification is skipped
when it already PASSED for the exact same dataset content, tracked by a hash-stamped
`.verify_passed` marker inside the `sft_dataset`):

```bash
bash train_sft.sh harmlessness-DPO-train-dataset-curation/outputs/main/dpo_dataset
PREP_ONLY=1 bash train_sft.sh <dpo_dataset>    # convert+verify only (any CPU box), no training
```

Prefer passing the dpo_dataset path explicitly — the no-argument form picks the
most-recently-modified `outputs/*/dpo_dataset`, and identical mtimes (e.g. after a fresh git
checkout) make that pick arbitrary (it can land on a smoke run's export).

The three underlying steps, if you need to run them individually:

```bash
# 1. Convert: exact dpo_dataset fed to DPO -> its OWN run folder outputs/<run>_sft/sft_dataset
#    (keep prompt+chosen, drop rejected; same rows, same order, same train/valid split;
#    provenance SHAs recorded in sft_dataset/provenance.json)
python convert_dpo_to_sft.py harmlessness-DPO-train-dataset-curation/outputs/main/dpo_dataset

# 2. Verify (GO/NO-GO GATE): prove per example that the SFT training sequence is
#    token-identical to what DPOTrainer trained on for the chosen response, that the loss
#    covers response tokens only with the final EOS included, and that nothing would be
#    truncated. Must print PASS before any GPU time is spent.
python verify_sft_tokenization.py \
    --dpo_dataset harmlessness-DPO-train-dataset-curation/outputs/main/dpo_dataset \
    --sft_dataset harmlessness-DPO-train-dataset-curation/outputs/main_sft/sft_dataset

# 3. Train
METHOD=sft bash train.sh harmlessness-DPO-train-dataset-curation/outputs/main_sft/sft_dataset
```

Deliberate differences from the DPO arm (everything not listed here is matched):

- **Loss**: causal-LM on the completion only (`completion_only_loss` — prompt tokens masked,
  EOS in loss), no reference model, no rejected term.
- **Learning rate**: default **1e-4** (vs DPO's 5e-6). LoRA SFT sits in the 1e-4–2e-4
  literature range (QLoRA used 2e-4; TRL's 2e-5 default targets full fine-tuning, and LoRA
  needs ~5–10× that); the data is on-policy/self-generated, so gradients are small and DPO's
  5e-6 would near-guarantee a null treatment. If Gate 1 (safety gain) fails at 1e-4, sweep the
  ladder 2e-5 / 5e-5 / 1e-4 / 2e-4 via `LEARNING_RATE=... METHOD=sft bash train.sh ...`.
- **Best checkpoint** is selected by **`eval_loss`** (lower is better; SFT has no
  `eval_rewards/accuracies`), with `eval_mean_token_accuracy` recorded alongside. The
  dose–response analysis uses ALL pushed step checkpoints regardless.
- **Checkpoint cadence knob**: `EVAL_FRACTION` (default 0.2 = DPO's schedule). The spec allows
  denser SFT checkpoints for the dose–response curve — e.g. `EVAL_FRACTION=0.1` for ~10
  intermediate checkpoints.
- **No `--max_prompt_length`**: `SFTConfig` doesn't have it, and SFT truncation is from the
  RIGHT (which would silently drop the EOS from the loss). Instead, the verifier asserts that
  no templated prompt exceeds the DPO prompt budget (4096) and no full sequence exceeds
  `MAX_LENGTH` (12288) — on the main export the max sequence is ~8.6k tokens, so there is
  ample headroom. If a future dataset violates these asserts, raise `MAX_LENGTH` rather than
  accept truncation, and record the deviation.

Formatting is byte-identical to the DPO chosen pathway by construction (same
`apply_chat_template(..., enable_thinking=True)` prompt, same manual
`<|im_start|>assistant\n…<|im_end|>\n` wrap that preserves the `<think>` block, plus the same
trailing EOS token that `DPOTrainer.tokenize_row` appended — including its double-EOS quirk;
do not "fix" that in `train_sft.py`, the ablation depends on it).

**Provenance** (auditable lineage): `convert_dpo_to_sft.py` writes
`sft_dataset/provenance.json` with SHA-256 hashes of the source `dpo_dataset` and the derived
`sft_dataset`; `train_sft.py` recomputes the dataset hash at startup, warns loudly on
mismatch, and writes the merged record to `OUTPUT_DIR/dataset_provenance.json` (also mirrored
into the W&B config).

Defaults for `METHOD=sft`: `OUTPUT_DIR=checkpoints/qwen3-4B/harmless_sft`,
`HUB_MODEL_ID=er-abd/harmless-sft-qwen3-4B` (checkpoints push to
`er-abd/harmless-sft-qwen3-4B-step-<N>`), W&B project `harmless-sft-qwen3-4B`, log
`logs/train_sft.log`.

After training, follow the spec's evaluation gates **in order**: Gate 1 — safety/refusal
classifiers (`../RO1/harmlessness-eval/`) on base vs SFT checkpoints; only if safety measurably
improved, Gate 2 — the frozen monitorability suite. Never report monitorability numbers for a
null-treatment SFT model.

### Shared behaviour

The anti-syc wrapper takes an explicit dataset path and sets
`OUTPUT_DIR` (`checkpoints/qwen3-4B/anti_syc_dpo`), `HUB_MODEL_ID`
(`er-abd/anti-syc-dpo-qwen3-4B`), W&B names, and `MAX_PROMPT_LENGTH=1024` (trivia
prompts are short), then call `train.sh` — the trainer itself is identical across
objectives. Never train anti-syc through bare `train.sh`: its default `HUB_MODEL_ID`
would push checkpoints into the harmlessness HF repos. Override settings as needed:

```bash
bash train.sh path/to/dpo_dataset         # a specific dataset
EPOCHS=5 bash train.sh                     # more epochs (also works with the wrappers)
MODEL=Qwen/Qwen3-4B OUTPUT_DIR=checkpoints/run1 bash train.sh
MAX_LENGTH=8192 bash train.sh              # lower if you hit GPU OOM
```

Anti-syc `MAX_LENGTH` note: completions can theoretically reach 16384 tokens (the curation
max_new_tokens), but trivia CoT is much shorter in practice; check the curation
`qa_report.md` length stats and raise `MAX_LENGTH` (default 12288) only if the >12k share
is non-negligible.

Output: LoRA adapters under `checkpoints/qwen3-4B/harmless_dpo/` (`..._dpo_norm/` for
`METHOD=dpo_norm`, `..._sft/` for `METHOD=sft`). The best checkpoint — by eval
reward-accuracy for the DPO methods, by `eval_loss` for SFT — is kept as `checkpoint-best`;
others are pruned. Its identity — the winning step and the exact `<HUB_MODEL_ID>-step-<N>` HF
repo it was pushed to — is written to `OUTPUT_DIR/best_checkpoint.txt` (and mirrored into the
W&B run summary). A full log is written to `logs/train_<method>.log` (`logs/train_dpo.log` /
`logs/train_dpo_norm.log` / `logs/train_sft.log`). Override the log path with the `LOG_FILE`
env var. Anti-sycophancy runs (`train_antisyc.sh`) set it automatically to
`logs/anti_syc_train_<method>_<run_tag>.log`, where `<run_tag>` is the curation output dir
feeding the run (e.g. `logs/anti_syc_train_dpo_pilot.log` vs `logs/anti_syc_train_dpo_full_run.log`),
so pilot and full runs never overwrite each other or the harmlessness logs.

**Eval/checkpoint schedule**: fixed step intervals are epoch-count-proof and derived from the
trainer's own total-step count (dataset size × epochs × batch/grad-accum settings), not a fixed
number of steps. Eval + checkpoint fire at:
- step 0 (a pre-training baseline, before any optimizer step)
- every ~20% of total training steps (so ~5 evals spread across the run)
- the final training step, always — even if it doesn't land on a 20%-boundary

`--eval_strategy` / `--save_strategy` are forced to `"no"` inside `train_dpo.py` /
`train_dpo_norm.py` / `train_sft.py` regardless of CLI flags, so this schedule is the only thing
that triggers eval/save — nothing here sets `num_train_epochs` early-stop behavior; the run
always trains through every epoch. (For `METHOD=sft` the ~20% interval is tunable via
`EVAL_FRACTION`.)

**Eval metrics** — at every eval point the trainer writes one line to `OUTPUT_DIR/eval_metrics.jsonl`
with `step`, `epoch`, and all eval metrics for the method (DPO: `eval_loss`,
`eval_rewards/accuracies`, `eval_rewards/margins`, `eval_logps/*`, …; SFT: `eval_loss`,
`eval_mean_token_accuracy`). Plain JSONL for easy plotting of the training curves:
`pandas.read_json("eval_metrics.jsonl", lines=True)`.

`OUTPUT_DIR/train_log_history.json` is the **complete** structured record (every logging-step
training loss / LR / grad-norm plus every eval point) for offline curves without W&B;
`eval_metrics.jsonl` is the eval-only subset. Each eval row also carries GPU memory snapshots
(`gpu_mem_peak_reserved_gb` vs `gpu_mem_total_gb`), and a final summary lands in
`OUTPUT_DIR/gpu_memory.json` — read the peak-reserved headroom after a smoke run to decide
whether `per_device_train_batch_size` can be raised.

Both `run_all.sh` and `train.sh` run a **preflight** before any GPU work: `run_all.sh` verifies
the HF token is valid and the account can read the gated `allenai/wildjailbreak`; `train.sh`
verifies the HF credentials exist and the token is not read-only (checkpoint pushing is
always on).

These are training diagnostics for checkpoint selection and optimisation health — **not** a
behavioural evaluation. Actual safety + monitorability evaluation is done **separately, offline,
on the pushed checkpoints** (like the reference repo `PedroMLF/Reward-Hacking-in-Explanations`,
which generates offline from saved checkpoints rather than during training). Training itself
never generates.

**Experiment tracking (Weights & Biases)** — on by default (`REPORT_TO=wandb`). The HF `Trainer`
streams the full training log (loss, learning rate, per-eval DPO metrics, gradients/system stats)
to W&B, where the curves and figures render automatically — the same metrics also land in
`eval_metrics.jsonl`. Run `wandb login` once (or set `WANDB_API_KEY`) first; name the run with
`WANDB_PROJECT` / `RUN_NAME`. Disable with `REPORT_TO=none bash train.sh`.

For a long run, use tmux so an SSH drop doesn't kill it:

```bash
tmux new -s train
bash train.sh
# detach: Ctrl-b then d   |   reattach: tmux attach -t train
```

### Push adapters to the Hugging Face Hub (always on)

Hub pushing is **always enabled**. `train.sh` defaults `HUB_MODEL_ID` to
`er-abd/harmless-dpo-qwen3-4B` (the write token ships in the committed `.env`) and forwards
`--push_to_hub --hub_model_id`. Override the repo prefix with:

```bash
HUB_MODEL_ID=<your-namespace>/harmless-dpo-qwen3-4B bash train.sh
```

Every checkpoint the schedule above produces (step 0, each ~20%-interval checkpoint, and the
final step) is pushed to its own private repo `<HUB_MODEL_ID>-step-<N>` — none are skipped, even
though `clean_checkpoints()` later prunes all but `checkpoint-best` locally.

## 3. Inference (external)

The trained artifact is the LoRA **adapter** at `checkpoints/qwen3-4B/harmless_dpo/checkpoint-best`.
Load it on top of the base model — no merge step:

```python
# transformers
from peft import PeftModel
from transformers import AutoModelForCausalLM
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B")
model = PeftModel.from_pretrained(base, "checkpoints/qwen3-4B/harmless_dpo/checkpoint-best")
```

or serve it directly with vLLM (`--enable-lora` / `LoRARequest`).

## Quick reference

| Step | Command |
|------|---------|
| Setup | `pip install -r requirements.txt` (HF token ships in the committed `.env`; add `OPENAI_API_KEY` there for anti-syc) |
| Curate harmlessness dataset | `bash harmlessness-DPO-train-dataset-curation/run_all.sh` |
| Curate anti-syc dataset | `anti-syc-DPO-train-dataset-curation/run_all.sh` (smoke) or `run_phase{1..4}_{gpu,api}.sh` |
| Train harmlessness (DPO) | `bash train.sh` |
| Train harmlessness (length-normalized DPO) | `METHOD=dpo_norm bash train.sh <dpo_dataset>` |
| Train harmlessness (SFT-on-chosen ablation) | `bash train_sft.sh <dpo_dataset>` (convert → verify → train, idempotent; `PREP_ONLY=1` for data prep only) |
| Train anti-syc | `bash train_antisyc.sh <dpo_dataset>` |

## Notes

- **TRL version**: `train_dpo.py` and `train_dpo_norm.py` both use the TRL `DPOTrainer` /
  `DPOConfig` / `TrlParser` API (`processing_class`, `get_peft_config`); `train_dpo_norm.py`
  subclasses `DPOTrainer` (`LengthNormalizedDPOTrainer`) to length-average the completion
  log-probs; `train_sft.py` uses the same-version `SFTTrainer` / `SFTConfig`.
  `requirements.txt` pins `trl==0.22.2` **exactly** (not a floor) — `train.sh`'s CLI
  flags for all three methods were verified against this version, the length-normalization hook
  relies on TRL internals (`concatenated_forward` returning summed `chosen_logps`/`rejected_logps`
  and `concatenated_inputs(...)["completion_attention_mask"]`), and the SFT arm's DPO-matched
  tokenization relies on this version's `DPOTrainer.tokenize_row` (appends EOS to completions)
  and `SFTTrainer` prompt/completion behaviour (no chat template on string columns; `add_eos`
  no-ops when the completion already ends with EOS; `completion_only_loss` masks the prompt) —
  all of which have changed across TRL versions before. If you deliberately upgrade TRL, run
  `python train_dpo.py --help`, `python train_dpo_norm.py --help` and
  `python train_sft.py --help`, re-verify the subclass hook against the new
  `concatenated_forward`, re-run `verify_sft_tokenization.py` (it exercises the real TRL
  pathways), then re-pin.
- **dpo_norm knob** (`METHOD=dpo_norm`): `DPO_NORM_BETA` (default 5 — larger than standard DPO's
  0.1 because averaged per-token log-probs are ~1/|y| smaller; treat 5 as a starting point and
  tune). The length-averaging follows open-instruct's `dpo_norm` and is the fix for DPO's
  response-length inflation — background and evidence in
  `../RO1/notes/harmless_dpo_response_length_inflation.md`. The subclass assumes completions are
  **not** truncated by `max_length` (train.sh's 12288 default is well above the ~1.5k-token
  completions); if you lower `max_length` below the completion length, the normalization
  denominator would differ from the tokens actually scored — revisit the subclass then.
- **Dataset schema contract**: DPO training expects exactly `prompt` / `chosen` / `rejected`
  string columns with `train` / `valid` splits — which is what stage 06 emits. `chosen`/`rejected`
  retain the `<think>…</think>` reasoning (thinking-mode preference training); both `train_dpo.py`
  and `train_dpo_norm.py` wrap them as assistant turns without stripping the CoT. The SFT arm
  expects the `prompt` / `chosen` derivative produced by `convert_dpo_to_sft.py` (same splits,
  same rows, `rejected` dropped); `train_sft.py` refuses a dataset that still has a `rejected`
  column so the provenance-recording converter can't be skipped. The converter is generic
  across objectives — it works on any `dpo_dataset` following this contract (harmlessness and
  anti-syc alike).
