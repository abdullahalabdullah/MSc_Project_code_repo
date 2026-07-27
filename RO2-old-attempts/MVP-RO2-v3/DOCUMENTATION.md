# Documentation — end-to-end runbook

Harmlessness DPO pipeline: **curate the preference dataset → train the model**.
Two commands, in order.

```
DPO-train-dataset-curation/   # dataset curation module (WildJailbreak → preference pairs)
train_dpo.py                  # LoRA DPO trainer (self-contained; consumes the exported dataset)
train.sh                      # one-command training wrapper
requirements.txt              # unified environment
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

Requires a CUDA GPU (vLLM generation + Beaver/WildGuard scoring + DPO training).

## 1. Curate the DPO dataset — one command

```bash
bash DPO-train-dataset-curation/run_all.sh
```

Runs stages 01→06 (prepare prompts → generate Qwen3-4B responses → WildGuard labels →
Beaver cost scoring → filter/report → **export**). By default it curates **ALL** unique
WildJailbreak harmful prompts (`vanilla_harmful` + `adversarial_harmful`, train config);
set `SAMPLE=<int>` for a smaller seeded pilot. Writes a fresh
`DPO-train-dataset-curation/outputs/run_<timestamp>/` containing, among the diagnostics:

- **`dpo_dataset/`** — the handoff: a HF `DatasetDict({train,valid})` with columns
  `prompt` / `chosen` / `rejected`. This is what training consumes.
- **`run_all.log`** — the run's complete console output (all stages, stdout+stderr),
  appended live, so the log survives closed terminals and ships with the run folder.

Useful knobs (env vars, all optional):

```bash
SAMPLE=300 bash DPO-train-dataset-curation/run_all.sh                      # pilot sample instead of ALL prompts
SEED=1 bash DPO-train-dataset-curation/run_all.sh                          # single seed for every stage (default 0; same var train.sh uses)
RUN_OUTPUT_DIR=outputs/my_run bash DPO-train-dataset-curation/run_all.sh   # pin the output folder
VALID_FRACTION=0.1  bash DPO-train-dataset-curation/run_all.sh             # valid split size (seeded random sample of the final pairs)
STRATIFY_VALID=true bash DPO-train-dataset-curation/run_all.sh             # 50/50 vanilla/adversarial valid mix instead of random
EXPORT_FIELD=answer bash DPO-train-dataset-curation/run_all.sh             # export visible answer only (default: full, with CoT)
EXCLUDE_RUNS=outputs/pilot bash DPO-train-dataset-curation/run_all.sh      # skip prompts a previous run already used
```

**Two-phase curation** (pilot for a training smoke test, then a top-up of *new* prompts,
merged into one dataset):

```bash
cd DPO-train-dataset-curation
SAMPLE=4500 RUN_OUTPUT_DIR=outputs/pilot bash run_all.sh
SAMPLE=24000 EXCLUDE_RUNS=outputs/pilot RUN_OUTPUT_DIR=outputs/main bash run_all.sh
TARGET_PAIRS=5000 python merge_runs.py outputs/pilot outputs/main --out outputs/merged
cd .. && bash train.sh DPO-train-dataset-curation/outputs/merged/dpo_dataset
```

`TARGET_PAIRS` (optional) caps the merged pool at an exact size via a seeded random
downsample before the export — the run oversamples prompts for safety (filter survival is
~19% and noisy) and the merge trims the surplus, so the final dataset lands on exactly the
target. Unset = keep every merged pair.

This works from a fresh clone on a new pod: the two files the workflow reads —
`01_prompts_meta.joblib` (per-run consumed prompts, for `EXCLUDE_RUNS`) and
`final_pairs.joblib` (per-run final pairs, for `merge_runs.py`) — are git-tracked via
narrow `.gitignore` re-includes (the other, heavy stage joblibs stay ignored).

Smoke test first (`SAMPLE=8 bash DPO-train-dataset-curation/run_all.sh`) to confirm the
whole chain produces `dpo_dataset/` before a full run. `SAMPLE` is the only smoke-test
knob — `N` and `MAX_TOKENS` are fixed across all runs so runs stay mergeable/comparable.

## 2. Train — one command

```bash
bash train.sh
```

Auto-discovers the most recent `dpo_dataset/` from step 1 and runs LoRA DPO on **Qwen/Qwen3-4B**.
Override the dataset or settings as needed:

```bash
bash train.sh path/to/dpo_dataset         # a specific dataset
EPOCHS=5 bash train.sh                     # more epochs
MODEL=Qwen/Qwen3-4B OUTPUT_DIR=checkpoints/run1 bash train.sh
MAX_LENGTH=8192 bash train.sh              # lower if you hit GPU OOM
```

Output: LoRA adapters under `checkpoints/qwen3-4B/harmless_dpo/` (default). The best checkpoint
by eval reward-accuracy is kept as `checkpoint-best`; others are pruned. Its identity — the winning
step and the exact `<HUB_MODEL_ID>-step-<N>` HF repo it was pushed to — is written to
`OUTPUT_DIR/best_checkpoint.txt` (and mirrored into the W&B run summary). A full log is written to
`logs/train.log`.

**Eval/checkpoint schedule**: fixed step intervals are epoch-count-proof and derived from the
trainer's own total-step count (dataset size × epochs × batch/grad-accum settings), not a fixed
number of steps. Eval + checkpoint fire at:
- step 0 (a pre-training baseline, before any optimizer step)
- every ~20% of total training steps (so ~5 evals spread across the run)
- the final training step, always — even if it doesn't land on a 20%-boundary

`--eval_strategy` / `--save_strategy` are forced to `"no"` inside `train_dpo.py` regardless of
CLI flags, so this schedule is the only thing that triggers eval/save — nothing here sets
`num_train_epochs` early-stop behavior; the run always trains through every epoch.

**Eval metrics** — at every eval point the trainer writes one line to `OUTPUT_DIR/eval_metrics.jsonl`
with `step`, `epoch`, and all DPO eval metrics (`eval_loss`, `eval_rewards/accuracies`,
`eval_rewards/margins`, `eval_logps/*`, …). Plain JSONL for easy plotting of the training curves:
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
| Setup | `pip install -r requirements.txt` + `export HF_TOKEN=...` |
| Curate dataset | `bash DPO-train-dataset-curation/run_all.sh` |
| Train | `bash train.sh` |

## Notes

- **TRL version**: `train_dpo.py` uses the TRL `DPOTrainer` / `DPOConfig` / `TrlParser` API
  (`processing_class`, `get_peft_config`). `requirements.txt` pins `trl==0.22.2` **exactly**
  (not a floor) — all of `train.sh`'s CLI flags were verified against this version, and TRL's
  `DPOConfig` API has changed across versions before. If you deliberately upgrade TRL, run
  `python train_dpo.py --help` first to confirm every flag still parses, then re-pin.
- **Dataset schema contract**: training expects exactly `prompt` / `chosen` / `rejected` string
  columns with `train` / `valid` splits — which is what stage 06 emits. `chosen`/`rejected` retain
  the `<think>…</think>` reasoning (thinking-mode DPO); `train_dpo.py` wraps them as assistant turns
  without stripping the CoT.
