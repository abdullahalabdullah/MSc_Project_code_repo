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

Provide a Hugging Face token (WildJailbreak and WildGuard are gated) via the environment or a
`.env` file — never hard-code it:

```bash
export HF_TOKEN=hf_xxx          # or put HF_TOKEN=... in .env
# optional: export HF_MODELS_CACHE_DIR=/path/to/cache
```

Requires a CUDA GPU (vLLM generation + Beaver/WildGuard scoring + DPO training).

## 1. Curate the DPO dataset — one command

```bash
bash DPO-train-dataset-curation/run_all.sh
```

Runs stages 01→06 (prepare prompts → generate Qwen3-4B responses → WildGuard labels →
Beaver cost scoring → filter/report → **export**). Writes a fresh
`DPO-train-dataset-curation/outputs/run_<timestamp>/` containing, among the diagnostics:

- **`dpo_dataset/`** — the handoff: a HF `DatasetDict({train,valid})` with columns
  `prompt` / `chosen` / `rejected`. This is what training consumes.

Useful knobs (env vars, all optional):

```bash
RUN_OUTPUT_DIR=outputs/my_run bash DPO-train-dataset-curation/run_all.sh   # pin the output folder
VALID_FRACTION=0.1  bash DPO-train-dataset-curation/run_all.sh             # valid split size
EXPORT_FIELD=answer bash DPO-train-dataset-curation/run_all.sh             # export visible answer only (default: full, with CoT)
```

Smoke test first (edit `DPO-train-dataset-curation/config.py`: `SAMPLE=8, N=3, MAX_TOKENS=1024`)
to confirm the whole chain produces `dpo_dataset/` before a full run.

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
by eval reward-accuracy is kept as `checkpoint-best`; others are pruned. A full log is written to
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

For a long run, use tmux so an SSH drop doesn't kill it:

```bash
tmux new -s train
bash train.sh
# detach: Ctrl-b then d   |   reattach: tmux attach -t train
```

### Optional: push adapters to the Hugging Face Hub

Hub pushing is off by default. Set `HUB_MODEL_ID` and `train.sh` forwards `--push_to_hub
--hub_model_id`:

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
