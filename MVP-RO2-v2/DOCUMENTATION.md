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

For a long run, use tmux so an SSH drop doesn't kill it:

```bash
tmux new -s train
bash train.sh
# detach: Ctrl-b then d   |   reattach: tmux attach -t train
```

### Optional: push adapters to the Hugging Face Hub

Hub pushing is off by default. To push each saved checkpoint to its own repo, pass through the
trainer flags (train.sh forwards nothing extra, so call the trainer directly):

```bash
python train_dpo.py --dataset_path <.../dpo_dataset> --model_name_or_path Qwen/Qwen3-4B \
  --output_dir checkpoints/run1 --use_peft --lora_r 16 --lora_alpha 32 \
  --push_to_hub --hub_model_id <your-namespace>/harmless-dpo-qwen3-4B
```

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
