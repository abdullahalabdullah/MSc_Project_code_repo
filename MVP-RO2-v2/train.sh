#!/usr/bin/env bash
#
# One-command LoRA DPO training on the harmlessness preference pairs.
#
# It trains on the dataset exported by the curation module (stage 06). By default it
# auto-discovers the MOST RECENT export under DPO-train-dataset-curation/outputs/*/dpo_dataset;
# pass an explicit path as the first argument to override.
#
# Usage:
#   bash train.sh                                  # newest exported dataset
#   bash train.sh path/to/dpo_dataset              # a specific dataset
#   EPOCHS=5 MODEL=Qwen/Qwen3-4B bash train.sh     # override via env vars
#   HUB_MODEL_ID=<namespace>/harmless-dpo-qwen3-4B bash train.sh
#                                                   # override the HF repo prefix. Pushing is
#                                                   # ALWAYS ON: every checkpoint (step 0, every
#                                                   # ~20% of training, and the final step) goes
#                                                   # to its own HF repo "<HUB_MODEL_ID>-step-<N>"
#   WANDB_PROJECT=my-proj RUN_NAME=exp1 bash train.sh   # W&B logging (run `wandb login` first)
#   REPORT_TO=none bash train.sh                        # disable W&B
#
# Run inside tmux for long jobs:  tmux new -s train ;  bash train.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Config (override via env) ───────────────────────────────────────────────
MODEL="${MODEL:-Qwen/Qwen3-4B}"
EPOCHS="${EPOCHS:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/qwen3-4B/harmless_dpo}"
SEED="${SEED:-0}"
# max_length must fit prompt + full (CoT) completion. The module generates up to 8192
# completion tokens with a ~4096 prompt budget; lower this if you hit GPU OOM.
MAX_LENGTH="${MAX_LENGTH:-12288}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
# Hub pushing is ALWAYS ON: every checkpoint is pushed to its own private HF repo
# "<HUB_MODEL_ID>-step-<N>" under the er-abd account (token ships in the committed .env).
HUB_MODEL_ID="${HUB_MODEL_ID:-er-abd/harmless-dpo-qwen3-4B}"
# Experiment tracking. Defaults to Weights & Biases (run `wandb login` once first, or set
# WANDB_API_KEY). Set REPORT_TO=none to disable. WANDB_PROJECT / RUN_NAME name the run.
REPORT_TO="${REPORT_TO:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-harmless-dpo-qwen3-4B}"
RUN_NAME="${RUN_NAME:-harmless_dpo_$(basename "${OUTPUT_DIR}")_seed-${SEED}}"
export WANDB_PROJECT

# ── Locate the exported DPO dataset ─────────────────────────────────────────
DATASET_PATH="${1:-$(ls -dt DPO-train-dataset-curation/outputs/*/dpo_dataset 2>/dev/null | head -1 || true)}"
if [ -z "${DATASET_PATH}" ] || [ ! -d "${DATASET_PATH}" ]; then
  echo "ERROR: no exported dpo_dataset found."
  echo "  Run the curation pipeline first:  bash DPO-train-dataset-curation/run_all.sh"
  echo "  or pass an explicit path:          bash train.sh path/to/dpo_dataset"
  exit 1
fi

mkdir -p logs
echo "== Model:   ${MODEL}"
echo "== Dataset: ${DATASET_PATH}"
echo "== Output:  ${OUTPUT_DIR}  (epochs=${EPOCHS}, seed=${SEED})"

# ── Preflight: pushing is mandatory, so fail here (before any GPU time) if the HF
# credentials are missing or read-only ─────────────────────────────────────────────
echo "== Preflight: checking Hugging Face credentials (checkpoint push is always on) =="
python - <<'PY'
import os, sys
try:
    # Explicit path: parameterless load_dotenv() inspects the caller's file, which does
    # not exist for stdin scripts (heredoc) and crashes. cwd is the repo root here.
    from dotenv import load_dotenv; load_dotenv(".env")
except ImportError:
    pass
try:
    from huggingface_hub import HfApi
    info = HfApi(token=os.environ.get("HF_TOKEN") or None).whoami()
except Exception as e:
    sys.exit(f"ERROR: no valid HF credentials — set HF_TOKEN (or `huggingface-cli login`). ({e})")
role = ((info.get("auth") or {}).get("accessToken") or {}).get("role")
if role == "read":
    sys.exit("ERROR: HF token is READ-only, but every checkpoint is pushed to the Hub.\n"
             "       Create a WRITE token at https://huggingface.co/settings/tokens")
if role == "fineGrained":
    print("NOTE: fine-grained HF token — ensure it has write access to repos in your "
          "namespace (not automatically verifiable).")
print(f"HF preflight OK: user={info.get('name')}, token role={role or 'unknown'}")
PY

# Hub push is ALWAYS enabled: train_dpo.py pushes every checkpoint it writes (step 0,
# every ~20% of training, and the final step) to "<HUB_MODEL_ID>-step-<N>" — none skipped.
if [ -z "${HUB_MODEL_ID}" ]; then
  echo "ERROR: HUB_MODEL_ID is empty — checkpoint pushing is mandatory."
  echo "  Leave it unset for the default (er-abd/harmless-dpo-qwen3-4B),"
  echo "  or set HUB_MODEL_ID=<namespace>/<name>."
  exit 1
fi
echo "== Hub:     pushing every checkpoint to ${HUB_MODEL_ID}-step-<N>"
HUB_ARGS=(--push_to_hub --hub_model_id "${HUB_MODEL_ID}")

python -u train_dpo.py \
    --dataset_path "${DATASET_PATH}" \
    --model_name_or_path "${MODEL}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs "${EPOCHS}" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --per_device_eval_batch_size 1 \
    --gradient_checkpointing \
    --optim adamw_torch \
    --learning_rate 5.0e-6 \
    --weight_decay 0.01 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --max_length "${MAX_LENGTH}" \
    --max_prompt_length "${MAX_PROMPT_LENGTH}" \
    --logging_steps 10 \
    --bf16 True \
    --tf32 True \
    --save_only_model \
    --no_remove_unused_columns \
    --beta 0.1 \
    --seed "${SEED}" \
    --report_to "${REPORT_TO}" \
    --run_name "${RUN_NAME}" \
    --use_peft \
    --lora_alpha 32 \
    --lora_r 16 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    "${HUB_ARGS[@]+"${HUB_ARGS[@]}"}" \
    2>&1 | tee logs/train.log

echo "== Done. Best adapter: ${OUTPUT_DIR}/checkpoint-best =="
echo "   Load it on top of the base model for inference"
echo "   (transformers PeftModel, or vLLM with --enable-lora) — base: ${MODEL}"
echo "== Eval metrics (per-eval DPO loss/reward curves): ${OUTPUT_DIR}/eval_metrics.jsonl =="
echo "== Best checkpoint id (incl. HF repo if pushed):    ${OUTPUT_DIR}/best_checkpoint.txt =="
