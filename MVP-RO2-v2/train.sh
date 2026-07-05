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
#                                                   # push every checkpoint (step 0, every ~20%
#                                                   # of training, and the final step) to its own
#                                                   # HF repo "<HUB_MODEL_ID>-step-<N>"
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
# Optional: set to push every checkpoint to its own private HF repo "<HUB_MODEL_ID>-step-<N>".
HUB_MODEL_ID="${HUB_MODEL_ID:-}"
# Per-eval generation on the valid prompts (thinking-mode sampling) → eval_generations/step-*.jsonl.gz.
# On by default; set EVAL_GENERATE=False to skip it if the extra decode time per eval is too costly.
EVAL_GENERATE="${EVAL_GENERATE:-True}"
EVAL_GEN_MAX_NEW_TOKENS="${EVAL_GEN_MAX_NEW_TOKENS:-8192}"  # lower to speed up eval (may truncate CoT)
EVAL_GEN_BATCH_SIZE="${EVAL_GEN_BATCH_SIZE:-8}"

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

# Hub push is opt-in: only forwarded when HUB_MODEL_ID is set. train_dpo.py pushes every
# checkpoint it writes (step 0, every ~20% of training, and the final step) when these
# flags are present — none are skipped.
HUB_ARGS=()
if [ -n "${HUB_MODEL_ID}" ]; then
  echo "== Hub:     pushing every checkpoint to ${HUB_MODEL_ID}-step-<N>"
  HUB_ARGS=(--push_to_hub --hub_model_id "${HUB_MODEL_ID}")
fi

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
    --save_only_model \
    --no_remove_unused_columns \
    --beta 0.1 \
    --seed "${SEED}" \
    --report_to none \
    --use_peft \
    --lora_alpha 32 \
    --lora_r 16 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --eval_generate "${EVAL_GENERATE}" \
    --eval_gen_max_new_tokens "${EVAL_GEN_MAX_NEW_TOKENS}" \
    --eval_gen_batch_size "${EVAL_GEN_BATCH_SIZE}" \
    "${HUB_ARGS[@]+"${HUB_ARGS[@]}"}" \
    2>&1 | tee logs/train.log

echo "== Done. Best adapter: ${OUTPUT_DIR}/checkpoint-best =="
echo "   Load it on top of the base model for inference"
echo "   (transformers PeftModel, or vLLM with --enable-lora) — base: ${MODEL}"
echo "== Eval artifacts: ${OUTPUT_DIR}/eval_metrics.jsonl + ${OUTPUT_DIR}/eval_generations/step-*.jsonl.gz =="
