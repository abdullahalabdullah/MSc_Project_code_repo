#!/usr/bin/env bash
# Sham-control DPO training (anti-sycophancy pairs with the preference signal destroyed
# — spec Stage I). Same wrapper pattern as train_antisyc.sh with sham-specific names.
#
# Usage:
#   bash train_antisyc_sham.sh anti-syc-DPO-train-dataset-curation/outputs/full_run/dpo_dataset_sham
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -lt 1 ]; then
  echo "Usage: bash train_antisyc_sham.sh <path/to/dpo_dataset_sham>"
  exit 1
fi

export OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/qwen3-4B/anti_syc_dpo_sham}"
export HUB_MODEL_ID="${HUB_MODEL_ID:-er-abd/anti-syc-dpo-sham-qwen3-4B}"
export WANDB_PROJECT="${WANDB_PROJECT:-anti-syc-dpo-qwen3-4B}"
export RUN_NAME="${RUN_NAME:-anti_syc_dpo_sham_seed-${SEED:-0}}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"

bash train.sh "$1"
