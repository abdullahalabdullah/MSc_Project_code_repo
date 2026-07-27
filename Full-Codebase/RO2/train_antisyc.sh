#!/usr/bin/env bash
# Anti-sycophancy DPO training — thin wrapper around train.sh that pins the anti-syc
# output dir, HF hub repo, and W&B names, so a run can never accidentally push into the
# harmlessness repos (train.sh's defaults). Training code itself is unchanged.
#
# Usage:
#   bash train_antisyc.sh anti-syc-DPO-train-dataset-curation/outputs/full_run/dpo_dataset
#   EPOCHS=3 SEED=0 bash train_antisyc.sh <path/to/dpo_dataset>       # same env knobs as train.sh
#
# MAX_PROMPT_LENGTH: anti-syc prompts are a short opinionated trivia question (<1024
# tokens), unlike WildJailbreak's long jailbreak wrappers. MAX_LENGTH stays at
# train.sh's 12288 default — completions can theoretically reach 16384 tokens, but
# trivia CoT is far shorter in practice; check the curation qa_report.md length stats
# and raise MAX_LENGTH (env) only if the >12k share is non-negligible.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -lt 1 ]; then
  echo "Usage: bash train_antisyc.sh <path/to/dpo_dataset>"
  echo "  (the curation export: anti-syc-DPO-train-dataset-curation/outputs/<run>/dpo_dataset)"
  exit 1
fi

export OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/qwen3-4B/anti_syc_dpo}"
export HUB_MODEL_ID="${HUB_MODEL_ID:-er-abd/anti-syc-dpo-qwen3-4B}"
export WANDB_PROJECT="${WANDB_PROJECT:-anti-syc-dpo-qwen3-4B}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"

# Run tag distinguishes a pilot from the full run in log/W&B names. It is derived from the
# curation output dir feeding this run (…/outputs/<run_tag>/dpo_dataset), so a pilot export
# (…/outputs/pilot/dpo_dataset) and the full export (…/outputs/full_run/dpo_dataset) never
# share a log. Override with RUN_TAG=<name> if your dataset path is non-standard.
RUN_TAG="${RUN_TAG:-$(basename "$(dirname "$1")")}"
export RUN_NAME="${RUN_NAME:-anti_syc_dpo_${RUN_TAG}_seed-${SEED:-0}}"

# Unique, non-colliding log per experiment: anti-syc, its method, and the run tag — never the
# generic logs/train_dpo.log that the harmlessness runs write.
export LOG_FILE="${LOG_FILE:-logs/anti_syc_train_${METHOD:-dpo}_${RUN_TAG}.log}"

bash train.sh "$1"
