#!/usr/bin/env bash
#
# run_deepseek.sh — full RO1 pipeline over all datasets for DeepSeek-R1-Distill-Qwen-1.5B.
#
# Order:
#   1. Prepare every dataset index (model-independent, idempotent).
#   2. Inference: DeepSeek-R1-Distill-Qwen-1.5B.
#   3. git add / commit / push.
#
# Fail-soft: if any single command errors, it is logged as FAIL and the script
# moves on to the next one (no `set -e`). A summary is printed at the end.
#
# Usage:
#   # HF auth is handled in-code via .hf_token / HF_TOKEN (see src/dataset.py)
#   bash run_deepseek.sh

set -uo pipefail

# Run from the MVP-RO1 root (this script's directory) so scripts/ + config/ resolve.
cd "$(dirname "$0")"

# --- Configuration ----------------------------------------------------------
# Datasets to process. Edit this list to add/remove tasks (e.g. drop "mmlu" if
# its results already exist — re-runs resume from checkpoints and are cheap).
DATASETS=(gpqa gsm_mc math_mc agieval wmdp)

# Config model name (the `name:` key in config/mvp_config.yaml) that maps to:
#   deepseek_r1_distill_qwen_1_5b -> deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
DEEPSEEK_MODEL="deepseek_r1_distill_qwen_1_5b"

LOGFILE="run_deepseek_$(date +%Y%m%d_%H%M%S).log"

# --- Fail-soft command runner ------------------------------------------------
SUMMARY=()

run() {
    echo ""
    echo "=================================================================="
    echo ">>> $* "
    echo "=================================================================="
    if "$@"; then
        echo "<<< OK: $*"
        SUMMARY+=("OK    | $*")
    else
        local rc=$?
        echo "<<< FAILED (exit $rc) — skipping and continuing: $*"
        SUMMARY+=("FAIL  | $*")
    fi
}

# Mirror all output to a timestamped log file as well as the console.
exec > >(tee -a "$LOGFILE") 2>&1

echo "RO1 DeepSeek run started: $(date)"
echo "Datasets: ${DATASETS[*]}"
echo "Model:    $DEEPSEEK_MODEL (DeepSeek)"
echo "Log file: $LOGFILE"

# --- Phase 1: prepare dataset indexes (model-independent) --------------------
echo ""
echo "########## PHASE 1: prepare dataset indexes ##########"
for ds in "${DATASETS[@]}"; do
    run python scripts/01_prepare_dataset.py --dataset "$ds"
done

# --- Phase 2: inference ------------------------------------------------------
echo ""
echo "########## PHASE 2: inference — $DEEPSEEK_MODEL ##########"
for ds in "${DATASETS[@]}"; do
    run python scripts/02_run_inference_deepseek.py --dataset "$ds" --model "$DEEPSEEK_MODEL"
done

# --- Phase 3: commit & push --------------------------------------------------
echo ""
echo "########## PHASE 3: git add / commit / push ##########"
run git add -A
run git commit -m "deepseek results"
run git push

# --- Summary -----------------------------------------------------------------
echo ""
echo "=================================================================="
echo "RUN SUMMARY ($(date))"
echo "=================================================================="
fail_count=0
for line in "${SUMMARY[@]}"; do
    echo "$line"
    [[ "$line" == FAIL* ]] && fail_count=$((fail_count + 1))
done
echo "------------------------------------------------------------------"
echo "Total commands: ${#SUMMARY[@]}  |  Failed: $fail_count"
echo "Full log: $LOGFILE"
