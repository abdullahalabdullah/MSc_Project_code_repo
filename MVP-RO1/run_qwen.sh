#!/usr/bin/env bash
#
# run_qwen.sh — full RO1 pipeline over all datasets for Qwen3-4B.
#
# Order:
#   1. Prepare every dataset index (model-independent, idempotent).
#   2. Inference: Qwen3-4B.
#   3. git add / commit / push.
#
# Fail-soft: if any single command errors, it is logged as FAIL and the script
# moves on to the next one (no `set -e`). A summary is printed at the end.
#
# Usage:
#   # HF auth is handled in-code via .hf_token / HF_TOKEN (see src/dataset.py)
#   bash run_qwen.sh

set -uo pipefail

# Run from the MVP-RO1 root (this script's directory) so scripts/ + config/ resolve.
cd "$(dirname "$0")"

# --- Configuration ----------------------------------------------------------
# Datasets to process. Edit this list to add/remove tasks (e.g. drop "mmlu" if
# its results already exist — re-runs resume from checkpoints and are cheap).
DATASETS=(gpqa gsm_mc math_mc agieval wmdp)

# Config model name (the `name:` key in config/mvp_config.yaml) that maps to:
#   qwen3_4b -> Qwen/Qwen3-4B
QWEN_MODEL="qwen3_4b"

LOGFILE="run_qwen_$(date +%Y%m%d_%H%M%S).log"

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

echo "RO1 Qwen3 run started: $(date)"
echo "Datasets: ${DATASETS[*]}"
echo "Model:    $QWEN_MODEL (Qwen3)"
echo "Log file: $LOGFILE"

# --- Phase 1: prepare dataset indexes (model-independent) --------------------
echo ""
echo "########## PHASE 1: prepare dataset indexes ##########"
for ds in "${DATASETS[@]}"; do
    run python scripts/01_prepare_dataset.py --dataset "$ds"
done

# --- Phase 2: inference ------------------------------------------------------
echo ""
echo "########## PHASE 2: inference — $QWEN_MODEL ##########"
for ds in "${DATASETS[@]}"; do
    run python scripts/02_run_inference_qwen.py --dataset "$ds" --model "$QWEN_MODEL"
done

# --- Phase 3: commit & push --------------------------------------------------
echo ""
echo "########## PHASE 3: git add / commit / push ##########"
run git add -A
run git commit -m "qwen results"
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
