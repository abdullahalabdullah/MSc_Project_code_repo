#!/usr/bin/env bash
#
# run_all.sh — full RO1 pipeline over all datasets for both reasoning models.
#
# Order:
#   1. Prepare every dataset index (model-independent, idempotent).
#   2. Inference: DeepSeek-R1-Distill-Qwen-1.5B first, then Qwen3-4B.
#   3. Monitor (needs OPENAI_API_KEY) for each model, after all inference is done.
#   4. git add / commit / push.
#
# Fail-soft: if any single command errors, it is logged as FAIL and the script
# moves on to the next one (no `set -e`). A summary is printed at the end.
#
# Usage:
#   export OPENAI_API_KEY=sk-...        # needed for step 3 (monitor)
#   # HF auth is handled in-code via .hf_token / HF_TOKEN (see src/dataset.py)
#   bash run_all.sh

set -uo pipefail

# Run from the MVP-RO1 root (this script's directory) so scripts/ + config/ resolve.
cd "$(dirname "$0")"

# --- Configuration ----------------------------------------------------------
# Datasets to process. Edit this list to add/remove tasks (e.g. drop "mmlu" if
# its results already exist — re-runs resume from checkpoints and are cheap).
DATASETS=(mmlu gpqa gsm_mc math_mc agieval wmdp)

# Config model names (the `name:` keys in config/mvp_config.yaml) that map to:
#   deepseek_r1_distill_qwen_1_5b -> deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
#   qwen3_4b                      -> Qwen/Qwen3-4B
DEEPSEEK_MODEL="deepseek_r1_distill_qwen_1_5b"
QWEN_MODEL="qwen3_4b"

LOGFILE="run_all_$(date +%Y%m%d_%H%M%S).log"

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

echo "RO1 full run started: $(date)"
echo "Datasets: ${DATASETS[*]}"
echo "Models:   $DEEPSEEK_MODEL (DeepSeek) then $QWEN_MODEL (Qwen3)"
echo "Log file: $LOGFILE"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "WARNING: OPENAI_API_KEY is not set — step 3 (monitor) will fail for every"
    echo "         dataset (and be skipped). Export it before running for monitor results."
fi

# --- Phase 1: prepare dataset indexes (model-independent) --------------------
echo ""
echo "########## PHASE 1: prepare dataset indexes ##########"
for ds in "${DATASETS[@]}"; do
    run python scripts/01_prepare_dataset.py --dataset "$ds"
done

# --- Phase 2: inference (DeepSeek first, then Qwen3) -------------------------
echo ""
echo "########## PHASE 2a: inference — $DEEPSEEK_MODEL ##########"
for ds in "${DATASETS[@]}"; do
    run python scripts/02_run_inference_deepseek.py --dataset "$ds" --model "$DEEPSEEK_MODEL"
done

echo ""
echo "########## PHASE 2b: inference — $QWEN_MODEL ##########"
for ds in "${DATASETS[@]}"; do
    run python scripts/02_run_inference_qwen.py --dataset "$ds" --model "$QWEN_MODEL"
done

# --- Phase 3: monitor (after all inference, both models) ---------------------
echo ""
echo "########## PHASE 3: monitor (both models) ##########"
for model in "$DEEPSEEK_MODEL" "$QWEN_MODEL"; do
    for ds in "${DATASETS[@]}"; do
        run python scripts/03_run_monitor.py --dataset "$ds" --model "$model"
    done
done

# --- Phase 4: commit & push --------------------------------------------------
echo ""
echo "########## PHASE 4: git add / commit / push ##########"
run git add -A
run git commit -m "all results"
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
