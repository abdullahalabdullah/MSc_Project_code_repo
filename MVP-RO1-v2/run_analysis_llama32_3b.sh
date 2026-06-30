#!/usr/bin/env bash
#
# run_analysis_llama32_3b.sh — post-monitor analysis pipeline (steps 04-07) for
# meta-llama/Llama-3.2-3B-Instruct (model name: llama32_3b_instruct), over all datasets.
#
# Per dataset, in order:
#   04_compute_gmean2.py    — g-mean^2 metric from monitor_outputs/
#   05_analyze_gmean2.py    — analysis on top of the metric
#   07_estimate_accuracy.py — accuracy estimate (RE-SAMPLES via vLLM — needs GPU)
#   06_export_excel.py      — Excel summary export
#
# NOTE: step 07 loads the model and generates fresh samples, so run this on the
# GPU/vLLM box. To skip it (pure CPU run), set RUN_ACCURACY=0.
#
# Fail-soft: a failure on one command is logged and the script continues.
#
# Usage:
#   bash run_analysis_llama32_3b.sh
#   RUN_ACCURACY=0 bash run_analysis_llama32_3b.sh   # skip step 07

set -uo pipefail

# Run from the MVP-RO1 root (this script's directory) so scripts/ + config/ resolve.
cd "$(dirname "$0")"

# --- Configuration ----------------------------------------------------------
DATASETS=(mmlu gpqa gsm_mc math_mc agieval wmdp)
MODEL="llama32_3b_instruct"
RUN_ACCURACY="${RUN_ACCURACY:-1}"   # set to 0 to skip step 07 (the GPU re-sampling step)

LOGFILE="run_analysis_llama32_3b_$(date +%Y%m%d_%H%M%S).log"

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

echo "Analysis run (llama32_3b_instruct) started: $(date)"
echo "Datasets:     ${DATASETS[*]}"
echo "Model:        $MODEL"
echo "Run accuracy: $RUN_ACCURACY (step 07)"
echo "Log file:     $LOGFILE"

# --- Per-dataset analysis ----------------------------------------------------
for ds in "${DATASETS[@]}"; do
    echo ""
    echo "########## DATASET: $ds ##########"
    run python scripts/04_compute_gmean2.py --model "$MODEL" --dataset "$ds"
    run python scripts/05_analyze_gmean2.py --model "$MODEL" --dataset "$ds"
    if [[ "$RUN_ACCURACY" == "1" ]]; then
        run python scripts/07_estimate_accuracy.py --model "$MODEL" --dataset "$ds"
    fi
    run python scripts/06_export_excel.py --model "$MODEL" --dataset "$ds"
done

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
