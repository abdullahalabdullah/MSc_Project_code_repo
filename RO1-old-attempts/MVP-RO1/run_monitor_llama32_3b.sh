#!/usr/bin/env bash
#
# run_monitor_llama32_3b.sh — run the CoT monitor over all datasets for
# meta-llama/Llama-3.2-3B-Instruct (model name: llama32_3b_instruct).
#
# Reads outputs/{ds}/llama32_3b_instruct/{hinted,unhinted}.jsonl and writes
# monitor_outputs/{ds}/llama32_3b_instruct/monitor_results.jsonl per dataset.
# The monitor resumes from its checkpoint, so re-running this script after an
# interruption (e.g. a dropped connection) skips already-completed calls.
#
# Requires OPENAI_API_KEY in the environment.
#
# Fail-soft: a failure on one dataset is logged and the script continues to the
# next. A summary is printed at the end.
#
# Usage:
#   export OPENAI_API_KEY=<your-key>
#   bash run_monitor_llama32_3b.sh

set -uo pipefail

# Run from the MVP-RO1 root (this script's directory) so scripts/ + config/ resolve.
cd "$(dirname "$0")"

# --- Configuration ----------------------------------------------------------
DATASETS=(mmlu gpqa gsm_mc math_mc agieval wmdp)
MODEL="llama32_3b_instruct"

LOGFILE="run_monitor_llama32_3b_$(date +%Y%m%d_%H%M%S).log"

# --- Preflight: API key ------------------------------------------------------
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY is not set. Run:  export OPENAI_API_KEY=<your-key>"
    exit 1
fi

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

echo "Monitor run (llama32_3b_instruct) started: $(date)"
echo "Datasets: ${DATASETS[*]}"
echo "Model:    $MODEL"
echo "Log file: $LOGFILE"

# --- Monitor over every dataset ----------------------------------------------
for ds in "${DATASETS[@]}"; do
    run python scripts/03_run_monitor.py --model "$MODEL" --dataset "$ds"
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
