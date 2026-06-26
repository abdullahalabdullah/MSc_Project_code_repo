#!/usr/bin/env bash
#
# run_monitor_all.sh — run the CoT monitor over EVERY (dataset, model) pair
# found under outputs/.
#
# Auto-discovers pairs from the directory layout:
#   outputs/{dataset}/{model}/...
# and runs:  python scripts/03_run_monitor.py --model {model} --dataset {dataset}
#
# Writes monitor_outputs/{dataset}/{model}/monitor_results.jsonl per pair.
# The monitor resumes from its checkpoint, so re-running after an interruption
# skips already-completed calls.
#
# Requires OPENAI_API_KEY in the environment.
#
# Fail-soft: a failure on one pair is logged and the script continues to the
# next. A summary is printed at the end.
#
# Usage:
#   export OPENAI_API_KEY=<your-key>
#   bash run_monitor_all.sh

set -uo pipefail

# Run from the MVP-RO1 root (this script's directory) so scripts/ + config/ resolve.
cd "$(dirname "$0")"

# --- Configuration ----------------------------------------------------------
OUTPUTS_DIR="outputs"
MODEL="qwen3_4b_nothink"
LOGFILE="run_monitor_all_$(date +%Y%m%d_%H%M%S).log"

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

echo "Monitor run ($MODEL) started: $(date)"
echo "Outputs dir: $OUTPUTS_DIR"
echo "Model:       $MODEL"
echo "Log file:    $LOGFILE"

# --- Monitor every dataset that has outputs for $MODEL -----------------------
for ds_dir in "$OUTPUTS_DIR"/*/; do
    ds="$(basename "$ds_dir")"
    # Only run where this model actually has outputs.
    [[ -d "$ds_dir$MODEL" ]] || continue
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
