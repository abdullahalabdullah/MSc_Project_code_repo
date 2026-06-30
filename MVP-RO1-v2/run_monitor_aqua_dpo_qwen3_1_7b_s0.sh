#!/usr/bin/env bash
#
# run_monitor_aqua_dpo_qwen3_1_7b_s0.sh — run the CoT monitor over all datasets
# for the aqua-DPO Qwen3-1.7B seed-0 model (aqua_dpo_qwen3_1_7b_s0).
#
# Reads outputs/{ds}/aqua_dpo_qwen3_1_7b_s0/{hinted,unhinted}.jsonl and writes
# monitor_outputs/{ds}/aqua_dpo_qwen3_1_7b_s0/monitor_results.jsonl per dataset.
# Resumes from checkpoint on re-run — safe to restart after interruption.
#
# Requires OPENAI_API_KEY in the environment.
#
# Usage:
#   export OPENAI_API_KEY=<your-key>
#   bash run_monitor_aqua_dpo_qwen3_1_7b_s0.sh

set -uo pipefail

cd "$(dirname "$0")"

# --- Configuration ----------------------------------------------------------
DATASETS=(mmlu gpqa wmdp agieval)
MODEL="aqua_dpo_qwen3_1_7b_s0"

LOGFILE="run_monitor_aqua_dpo_qwen3_1_7b_s0_$(date +%Y%m%d_%H%M%S).log"

# --- Preflight: API key ------------------------------------------------------
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY is not set. Run:  export OPENAI_API_KEY=<your-key>"
    exit 1
fi

# --- Preflight: check inference outputs exist --------------------------------
echo "Checking inference outputs..."
missing=0
for ds in "${DATASETS[@]}"; do
    for arm in hinted unhinted; do
        f="outputs/${ds}/${MODEL}/${arm}.jsonl"
        if [[ -f "$f" ]]; then
            echo "  OK  $f"
        else
            echo "  MISSING  $f"
            missing=$((missing + 1))
        fi
    done
done
if [[ $missing -gt 0 ]]; then
    echo "ERROR: $missing inference file(s) missing. Run inference first."
    exit 1
fi

# --- Fail-soft command runner ------------------------------------------------
SUMMARY=()

run() {
    echo ""
    echo "=================================================================="
    echo ">>> $*"
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

exec > >(tee -a "$LOGFILE") 2>&1

echo "Monitor run ($MODEL) started: $(date)"
echo "Datasets: ${DATASETS[*]}"
echo "Model:    $MODEL"
echo "Log file: $LOGFILE"

# --- Monitor over every dataset ----------------------------------------------
for ds in "${DATASETS[@]}"; do
    run python scripts/03_run_monitor.py --model "$MODEL" --dataset "$ds"
done

# --- Metrics + analysis + export --------------------------------------------
for ds in "${DATASETS[@]}"; do
    run python scripts/04_compute_gmean2.py  --model "$MODEL" --dataset "$ds"
    run python scripts/05_analyze_gmean2.py  --model "$MODEL" --dataset "$ds"
    run python scripts/06_export_excel.py    --model "$MODEL" --dataset "$ds"
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
