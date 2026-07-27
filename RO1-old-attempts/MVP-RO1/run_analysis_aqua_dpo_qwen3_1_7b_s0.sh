#!/usr/bin/env bash
#
# run_analysis_aqua_dpo_qwen3_1_7b_s0.sh — compute g-mean², analyze, and
# export Excel for aqua-DPO Qwen3-1.7B seed-0 across all datasets.
#
# Usage:
#   bash run_analysis_aqua_dpo_qwen3_1_7b_s0.sh

set -uo pipefail

cd "$(dirname "$0")"

DATASETS=(mmlu gpqa wmdp agieval)
MODEL="aqua_dpo_qwen3_1_7b_s0"

LOGFILE="run_analysis_aqua_dpo_qwen3_1_7b_s0_$(date +%Y%m%d_%H%M%S).log"

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

echo "Analysis run ($MODEL) started: $(date)"
echo "Datasets: ${DATASETS[*]}"

for ds in "${DATASETS[@]}"; do
    run python scripts/04_compute_gmean2.py --model "$MODEL" --dataset "$ds"
    run python scripts/05_analyze_gmean2.py --model "$MODEL" --dataset "$ds"
    run python scripts/06_export_excel.py   --model "$MODEL" --dataset "$ds"
done

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
