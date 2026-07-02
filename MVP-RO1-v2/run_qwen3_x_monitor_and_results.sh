#!/usr/bin/env bash
#
# run_qwen3_x_monitor_and_results.sh — Qwen3 monitor + results calculation.
# Run this AFTER run_qwen3_x_inference.sh has produced the hinted/unhinted
# outputs. No GPU needed (monitor uses the OpenAI API).
#
# Phases (per dataset):
#   1. Monitor (03) — CoT monitor over both arms; resumes from checkpoint.
#   2. Results calculation:
#        04_compute_gmean2  — per-instance + dataset g-mean² metrics
#        05_analyze_gmean2  — analysis + plots
#        06_export_excel    — Excel export
#   Commits + pushes the monitor outputs and analysis at the end.
#
# Requires OPENAI_API_KEY in the environment for the monitor phase.
#
# Usage:
#   export OPENAI_API_KEY=<your-key>
#   bash run_qwen3_x_monitor_and_results.sh

set -uo pipefail

# Run from the repo root (this script's directory) so scripts/ + config/ resolve.
cd "$(dirname "$0")"

# --- Configuration ----------------------------------------------------------
# Must match the HINT_TYPE used in run_qwen3_x_inference.sh. One of: professor,
# visual_pattern, metadata, wrong_few_shot, grader_hacking, unethical.
HINT_TYPE="professor"
# Models to process — run one after the other. Must match the MODELs used in inference.
MODELS=(qwen3_4b_saferl)
# Active datasets for this run — must match DATASETS in run_qwen3_x_inference.sh
# (mathbench_* and compmath excluded; logieval is commented out in the config
# registry; aqua_rat/logiqa_en were trialled and dropped — see config/mvp_config.yaml).
DATASETS=(mmlu_redux gpqa_diamond wmdp codemmlu lsat_ar lsat_lr)

LOGFILE="run_monitor_and_results_$(date +%Y%m%d_%H%M%S).log"

# --- Preflight: API key ------------------------------------------------------
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY is not set. Run:  export OPENAI_API_KEY=<your-key>"
    exit 1
fi

# --- Preflight: inference outputs must exist ---------------------------------
echo "Checking inference outputs for ${MODELS[*]} ($HINT_TYPE) ..."
missing=0
for model in "${MODELS[@]}"; do
    for ds in "${DATASETS[@]}"; do
        for arm in hinted unhinted; do
            f="outputs-and-results/outputs/${ds}/${model}/${HINT_TYPE}/${arm}.jsonl"
            if [[ -f "$f" ]]; then
                echo "  OK  $f"
            else
                echo "  MISSING  $f"
                missing=$((missing + 1))
            fi
        done
    done
done
if [[ $missing -gt 0 ]]; then
    echo "ERROR: $missing inference file(s) missing — run run_qwen3_1_7b_inference.sh first."
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

# Mirror all output to a timestamped log file as well as the console.
exec > >(tee -a "$LOGFILE") 2>&1

echo "Monitor + results run started: $(date)"
echo "Models:   ${MODELS[*]}"
echo "Hint:     $HINT_TYPE"
echo "Datasets: ${DATASETS[*]}"
echo "Log file: $LOGFILE"

# --- Phase 0: git identity ---------------------------------------------------
echo ""
echo "########## PHASE 0: git identity ##########"
run git config --global user.email "abd@gmail.com"
run git config --global user.name "Abd"

commit_push() {
    local msg="$1"
    run git add -A
    if git diff --cached --quiet; then
        echo "  (nothing to commit for: $msg)"
        SUMMARY+=("OK    | git commit (no changes) | $msg")
        return
    fi
    run git commit -m "$msg"
    run git push
}

# --- Phase 1: monitor --------------------------------------------------------
echo ""
echo "########## PHASE 1: monitor ($HINT_TYPE) ##########"
for model in "${MODELS[@]}"; do
    echo ""
    echo "---------- model: $model ----------"
    for ds in "${DATASETS[@]}"; do
        run python scripts/03_run_monitor.py --model "$model" --dataset "$ds" --hint-type "$HINT_TYPE"
    done
done

# --- Phase 2: results calculation (g-mean² + analysis + export) --------------
echo ""
echo "########## PHASE 2: results calculation ($HINT_TYPE) ##########"
for model in "${MODELS[@]}"; do
    echo ""
    echo "---------- model: $model ----------"
    for ds in "${DATASETS[@]}"; do
        run python scripts/04_compute_gmean2.py --model "$model" --dataset "$ds" --hint-type "$HINT_TYPE"
        run python scripts/05_analyze_gmean2.py --model "$model" --dataset "$ds" --hint-type "$HINT_TYPE"
        run python scripts/06_export_excel.py   --model "$model" --dataset "$ds" --hint-type "$HINT_TYPE"
    done
done

commit_push "monitor + results ($HINT_TYPE: gmean2 / analysis / excel) — ${MODELS[*]}"

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
