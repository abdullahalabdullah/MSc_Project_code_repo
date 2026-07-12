#!/usr/bin/env bash
#
# run_antisyc_results.sh — anti-sycophancy API + analysis phase for one model (NO GPU).
#
# Grades every response with gpt-5-mini (resumable; requires OPENAI_API_KEY), computes
# M1/M2/M3, and renders the single-model summary. Run after run_antisyc_inference.sh.
#
# Usage:
#   OPENAI_API_KEY=<key> bash anti-sycophancy-eval/run_antisyc_results.sh <model_id>

set -uo pipefail
cd "$(dirname "$0")/.."

USAGE="usage: bash anti-sycophancy-eval/run_antisyc_results.sh <model_id>"
MODEL_ID="${1:?$USAGE}"
MODEL_NAME="${MODEL_ID//\//_}"

SCRIPTS="anti-sycophancy-eval/scripts"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY is not set (the grader needs it)."
    exit 1
fi
RESP_DIR="outputs-and-results/anti-sycophancy-results/responses/${MODEL_NAME}"
if [[ ! -f "$RESP_DIR/id_responses.jsonl" || ! -f "$RESP_DIR/ood_responses.jsonl" ]]; then
    echo "ERROR: responses not found under $RESP_DIR — run run_antisyc_inference.sh first."
    exit 1
fi

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

mkdir -p logs
LOGFILE="logs/antisyc_results_${MODEL_NAME}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "Anti-sycophancy results started: $(date)"
echo "Model:    $MODEL_ID"
echo "Log file: $LOGFILE"

echo ""
echo "########## PHASE 0: git identity ##########"
run git config --global user.email "abd@gmail.com"
run git config --global user.name "Abd"

echo ""
echo "########## PHASE 1: gpt-5-mini grading (API) ##########"
run python "$SCRIPTS/03_run_grader.py" --model "$MODEL_ID"
commit_push "anti-sycophancy: $MODEL_NAME grades"

echo ""
echo "########## PHASE 2: metrics ##########"
run python "$SCRIPTS/04_compute_metrics.py" --model "$MODEL_ID"

echo ""
echo "########## PHASE 3: markdown report ##########"
# ONE separable step: delete this line (plus scripts/06_generate_reports.py and
# src/reporting.py) to strip all .md report generation for a public release.
run python "$SCRIPTS/06_generate_reports.py" --model "$MODEL_ID"

echo ""
echo "########## FINAL PUSH ##########"
commit_push "anti-sycophancy: $MODEL_NAME results complete"

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
echo "Paired report:  bash anti-sycophancy-eval/run_antisyc_compare.sh <base_id> $MODEL_ID"
