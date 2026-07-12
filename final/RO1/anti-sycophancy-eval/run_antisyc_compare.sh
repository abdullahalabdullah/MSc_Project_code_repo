#!/usr/bin/env bash
#
# run_antisyc_compare.sh — paired base-vs-trained report (CPU, no GPU, no API).
#
# Both models must already have results (run_antisyc_results.sh). Computes
# per-question paired deltas with bootstrap CIs + exact McNemar (spec §4.3 rule 1)
# and renders the comparison markdown.
#
# Usage:
#   bash anti-sycophancy-eval/run_antisyc_compare.sh <base_id> <trained_id> [--capability-json <path>]

set -uo pipefail
cd "$(dirname "$0")/.."

USAGE="usage: bash anti-sycophancy-eval/run_antisyc_compare.sh <base_id> <trained_id> [--capability-json <path>]"
BASE_ID="${1:?$USAGE}"
TRAINED_ID="${2:?$USAGE}"
shift 2
EXTRA_ARGS=("$@")

BASE_NAME="${BASE_ID//\//_}"
TRAINED_NAME="${TRAINED_ID//\//_}"
SCRIPTS="anti-sycophancy-eval/scripts"

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
LOGFILE="logs/antisyc_compare_${BASE_NAME}__vs__${TRAINED_NAME}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "Anti-sycophancy comparison started: $(date)"
echo "Base:    $BASE_ID"
echo "Trained: $TRAINED_ID"

echo ""
echo "########## PHASE 0: git identity ##########"
run git config --global user.email "abd@gmail.com"
run git config --global user.name "Abd"

echo ""
echo "########## PHASE 1: paired statistics ##########"
run python "$SCRIPTS/05_compare_models.py" --base "$BASE_ID" --trained "$TRAINED_ID" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo ""
echo "########## PHASE 2: comparison report ##########"
run python "$SCRIPTS/06_generate_reports.py" --base "$BASE_ID" --trained "$TRAINED_ID"

echo ""
echo "########## FINAL PUSH ##########"
commit_push "anti-sycophancy: paired comparison ${BASE_NAME} vs ${TRAINED_NAME}"

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
