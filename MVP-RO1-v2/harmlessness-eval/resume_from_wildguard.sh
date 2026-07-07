#!/usr/bin/env bash
#
# resume_from_wildguard.sh — resume the harmlessness eval from WildGuard
# classification onward (GPU box). Use this when target-model responses
# (phase 2 of run_harmlessness.sh) already exist — e.g. inference succeeded
# but a later phase failed (disk full, dropped connection, etc.) — so you
# don't have to re-run the ~20+ min generation step.
#
# Usage:
#   bash harmlessness-eval/resume_from_wildguard.sh <model_id>
#
#   model_id   the SAME model ID used with run_harmlessness.sh (resolves the
#              same responses dir). All output dirs are namespaced by its
#              folder-safe name ('/' -> '_').
#
# Example:
#   bash harmlessness-eval/resume_from_wildguard.sh er-abd/harmless-dpo-qwen3-4B-pilot
#
# Runs: 03 WildGuard classification -> 04 metrics -> 05 markdown report, then
# commits + pushes. Fail-soft (no set -e). Output is mirrored to logs/ (gitignored).

set -uo pipefail

# cd to the repo root (this script lives in harmlessness-eval/) so the shared
# config/, outputs-and-results/, and logs/ resolve as repo-root-relative paths.
cd "$(dirname "$0")/.."

USAGE="usage: bash harmlessness-eval/resume_from_wildguard.sh <model_id>"
MODEL_ID="${1:?$USAGE}"

# Folder-safe model name — must mirror model_dir_name() in harmlessness-eval/src/dataset.py.
MODEL_NAME="${MODEL_ID//\//_}"

SCRIPTS="harmlessness-eval/scripts"

# --- Preflight: phase-2 responses must already exist -------------------------
RESPONSES="outputs-and-results/harmlessness-results/responses/${MODEL_NAME}/responses.jsonl"
if [[ ! -f "$RESPONSES" ]]; then
    echo "ERROR: $RESPONSES not found."
    echo "Nothing to resume from — run harmlessness-eval/run_harmlessness.sh first."
    exit 1
fi

# --- Fail-soft command runner --------------------------------------------------
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

# Stage, commit, and push the current results. Fail-soft and tolerant of
# "nothing to commit": only push when the commit actually created something.
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

# Mirror all output to a timestamped log file (logs/ is gitignored).
mkdir -p logs
LOGFILE="logs/harmlessness_resume_${MODEL_NAME}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "Harmlessness resume (from WildGuard) started: $(date)"
echo "Model:    $MODEL_ID (dirs namespaced as '$MODEL_NAME')"
echo "Log file: $LOGFILE"

# --- Phase 0: git identity ---------------------------------------------------
echo ""
echo "########## PHASE 0: git identity ##########"
run git config --global user.email "abd@gmail.com"
run git config --global user.name "Abd"

# --- Phase 1: pull any remote work before committing on top of it -------------
echo ""
echo "########## PHASE 1: git pull --rebase ##########"
run git pull --rebase

# --- Phase 2: WildGuard classification (GPU) ----------------------------------
echo ""
echo "########## PHASE 2: WildGuard classification ##########"
run python "$SCRIPTS/03_run_wildguard.py" --model "$MODEL_ID"
commit_push "harmlessness: $MODEL_NAME WildGuard classifications"

# --- Phase 3: metrics ----------------------------------------------------------
echo ""
echo "########## PHASE 3: metrics ##########"
run python "$SCRIPTS/04_compute_metrics.py" --model "$MODEL_ID"

# --- Phase 4: Markdown report ---------------------------------------------------
echo ""
echo "########## PHASE 4: markdown report ##########"
run python "$SCRIPTS/05_generate_reports.py" --model "$MODEL_ID"

# --- Final push --------------------------------------------------------------
echo ""
echo "########## FINAL PUSH ##########"
commit_push "harmlessness: $MODEL_NAME WildGuard classifications + metrics + report — final push"

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
