#!/usr/bin/env bash
#
# run_antisyc_inference.sh — anti-sycophancy GPU phase for one model.
#
# Verifies the frozen eval data, then runs the three-pass inference (ID arms, OOD
# turn 1, OOD "Are you sure?" turn 2). Grading is an API step run separately with
# run_antisyc_results.sh — nothing here needs OPENAI_API_KEY, and nothing there
# needs a GPU.
#
# Usage:
#   bash anti-sycophancy-eval/run_antisyc_inference.sh <model_id> [--merge]
#
#   model_id   HF model ID (e.g. Qwen/Qwen3-4B) or a local model dir. All output
#              dirs are namespaced by its folder-safe name ('/' -> '_').
#   --merge    OPTIONAL. Treat model_id as a LoRA ADAPTER repo (RO2 checkpoints):
#              merged into the config merge: base before inference.
#
# Examples:
#   bash anti-sycophancy-eval/run_antisyc_inference.sh Qwen/Qwen3-4B
#   bash anti-sycophancy-eval/run_antisyc_inference.sh er-abd/anti-syc-dpo-qwen3-4B-step-N --merge
#
# Fail-soft: a failing step is logged as FAIL and the run continues (no set -e).
# Output is mirrored to logs/ (gitignored); results are committed + pushed.

set -uo pipefail
cd "$(dirname "$0")/.."

USAGE="usage: bash anti-sycophancy-eval/run_antisyc_inference.sh <model_id> [--merge]"
MODEL_ID="${1:?$USAGE}"
MERGE_FLAG=""
[[ "${2:-}" == "--merge" ]] && MERGE_FLAG="--merge"

# Folder-safe model name — must mirror model_dir_name() in anti-sycophancy-eval/src/dataset.py.
MODEL_NAME="${MODEL_ID//\//_}"

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
LOGFILE="logs/antisyc_inference_${MODEL_NAME}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "Anti-sycophancy inference started: $(date)"
echo "Model:    $MODEL_ID (dirs namespaced as '$MODEL_NAME')"
echo "Log file: $LOGFILE"

echo ""
echo "########## PHASE 0: git identity ##########"
run git config --global user.email "abd@gmail.com"
run git config --global user.name "Abd"

echo ""
echo "########## PHASE 1: verify frozen eval data + indexes ##########"
run python "$SCRIPTS/01_prepare_eval_data.py"

echo ""
echo "########## PHASE 2: inference (GPU) — $MODEL_ID ##########"
run python "$SCRIPTS/02_run_inference.py" --model "$MODEL_ID" $MERGE_FLAG
commit_push "anti-sycophancy: $MODEL_NAME responses"

echo ""
echo "########## FINAL PUSH ##########"
commit_push "anti-sycophancy: $MODEL_NAME inference complete"

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
echo "Next (no GPU): bash anti-sycophancy-eval/run_antisyc_results.sh $MODEL_ID"
