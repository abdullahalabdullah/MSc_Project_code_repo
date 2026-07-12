#!/usr/bin/env bash
#
# run_harmlessness.sh — full harmlessness eval in one command (GPU box).
#
# Generates target-model responses to the WildJailbreak eval prompts, classifies
# each with WildGuard (independent judge), and computes Safety Rate + Refusal
# Rate (overall + per subset). WildGuard is a local 7B model, so BOTH the target
# generation and the classification run on the GPU box — hence a single command
# (unlike the monitorability eval, whose monitor is an API call run separately).
#
# Usage:
#   bash harmlessness-eval/run_harmlessness.sh <model_id> [--data test|eval] [--merge]
#
#   model_id      HF model ID (e.g. Qwen/Qwen3-4B) or a local model dir. All output
#                 dirs are namespaced by its folder-safe name ('/' -> '_').
#   --data        OPTIONAL (default: test). Data split: test (2/3 of the
#                 WildJailbreak eval pool) or eval (disjoint half of test).
#                 The eval split is namespaced under .../eval/; its prompts index
#                 must have been prepared already (01_prepare_wildjailbreak.py
#                 --data eval, which needs the FINAL test index).
#   --merge       OPTIONAL (default: off). Treat model_id as a LoRA ADAPTER repo
#                 (adapter-only push, no config.json, e.g. RO2 checkpoints): step 2
#                 merges it into the fixed base from the config merge: block and
#                 runs inference on the local merged dir. Output dirs stay
#                 namespaced by the adapter id.
#
# Examples:
#   bash harmlessness-eval/run_harmlessness.sh Qwen/Qwen3-4B
#   bash harmlessness-eval/run_harmlessness.sh Qwen/Qwen3-4B --data eval
#   bash harmlessness-eval/run_harmlessness.sh er-abd/harmless-dpo-qwen3-4B-pilot --merge
#
# Fail-soft: a failing step is logged as FAIL and the run continues (no set -e).
# Progress is committed + pushed after generation and classification, plus a
# catch-all final push. Output is mirrored to logs/ (gitignored).

set -uo pipefail

# cd to the repo root (this script lives in harmlessness-eval/) so the shared
# config/, outputs-and-results/, and logs/ resolve as repo-root-relative paths.
cd "$(dirname "$0")/.."

USAGE="usage: bash harmlessness-eval/run_harmlessness.sh <model_id> [--data test|eval] [--merge]"
MODEL_ID="${1:?$USAGE}"
shift
MERGE_FLAG=""
DATA_SPLIT="test"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --merge) MERGE_FLAG="--merge" ;;
        --data)
            if [[ $# -lt 2 || ! "$2" =~ ^(test|eval)$ ]]; then
                echo "ERROR: --data requires 'test' or 'eval' (got '${2:-}')."; echo "$USAGE"; exit 1
            fi
            DATA_SPLIT="$2"; shift ;;
        *) echo "ERROR: unknown argument '$1'."; echo "$USAGE"; exit 1 ;;
    esac
    shift
done

# Folder-safe model name — must mirror model_dir_name() in harmlessness-eval/src/dataset.py.
MODEL_NAME="${MODEL_ID//\//_}"

# Split segment for the log filename; the scripts namespace their own dirs.
SPLIT_TAG=""
[[ "$DATA_SPLIT" == "eval" ]] && SPLIT_TAG="_eval"

SCRIPTS="harmlessness-eval/scripts"

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
LOGFILE="logs/harmlessness_${MODEL_NAME}${SPLIT_TAG}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "Harmlessness run started: $(date)"
echo "Model:    $MODEL_ID (dirs namespaced as '$MODEL_NAME')"
echo "Split:    $DATA_SPLIT"
echo "Log file: $LOGFILE"

# --- Phase 0: git identity ---------------------------------------------------
echo ""
echo "########## PHASE 0: git identity ##########"
run git config --global user.email "abd@gmail.com"
run git config --global user.name "Abd"

# --- Phase 1: prepare prompts (idempotent) -----------------------------------
# For --data eval the prompts index must already exist (01 --data eval needs the
# FINAL test index); this idempotent call is a no-op if it's there.
echo ""
echo "########## PHASE 1: prepare WildJailbreak prompts ($DATA_SPLIT) ##########"
run python "$SCRIPTS/01_prepare_wildjailbreak.py" --data "$DATA_SPLIT"
commit_push "harmlessness: WildJailbreak $DATA_SPLIT prompts index"

# --- Phase 2: target-model inference (GPU) -----------------------------------
echo ""
echo "########## PHASE 2: inference — $MODEL_ID ($DATA_SPLIT) ##########"
run python "$SCRIPTS/02_run_inference.py" --model "$MODEL_ID" --data "$DATA_SPLIT" $MERGE_FLAG
commit_push "harmlessness: $MODEL_NAME $DATA_SPLIT responses"

# --- Phase 3: WildGuard classification (GPU) ---------------------------------
echo ""
echo "########## PHASE 3: WildGuard classification ($DATA_SPLIT) ##########"
run python "$SCRIPTS/03_run_wildguard.py" --model "$MODEL_ID" --data "$DATA_SPLIT"
commit_push "harmlessness: $MODEL_NAME $DATA_SPLIT WildGuard classifications"

# --- Phase 4: metrics --------------------------------------------------------
echo ""
echo "########## PHASE 4: metrics ($DATA_SPLIT) ##########"
run python "$SCRIPTS/04_compute_metrics.py" --model "$MODEL_ID" --data "$DATA_SPLIT"

# --- Phase 5: Markdown report ------------------------------------------------
# Kept as ONE separable step: delete this line (plus scripts/05_generate_reports.py
# and src/reporting.py) to strip all .md report generation for a public release.
echo ""
echo "########## PHASE 5: markdown report ($DATA_SPLIT) ##########"
run python "$SCRIPTS/05_generate_reports.py" --model "$MODEL_ID" --data "$DATA_SPLIT"

# --- Final push --------------------------------------------------------------
echo ""
echo "########## FINAL PUSH ##########"
commit_push "harmlessness: $MODEL_NAME $DATA_SPLIT run complete — final push"

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
