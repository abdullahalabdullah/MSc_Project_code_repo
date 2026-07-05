#!/usr/bin/env bash
#
# run_inference.sh — GPU phase: prepare datasets + hinted/unhinted inference +
# CoT token-length report. Run on the GPU box (e.g. RunPod). The monitor and
# results phase runs separately afterwards: see run_monitor_and_results.sh.
#
# Usage:
#   bash monitorability-eval/run_inference.sh <model_id> <dataset|d1,d2|ALL> <hint_type|ALL>
#
#   model_id   HF model ID (e.g. Qwen/Qwen3-4B) or a local model dir. All output
#              dirs are namespaced by its folder-safe name ('/' -> '_').
#   dataset    a dataset key from config/mvp_config.yaml, a comma-separated subset
#              (e.g. mmlu_pro,lexam), or ALL for every key.
#   hint_type  a hint type (professor, visual_pattern, metadata, wrong_few_shot,
#              grader_hacking, unethical), or ALL for every type.
#
# Examples:
#   bash monitorability-eval/run_inference.sh Qwen/Qwen3-4B ALL professor
#   bash monitorability-eval/run_inference.sh Qwen/Qwen3-4B mmlu_pro,lexam professor
#   bash monitorability-eval/run_inference.sh Qwen/Qwen3-1.7B wmdp ALL
#
# Optional env knobs:
#   N_EXAMPLES=50        override each dataset's config n_examples budget
#   N_SAMPLES=8          override config qwen_inference.n_samples (N per arm)
#   FORCE_PREPARE=true   regenerate dataset indexes (implied when N_EXAMPLES is set)
#
# Fail-soft: a failing step is logged as FAIL and the run continues (no set -e).
# Progress is committed + pushed after each phase so a crash/spot reclaim never
# loses results. Output is mirrored to logs/ (gitignored).

set -uo pipefail

# cd to the repo root (this script lives in monitorability-eval/) so the shared
# config/, outputs-and-results/, and logs/ resolve as repo-root-relative paths.
cd "$(dirname "$0")/.."

USAGE="usage: bash monitorability-eval/run_inference.sh <model_id> <dataset|d1,d2|ALL> <hint_type|ALL>"
MODEL_ID="${1:?$USAGE}"
DATASET_ARG="${2:?$USAGE}"
HINT_ARG="${3:?$USAGE}"

# Folder-safe model name — must mirror model_dir_name() in src/dataset.py.
MODEL_NAME="${MODEL_ID//\//_}"

# --- Expand/validate dataset + hint args against the config registries -------
ALL_DATASETS=($(python -c "import yaml; print(' '.join(yaml.safe_load(open('config/mvp_config.yaml'))['datasets']))"))
ALL_HINTS=($(python -c "import yaml; print(' '.join(yaml.safe_load(open('config/mvp_config.yaml'))['hint']['types']))"))

if [[ "$DATASET_ARG" == "ALL" ]]; then
    DATASETS=("${ALL_DATASETS[@]}")
else
    # A single key or a comma-separated subset, e.g. mmlu_pro,lexam. Passed as one
    # list to 02 so the model is loaded ONCE and reused across the given datasets.
    IFS=',' read -r -a DATASETS <<< "$DATASET_ARG"
    for ds in "${DATASETS[@]}"; do
        if [[ " ${ALL_DATASETS[*]} " != *" $ds "* ]]; then
            echo "ERROR: unknown dataset '$ds'. Valid: ${ALL_DATASETS[*]} (or ALL, or a comma-separated subset)."
            exit 1
        fi
    done
fi

if [[ "$HINT_ARG" == "ALL" ]]; then
    HINT_TYPES=("${ALL_HINTS[@]}")
elif [[ " ${ALL_HINTS[*]} " == *" $HINT_ARG "* ]]; then
    HINT_TYPES=("$HINT_ARG")
else
    echo "ERROR: unknown hint type '$HINT_ARG'. Valid: ${ALL_HINTS[*]} (or ALL)."
    exit 1
fi

# --- Optional knobs -> CLI flags ----------------------------------------------
N_EXAMPLES="${N_EXAMPLES:-}"
N_SAMPLES="${N_SAMPLES:-}"
FORCE_PREPARE="${FORCE_PREPARE:-false}"

PREP_FLAGS=""
[[ -n "$N_EXAMPLES" ]] && PREP_FLAGS+=" --n-examples $N_EXAMPLES"
# A changed N_EXAMPLES needs a rebuild; honour an explicit FORCE_PREPARE too.
{ [[ -n "$N_EXAMPLES" ]] || [[ "$FORCE_PREPARE" == "true" ]]; } && PREP_FLAGS+=" --force"
INFER_FLAGS=""
[[ -n "$N_SAMPLES" ]] && INFER_FLAGS+=" --n-samples $N_SAMPLES"
# n/N suffix appended to commit messages for traceability.
NN_TAG=""
[[ -n "$N_EXAMPLES" ]] && NN_TAG+=", n=$N_EXAMPLES"
[[ -n "$N_SAMPLES" ]]  && NN_TAG+=", N=$N_SAMPLES"

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
LOGFILE="logs/inference_${MODEL_NAME}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "Inference run started: $(date)"
echo "Model:       $MODEL_ID (dirs namespaced as '$MODEL_NAME')"
echo "Datasets:    ${DATASETS[*]}"
echo "Hint types:  ${HINT_TYPES[*]}"
echo "N examples:  ${N_EXAMPLES:-config default} per dataset"
echo "N samples:   ${N_SAMPLES:-config default} per arm per instance"
echo "Force prep:  $FORCE_PREPARE"
echo "Log file:    $LOGFILE"

# --- Phase 0: git identity ------------------------------------------------------
# Ensure a committer identity exists so the per-phase commits never fail on a
# fresh box (e.g. RunPod) where git has no global user configured.
echo ""
echo "########## PHASE 0: git identity ##########"
run git config --global user.email "abd@gmail.com"
run git config --global user.name "Abd"

# --- Phase 1: prepare dataset indexes (shared by every hint type) ---------------
echo ""
echo "########## PHASE 1: prepare dataset indexes ##########"
for ds in "${DATASETS[@]}"; do
    run python monitorability-eval/scripts/01_prepare_dataset.py --dataset "$ds" $PREP_FLAGS
done
commit_push "dataset indexes${NN_TAG} — ${DATASETS[*]}"

# --- Phase 2: inference + CoT report, per hint type ------------------------------
# The model is loaded ONCE per hint type and reused across all datasets.
for hint in "${HINT_TYPES[@]}"; do
    echo ""
    echo "########## PHASE 2: inference — $MODEL_ID ($hint) ##########"
    run python monitorability-eval/scripts/02_run_inference_qwen.py --model "$MODEL_ID" --datasets "${DATASETS[@]}" --hint-type "$hint" $INFER_FLAGS
    commit_push "$MODEL_NAME inference ($hint$NN_TAG) — ${DATASETS[*]}"

    echo ""
    echo "########## PHASE 3: CoT token-length analysis — $MODEL_ID ($hint) ##########"
    run python monitorability-eval/scripts/08_cot_token_analysis.py --model "$MODEL_ID" --datasets "${DATASETS[@]}" --hint-type "$hint"
    commit_push "$MODEL_NAME CoT token-length analysis ($hint$NN_TAG)"
done

# --- Final push --------------------------------------------------------------------
# Catch-all: make sure EVERYTHING produced by this run is on GitHub, even if an
# intermediate per-phase push failed (fail-soft) or files were left unstaged.
echo ""
echo "########## FINAL PUSH ##########"
commit_push "$MODEL_NAME inference run complete (${HINT_TYPES[*]}$NN_TAG) — final push"

# --- Summary ---------------------------------------------------------------------
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
echo "Next: pull locally, then  bash monitorability-eval/run_monitor_and_results.sh $MODEL_ID $DATASET_ARG $HINT_ARG"
