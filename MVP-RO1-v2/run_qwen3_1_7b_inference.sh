#!/usr/bin/env bash
#
# run_qwen3_1_7b_inference.sh — Qwen3-1.7B inference run (NO monitor, NO results).
# One or more hint types, all active datasets, configurable examples-per-dataset
# (n) and samples-per-arm (N). Run this on the GPU box (e.g. RunPod, A100 SXM).
#
# Phases:
#   0. git identity (safe on a fresh pod).
#   1. Prepare dataset indexes (01), once per dataset (the index is shared by every
#      hint type). Idempotent by default; set N_EXAMPLES and/or FORCE_PREPARE=true
#      to (re)generate. Commits + pushes the indexes.
#   2. Inference (02) — for every (hint type × dataset): hinted + unhinted arms,
#      thinking ON. Commits + pushes the outputs after EACH (hint, dataset), so
#      progress survives a crash/spot reclaim.
#   3. CoT token-length analysis (08) — per hint type, across all datasets.
#
# The monitor (03) and results calculation (04/05/06) are NOT run here — use
# run_qwen3_1_7b_monitor_and_results.sh afterwards (set its HINT_TYPE to match
# whichever hint type you want results for).
#
# Fail-soft: a failing command is logged as FAIL and the run continues (no set -e).
#
# Usage:
#   # HF auth is handled in-code via config hf_token / HF_TOKEN (see src/dataset.py)
#   bash run_qwen3_1_7b_inference.sh
#
# Knobs: edit MODEL / HINT_TYPES / N_EXAMPLES / N_SAMPLES / FORCE_PREPARE / DATASETS.

set -uo pipefail

# Run from the repo root (this script's directory) so scripts/ + config/ resolve.
cd "$(dirname "$0")"

# --- Configuration ----------------------------------------------------------
# Config model name (the `name:` key in config/mvp_config.yaml). To run the DPO
# variant instead, set MODEL=aqua_dpo_qwen3_1_7b_s0 (and merge it first).
MODEL="qwen3_1_7b"                # thinking ON (temp 0.6, top_p 0.95, max_new_tokens 16384)

# Hint type(s) to run — a list, like DATASETS. The run sweeps every hint type over
# every dataset. Valid: professor, visual_pattern, metadata, wrong_few_shot,
# grader_hacking, unethical. Output is namespaced by hint type, so they never collide.
# For a single hint type, just list one, e.g. HINT_TYPES=(professor).
HINT_TYPES=(professor)

# Examples per dataset (n). Empty "" = use each dataset's config n_examples budget.
# Set a number (e.g. 50) to override it for this run.
N_EXAMPLES=""

# Samples per arm per instance (N). Empty "" = use config qwen_inference.n_samples.
# Set a number (e.g. 4, 8, 16) to override. No fixed cap.
N_SAMPLES=""

# Force-regenerate dataset indexes even if they already exist. Leave "false" for
# idempotent prep; set "true" when you change N_EXAMPLES (so the old index, e.g.
# 200 examples, is replaced). Note: setting N_EXAMPLES below also forces a rebuild.
FORCE_PREPARE="false"

# All active datasets (logieval is commented out in the config registry — omit it).
DATASETS=(mmlu_redux gpqa_diamond mathbench_primary mathbench_middle mathbench_high mathbench_college compmath wmdp codemmlu)

LOGFILE="run_qwen3_1_7b_inference_$(date +%Y%m%d_%H%M%S).log"

# Build the optional CLI flags from the knobs above.
PREP_FLAGS=""
[[ -n "$N_EXAMPLES" ]] && PREP_FLAGS+=" --n-examples $N_EXAMPLES"
# A changed N_EXAMPLES needs a rebuild; honour an explicit FORCE_PREPARE too.
{ [[ -n "$N_EXAMPLES" ]] || [[ "$FORCE_PREPARE" == "true" ]]; } && PREP_FLAGS+=" --force"
INFER_FLAGS=""
[[ -n "$N_SAMPLES" ]] && INFER_FLAGS+=" --n-samples $N_SAMPLES"
# n/N suffix appended to commit messages for traceability (hint-independent).
NN_TAG=""
[[ -n "$N_EXAMPLES" ]] && NN_TAG+=", n=$N_EXAMPLES"
[[ -n "$N_SAMPLES" ]]  && NN_TAG+=", N=$N_SAMPLES"

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

echo "Qwen3-1.7B inference run started: $(date)"
echo "Model:       $MODEL (qwen_inference, thinking ON)"
echo "Hints:       ${HINT_TYPES[*]}"
echo "N examples:  ${N_EXAMPLES:-config default} per dataset"
echo "N samples:   ${N_SAMPLES:-config default} per arm per instance"
echo "Force prep:  $FORCE_PREPARE"
echo "Datasets:    ${DATASETS[*]}"
echo "Log file:    $LOGFILE"

# --- Phase 0: git identity ---------------------------------------------------
# Ensure a committer identity exists so the per-dataset commits never fail on a
# fresh box (e.g. RunPod) where git has no global user configured.
echo ""
echo "########## PHASE 0: git identity ##########"
run git config --global user.email "abd@gmail.com"
run git config --global user.name "Abd"

# --- commit_push: stage, commit, and push the current results ----------------
# Fail-soft and tolerant of "nothing to commit": only push when the commit
# actually created something, so a no-op dataset doesn't abort the run.
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

# --- Phase 1: prepare dataset indexes ----------------------------------------
echo ""
echo "########## PHASE 1: prepare dataset indexes ##########"
for ds in "${DATASETS[@]}"; do
    run python scripts/01_prepare_dataset.py --dataset "$ds" $PREP_FLAGS
done
commit_push "dataset indexes (${N_EXAMPLES:-config budget}) — ${DATASETS[*]}"

# --- Phase 2: inference (commit + push after each hint type × dataset) --------
echo ""
echo "########## PHASE 2: inference — $MODEL ##########"
for ht in "${HINT_TYPES[@]}"; do
    echo ""
    echo "---------- hint type: $ht ----------"
    for ds in "${DATASETS[@]}"; do
        run python scripts/02_run_inference_qwen.py --dataset "$ds" --model "$MODEL" --hint-type "$ht" $INFER_FLAGS
        commit_push "$MODEL inference results ($ht$NN_TAG) — $ds"
    done
done

# --- Phase 3: cross-dataset CoT token-length analysis (per hint type) ---------
echo ""
echo "########## PHASE 3: CoT token-length analysis — $MODEL ##########"
for ht in "${HINT_TYPES[@]}"; do
    run python scripts/08_cot_token_analysis.py --model "$MODEL" --datasets "${DATASETS[@]}" --hint-type "$ht"
    commit_push "$MODEL — CoT token-length analysis ($ht$NN_TAG)"
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
