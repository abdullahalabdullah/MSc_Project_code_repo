#!/usr/bin/env bash
#
# run_mistral.sh — full RO1 pipeline over the mmlu dataset for Mistral-7B-Instruct-v0.1.
#
# Mirrors run_qwen.sh, but Mistral is a standard (non-thinking) instruct model, so it
# uses the standard inference path scripts/02_run_inference.py (-> src/inference.py,
# which builds prompts from _QUESTION_BLOCK_BODY), NOT the Qwen thinking path.
#
# Order:
#   1. Prepare every dataset index (model-independent, idempotent).
#   2. Inference: Mistral-7B-Instruct-v0.1.
#   3. git add / commit / push.
#
# Fail-soft: if any single command errors, it is logged as FAIL and the script
# moves on to the next one (no `set -e`). A summary is printed at the end.
#
# Usage:
#   # HF auth is handled in-code via .hf_token / HF_TOKEN (see src/dataset.py)
#   bash run_mistral.sh

set -uo pipefail

# Run from the MVP-RO1 root (this script's directory) so scripts/ + config/ resolve.
cd "$(dirname "$0")"

# --- Configuration ----------------------------------------------------------
# Datasets to process. Edit this list to add/remove tasks (re-runs resume from
# checkpoints and are cheap). Both use n_examples: 70 (config/mvp_config.yaml).
DATASETS=(mmlu gpqa)


# Config model name (the `name:` key in config/mvp_config.yaml) that maps to:
#   mistral_7b_instruct -> mistralai/Mistral-7B-Instruct-v0.1
MISTRAL_MODEL="mistral_7b_instruct"

LOGFILE="run_mistral_$(date +%Y%m%d_%H%M%S).log"

# --- Fail-soft command runner ------------------------------------------------
SUMMARY=()

run() {
    echo ""
    echo "=================================================================="
    echo ">>> $* "
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

echo "RO1 Mistral run started: $(date)"
echo "Datasets: ${DATASETS[*]}"
echo "Model:    $MISTRAL_MODEL (Mistral-7B-Instruct-v0.1)"
echo "Log file: $LOGFILE"

# --- Phase 1: prepare dataset indexes (model-independent) --------------------
echo ""
echo "########## PHASE 1: prepare dataset indexes ##########"
for ds in "${DATASETS[@]}"; do
    run python scripts/01_prepare_dataset.py --dataset "$ds"
done

# --- Phase 2: inference ------------------------------------------------------
echo ""
echo "########## PHASE 2: inference — $MISTRAL_MODEL ##########"
for ds in "${DATASETS[@]}"; do
    run python scripts/02_run_inference.py --dataset "$ds" --model "$MISTRAL_MODEL"
done

# --- Phase 2.5: cross-dataset CoT token-length analysis ----------------------
# Overall + per-dataset condition tables + per-dataset length buckets, to inform
# max_new_tokens. Writes analysis/cot_token_report_${MISTRAL_MODEL}.md.
echo ""
echo "########## PHASE 2.5: CoT token-length analysis — $MISTRAL_MODEL ##########"
run python scripts/08_cot_token_analysis.py --model "$MISTRAL_MODEL" --datasets "${DATASETS[@]}"

# --- Phase 3: commit & push --------------------------------------------------
echo ""
echo "########## PHASE 3: git add / commit / push ##########"
run git add -A
run git commit -m "mistral results"
run git push

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
