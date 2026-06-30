#!/usr/bin/env bash
#
# run_tulu_sft_no_safety.sh — RO1 inference over MMLU for the Tulu 3 8B
# SFT (no-safety-data) baseline checkpoint.
#
# Mirrors run_qwen.sh, but:
#   - uses the standard Llama/Tulu inference path (scripts/02_run_inference.py),
#   - runs on MMLU only,
#   - does NOT (re)prepare the dataset — data/mmlu/dataset_index.json already exists.
#
# 70 examples is set in config/mvp_config.yaml (dataset.n_examples: 70).
# Prompt body uses _QUESTION_BLOCK_BODY_V2 (see src/inference.py).
#
# Order:
#   1. Inference: tulu_sft_no_safety.
#   2. Cross-dataset CoT token-length analysis.
#   3. git add / commit / push.
#
# Fail-soft: if any single command errors, it is logged as FAIL and the script
# moves on to the next one (no `set -e`). A summary is printed at the end.
#
# Usage:
#   # HF auth is handled in-code via .hf_token / HF_TOKEN (see src/dataset.py)
#   bash run_tulu_sft_no_safety.sh

set -uo pipefail

# Run from the MVP-RO1 root (this script's directory) so scripts/ + config/ resolve.
cd "$(dirname "$0")"

# --- Configuration ----------------------------------------------------------
# MMLU only. Data already prepared in data/mmlu/ — no prepare phase.
DATASETS=(mmlu)

# Config model name (the `name:` key in config/mvp_config.yaml) that maps to:
#   tulu_sft_no_safety -> allenai/Llama-3.1-Tulu-3-8B-SFT-no-safety-data
TULU_MODEL="tulu_sft_no_safety"

LOGFILE="run_tulu_sft_no_safety_$(date +%Y%m%d_%H%M%S).log"

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

echo "RO1 Tulu SFT (no-safety) run started: $(date)"
echo "Datasets: ${DATASETS[*]}"
echo "Model:    $TULU_MODEL (Llama-3.1-Tulu-3-8B-SFT-no-safety-data)"
echo "Log file: $LOGFILE"

# --- Phase 1: inference ------------------------------------------------------
echo ""
echo "########## PHASE 1: inference — $TULU_MODEL ##########"
for ds in "${DATASETS[@]}"; do
    run python scripts/02_run_inference.py --dataset "$ds" --model "$TULU_MODEL"
done

# --- Phase 1.5: cross-dataset CoT token-length analysis ----------------------
# Overall + per-dataset condition tables + per-dataset length buckets, to inform
# max_new_tokens. Writes analysis/cot_token_report_${TULU_MODEL}.md.
echo ""
echo "########## PHASE 1.5: CoT token-length analysis — $TULU_MODEL ##########"
run python scripts/08_cot_token_analysis.py --model "$TULU_MODEL" --datasets "${DATASETS[@]}"

# --- Phase 2: commit & push --------------------------------------------------
echo ""
echo "########## PHASE 2: git add / commit / push ##########"
run git add -A
run git commit -m "tulu sft no-safety results (mmlu, V2 prompt)"
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
