#!/usr/bin/env bash
#
# run_llama_harmless_e4.sh — RO1 inference over MMLU for the RO2 merged SFT model
# er-abd/llama-3.2-3b-sft-harmlessness-hh-merged-epoch4 (config name:
# llama32_3b_sft_harmless_e4).
#
# Mirrors run_qwen.sh / run_llama32_3b.sh, but:
#   - uses the standard Llama/Tulu inference path (scripts/02_run_inference.py),
#     since this is a standard (non-thinking) Llama-3.2-3B model,
#   - MMLU only,
#   - does NOT (re)prepare the dataset — data/mmlu/dataset_index.json already exists.
#
# 70 examples is set in config/mvp_config.yaml (dataset.n_examples: 70).
# Prompt body uses _QUESTION_BLOCK_BODY (NOT _QUESTION_BLOCK_BODY_V2) — see src/inference.py.
#
# Order:
#   1. Inference: llama32_3b_sft_harmless_e4 over MMLU.
#   2. Cross-dataset CoT token-length analysis.
#   3. git add / commit / push.
#
# Fail-soft: if any single command errors, it is logged as FAIL and the script
# moves on to the next one (no `set -e`). A summary is printed at the end.
#
# Usage:
#   # HF auth is handled in-code via .hf_token / HF_TOKEN (see src/dataset.py)
#   bash run_llama_harmless_e4.sh

set -uo pipefail

# Run from the MVP-RO1 root (this script's directory) so scripts/ + config/ resolve.
cd "$(dirname "$0")"

# --- Configuration ----------------------------------------------------------
# MMLU only. Data already prepared in data/mmlu/ — no prepare phase.
DATASETS=(mmlu)

# Config model name (the `name:` key in config/mvp_config.yaml) that maps to:
#   llama32_3b_sft_harmless_e4 -> er-abd/llama-3.2-3b-sft-harmlessness-hh-merged-epoch4
LLAMA_MODEL="llama32_3b_sft_harmless_e4"

LOGFILE="run_llama_harmless_e4_$(date +%Y%m%d_%H%M%S).log"

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

echo "RO1 Llama-3.2-3B SFT-harmless (epoch 4) run started: $(date)"
echo "Datasets: ${DATASETS[*]}"
echo "Model:    $LLAMA_MODEL (er-abd/llama-3.2-3b-sft-harmlessness-hh-merged-epoch4)"
echo "Log file: $LOGFILE"

# --- Phase 1: inference ------------------------------------------------------
echo ""
echo "########## PHASE 1: inference — $LLAMA_MODEL ##########"
for ds in "${DATASETS[@]}"; do
    run python scripts/02_run_inference.py --dataset "$ds" --model "$LLAMA_MODEL"
done

# --- Phase 1.5: cross-dataset CoT token-length analysis ----------------------
# Overall + per-dataset condition tables + per-dataset length buckets, to inform
# max_new_tokens. Writes analysis/cot_token_report_${LLAMA_MODEL}.md.
echo ""
echo "########## PHASE 1.5: CoT token-length analysis — $LLAMA_MODEL ##########"
run python scripts/08_cot_token_analysis.py --model "$LLAMA_MODEL" --datasets "${DATASETS[@]}"

# --- Phase 2: commit & push --------------------------------------------------
echo ""
echo "########## PHASE 2: git add / commit / push ##########"
run git add -A
run git commit -m "llama3.2-3b sft-harmless epoch4 results (mmlu)"
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
