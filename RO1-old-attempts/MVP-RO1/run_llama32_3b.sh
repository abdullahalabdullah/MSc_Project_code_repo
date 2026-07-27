#!/usr/bin/env bash
#
# run_llama32_3b.sh — RO1 inference over all datasets for meta-llama/Llama-3.2-3B-Instruct.
#
# Mirrors run_qwen.sh, but:
#   - uses the standard Llama/Tulu inference path (scripts/02_run_inference.py),
#     since Llama-3.2-3B-Instruct is a standard (non-thinking) instruct model,
#   - does NOT (re)prepare the datasets — data/{ds}/dataset_index.json already exist.
#
# 70 examples per dataset is set in config/mvp_config.yaml (n_examples: 70).
# Prompt body uses _QUESTION_BLOCK_BODY (NOT _QUESTION_BLOCK_BODY_V2) — see src/inference.py.
#
# Order:
#   1. Inference: llama32_3b_instruct over every dataset.
#   2. Cross-dataset CoT token-length analysis.
#   3. git add / commit / push.
#
# Fail-soft: if any single command errors, it is logged as FAIL and the script
# moves on to the next one (no `set -e`). A summary is printed at the end.
#
# Usage:
#   # HF auth is handled in-code via .hf_token / HF_TOKEN (see src/dataset.py)
#   bash run_llama32_3b.sh

set -uo pipefail

# Run from the MVP-RO1 root (this script's directory) so scripts/ + config/ resolve.
cd "$(dirname "$0")"

# --- Configuration ----------------------------------------------------------
# All datasets. Data already prepared in data/{ds}/ — no prepare phase.
DATASETS=(mmlu gpqa gsm_mc math_mc agieval wmdp)

# Config model name (the `name:` key in config/mvp_config.yaml) that maps to:
#   llama32_3b_instruct -> meta-llama/Llama-3.2-3B-Instruct
LLAMA_MODEL="llama32_3b_instruct"

LOGFILE="run_llama32_3b_$(date +%Y%m%d_%H%M%S).log"

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

echo "RO1 Llama-3.2-3B-Instruct run started: $(date)"
echo "Datasets: ${DATASETS[*]}"
echo "Model:    $LLAMA_MODEL (meta-llama/Llama-3.2-3B-Instruct)"
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
run git commit -m "llama3.2-3b-instruct results (all datasets)"
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
