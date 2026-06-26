#!/usr/bin/env bash
#
# run_qwen3_1_7b.sh — RO1 pipeline over MMLU/AGIEval/GPQA/WMDP for Qwen3-1.7B (thinking ON).
#
# Same structure as run_qwen.sh, but for the smaller Qwen/Qwen3-1.7B checkpoint.
# Reads sampling params from the `qwen_inference:` config block (enable_thinking=true,
# temp 0.6, top_p 0.95, max_new_tokens 16384), so the model emits its <think> trace
# inline. The prompt is the shared _QUESTION_BLOCK_BODY from src/inference.py
# (via build_user_content in src/inference_qwen.py) — identical to every other arm.
#
# Data is already prepared under data/ (n_examples=70 per dataset), so this script
# skips dataset preparation and goes straight to inference.
#
# ENV: runs in the SAME venv as run_qwen.sh — the Qwen3 env from
# requirements-qwen.txt (vLLM 0.8.5 / transformers 4.51.3). No new dependencies.
#
# Order:
#   1. Inference: Qwen3-1.7B (thinking on) over each dataset.
#   2. Cross-dataset CoT token-length analysis.
#   3. git add / commit / push.
#
# Fail-soft: if any single command errors, it is logged as FAIL and the script
# moves on to the next one (no `set -e`). A summary is printed at the end.
#
# Usage:
#   # HF auth is handled in-code via .hf_token / HF_TOKEN (see src/dataset.py)
#   bash run_qwen3_1_7b.sh

set -uo pipefail

# Run from the MVP-RO1 root (this script's directory) so scripts/ + config/ resolve.
cd "$(dirname "$0")"

# --- Configuration ----------------------------------------------------------
# Datasets to process. Data is already prepared under data/, so no prep phase.
DATASETS=(mmlu agieval gpqa wmdp)

# Config model name (the `name:` key in config/mvp_config.yaml) that maps to:
#   qwen3_1_7b -> Qwen/Qwen3-1.7B
QWEN_MODEL="qwen3_1_7b"

# Config block (sampling params) for this run — thinking is enabled here.
QWEN_CONFIG_BLOCK="qwen_inference"

LOGFILE="run_qwen3_1_7b_$(date +%Y%m%d_%H%M%S).log"

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

echo "RO1 Qwen3-1.7B (thinking on) run started: $(date)"
echo "Datasets: ${DATASETS[*]}"
echo "Model:    $QWEN_MODEL (Qwen/Qwen3-1.7B, thinking on via $QWEN_CONFIG_BLOCK)"
echo "Log file: $LOGFILE"

# --- Phase 1: inference ------------------------------------------------------
# Data is already prepared (n_examples=70 per dataset), so we skip 01_prepare_dataset.py.
echo ""
echo "########## PHASE 1: inference — $QWEN_MODEL ##########"
for ds in "${DATASETS[@]}"; do
    run python scripts/02_run_inference_qwen.py --dataset "$ds" --model "$QWEN_MODEL" --config-block "$QWEN_CONFIG_BLOCK"
done

# --- Phase 1.5: cross-dataset CoT token-length analysis ----------------------
# Overall + per-dataset condition tables + per-dataset length buckets, to inform
# max_new_tokens. Writes analysis/cot_token_report_${QWEN_MODEL}.md.
echo ""
echo "########## PHASE 1.5: CoT token-length analysis — $QWEN_MODEL ##########"
run python scripts/08_cot_token_analysis.py --model "$QWEN_MODEL" --datasets "${DATASETS[@]}"

# --- Phase 2: commit & push --------------------------------------------------
echo ""
echo "########## PHASE 2: git add / commit / push ##########"
run git add -A
run git commit -m "qwen3-1.7b (thinking on) results"
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
