#!/usr/bin/env bash
#
# run_aqua_dpo_qwen3_1_7b_s0.sh — merge DPO adapter into Qwen3-1.7B, run RO1
# inference over MMLU/GPQA/WMDP/AGIEval (70 examples, N=4, thinking ON).
#
# Adapter:  er-abd/aqua-dpo-qwen3-1.7B-sk-llama_original-seed-0
# Base:     Qwen/Qwen3-1.7B
# Merged:   er-abd/aqua-dpo-qwen3-1.7B-sk-llama_original-seed-0-merged
# Config:   aqua_dpo_qwen3_1_7b_s0  (config/mvp_config.yaml)
#
# Order:
#   0. git identity (safe on fresh pod).
#   1. Merge adapter + push merged model to Hub.
#   2. Inference over each dataset, committing + pushing after each.
#   3. CoT token-length analysis.
#
# Fail-soft: each command is logged; failures are skipped, not fatal.
#
# Usage:
#   bash run_aqua_dpo_qwen3_1_7b_s0.sh

set -uo pipefail

cd "$(dirname "$0")"

# --- Configuration ----------------------------------------------------------
DATASETS=(mmlu gpqa wmdp agieval)
MODEL="aqua_dpo_qwen3_1_7b_s0"
CONFIG_BLOCK="qwen_inference"

BASE_MODEL="Qwen/Qwen3-1.7B"
ADAPTER="er-abd/aqua-dpo-qwen3-1.7B-sk-llama_original-seed-0"
MERGED_REPO="er-abd/aqua-dpo-qwen3-1.7B-sk-llama_original-seed-0-merged"
OUT_DIR="./merged_aqua_dpo_qwen3_1_7b_s0"

LOGFILE="run_aqua_dpo_qwen3_1_7b_s0_$(date +%Y%m%d_%H%M%S).log"

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

exec > >(tee -a "$LOGFILE") 2>&1

echo "aqua-dpo Qwen3-1.7B s0 run started: $(date)"
echo "Datasets: ${DATASETS[*]}"
echo "Model:    $MODEL"
echo "Log file: $LOGFILE"

# --- Phase 0: git identity ---------------------------------------------------
echo ""
echo "########## PHASE 0: git identity ##########"
run git config --global user.email "abd@gmail.com"
run git config --global user.name "Abd"

# --- commit_push helper ------------------------------------------------------
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

# --- Pre-flight: verify dataset indexes exist (no recreation) ----------------
# Datasets are already prepared under data/{ds}/dataset_index.json from a
# previous run of 01_prepare_dataset.py. This script never calls that script;
# it aborts here if any index is missing rather than silently recreating it.
echo ""
echo "########## PRE-FLIGHT: checking dataset indexes ##########"
missing=0
for ds in "${DATASETS[@]}"; do
    index="data/${ds}/dataset_index.json"
    if [[ -f "$index" ]]; then
        echo "  OK  $index"
    else
        echo "  MISSING  $index  — run 01_prepare_dataset.py --dataset $ds first"
        missing=$((missing + 1))
    fi
done
if [[ $missing -gt 0 ]]; then
    echo "ERROR: $missing dataset index(es) missing. Aborting."
    exit 1
fi

# --- Phase 1: merge adapter + push ------------------------------------------
echo ""
echo "########## PHASE 1: merge adapter + push ##########"
run pip install -q peft
run python scripts/merge_and_push.py \
    --base        "$BASE_MODEL" \
    --adapter     "$ADAPTER" \
    --merged-repo "$MERGED_REPO" \
    --out-dir     "$OUT_DIR"

# Abort if merge failed — inference will 404 without the merged repo on Hub.
if [[ ! -d "$OUT_DIR" ]]; then
    echo "ERROR: merge failed (out-dir $OUT_DIR not created). Aborting."
    exit 1
fi

# --- Phase 2: inference (commit + push after each dataset) ------------------
echo ""
echo "########## PHASE 2: inference — $MODEL ##########"
for ds in "${DATASETS[@]}"; do
    run python scripts/02_run_inference_qwen.py \
        --model        "$MODEL" \
        --dataset      "$ds" \
        --config-block "$CONFIG_BLOCK"
    commit_push "aqua-dpo qwen3-1.7b s0 (thinking on) results — $ds"
done

# --- Phase 3: CoT token-length analysis -------------------------------------
echo ""
echo "########## PHASE 3: CoT token-length analysis — $MODEL ##########"
run python scripts/08_cot_token_analysis.py --model "$MODEL" --datasets "${DATASETS[@]}"
commit_push "aqua-dpo qwen3-1.7b s0 — CoT token-length analysis"

# --- Summary ----------------------------------------------------------------
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
