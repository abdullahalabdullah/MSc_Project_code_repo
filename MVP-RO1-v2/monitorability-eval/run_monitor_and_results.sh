#!/usr/bin/env bash
#
# run_monitor_and_results.sh — monitor + results phase. Run AFTER
# run_inference.sh has produced the hinted/unhinted outputs (pull first if
# inference ran on another box). No GPU needed; the monitor uses the OpenAI API.
#
# Usage:
#   export OPENAI_API_KEY=sk-...
#   bash monitorability-eval/run_monitor_and_results.sh <model_id> <dataset|d1,d2|ALL> <hint_type|ALL>
#
#   model_id   the SAME model ID used with run_inference.sh (resolves the same dirs).
#   dataset    a dataset key from config/mvp_config.yaml, a comma-separated subset
#              (e.g. mmlu_pro,lexam), or ALL for every key.
#   hint_type  a hint type, or ALL for every type.
#
# Examples:
#   bash monitorability-eval/run_monitor_and_results.sh Qwen/Qwen3-4B ALL professor
#   bash monitorability-eval/run_monitor_and_results.sh Qwen/Qwen3-4B mmlu_pro,lexam professor
#
# Per (hint type x dataset): monitor (03), g-mean^2 metrics (04), plots (05),
# Excel export (06); then one cross-dataset summary table (07) per hint type.
# Fail-soft; commits + pushes the results at the end; logs to logs/ (gitignored).

set -uo pipefail

# cd to the repo root (this script lives in monitorability-eval/) so the shared
# config/, outputs-and-results/, and logs/ resolve as repo-root-relative paths.
cd "$(dirname "$0")/.."

USAGE="usage: bash monitorability-eval/run_monitor_and_results.sh <model_id> <dataset|d1,d2|ALL> <hint_type|ALL>"
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
    # A single key or a comma-separated subset, e.g. mmlu_pro,lexam.
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

# --- Preflight: API key ---------------------------------------------------------
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY is not set. Run:  export OPENAI_API_KEY=<your-key>"
    exit 1
fi

# --- Preflight: inference outputs must exist -------------------------------------
echo "Checking inference outputs for $MODEL_ID ..."
missing=0
for hint in "${HINT_TYPES[@]}"; do
    for ds in "${DATASETS[@]}"; do
        for arm in hinted unhinted; do
            f="outputs-and-results/outputs/${MODEL_NAME}/${ds}/${hint}/${arm}.jsonl"
            if [[ -f "$f" ]]; then
                echo "  OK  $f"
            else
                echo "  MISSING  $f"
                missing=$((missing + 1))
            fi
        done
    done
done
if [[ $missing -gt 0 ]]; then
    echo "ERROR: $missing inference file(s) missing — run run_inference.sh first (and pull)."
    exit 1
fi

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
LOGFILE="logs/monitor_results_${MODEL_NAME}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "Monitor + results run started: $(date)"
echo "Model:      $MODEL_ID (dirs namespaced as '$MODEL_NAME')"
echo "Datasets:   ${DATASETS[*]}"
echo "Hint types: ${HINT_TYPES[*]}"
echo "Log file:   $LOGFILE"

# --- Phase 0: git identity ---------------------------------------------------
echo ""
echo "########## PHASE 0: git identity ##########"
run git config --global user.email "abd@gmail.com"
run git config --global user.name "Abd"

# --- Phase 1: monitor ----------------------------------------------------------
for hint in "${HINT_TYPES[@]}"; do
    echo ""
    echo "########## PHASE 1: monitor ($hint) ##########"
    for ds in "${DATASETS[@]}"; do
        run python monitorability-eval/scripts/03_run_monitor.py --model "$MODEL_ID" --dataset "$ds" --hint-type "$hint"
    done
done

# --- Phase 2: results (g-mean^2 + plots + excel) --------------------------------
for hint in "${HINT_TYPES[@]}"; do
    echo ""
    echo "########## PHASE 2: results calculation ($hint) ##########"
    for ds in "${DATASETS[@]}"; do
        run python monitorability-eval/scripts/04_compute_gmean2.py --model "$MODEL_ID" --dataset "$ds" --hint-type "$hint"
        run python monitorability-eval/scripts/05_analyze_gmean2.py --model "$MODEL_ID" --dataset "$ds" --hint-type "$hint"
        run python monitorability-eval/scripts/06_export_excel.py   --model "$MODEL_ID" --dataset "$ds" --hint-type "$hint"
    done
done

# --- Phase 3: Markdown reports ----------------------------------------------------
# Kept as ONE separable step: delete this line (plus scripts/09_generate_reports.py
# and src/reporting.py) to strip all .md report generation for a public release.
echo ""
echo "########## PHASE 3: markdown reports ##########"
run python monitorability-eval/scripts/09_generate_reports.py --model "$MODEL_ID"

commit_push "monitor + results (${HINT_TYPES[*]}) — $MODEL_NAME"

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
