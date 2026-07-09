#!/usr/bin/env bash
#
# run_results_analysis.sh — results phase AFTER the monitor (03) has produced
# monitor_results.jsonl. Runs, per dataset:
#   04  g-mean^2 metrics   -> results_per_instance.csv/json, results_summary.json
#   06  Excel workbook     -> qualitative_export.xlsx
#   05  plots (optional)   -> plots/.../ *.png
# then once for the whole model:
#   08  CoT token stats    -> cot_token_stats.json   (feeds 09)
#   09  Markdown reports    -> results_*/cot_* .md
#
# These are local file-readers: no OPENAI_API_KEY, no config edits, no GPU.
#
# Usage:
#   bash monitorability-eval/run_results_analysis.sh [dataset ...]
#
#   With no args, runs the three datasets below. Pass dataset keys to override:
#     bash monitorability-eval/run_results_analysis.sh mmlu_pro lexam
#
# Env overrides:
#   MODEL_ID    inference model id    (default: Qwen/Qwen3-4B)
#   HINT_TYPE   hint type             (default: professor)
#   SKIP_PLOTS  set to 1 to skip step 05 plots
#   SKIP_REPORTS set to 1 to skip steps 08 + 09

set -uo pipefail

# Repo root = parent of monitorability-eval/ (this script's dir).
cd "$(dirname "$0")/.."

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B}"
HINT_TYPE="${HINT_TYPE:-professor}"
SKIP_PLOTS="${SKIP_PLOTS:-0}"
SKIP_REPORTS="${SKIP_REPORTS:-0}"
S=monitorability-eval/scripts

# Datasets: CLI args if given, else the three present for this model.
if [[ $# -gt 0 ]]; then
    DATASETS=("$@")
else
    DATASETS=(mmlu_pro lexam medxpertqa)
fi

echo ">> model=${MODEL_ID}  hint=${HINT_TYPE}  datasets=${DATASETS[*]}"
echo ">> skip_plots=${SKIP_PLOTS}  skip_reports=${SKIP_REPORTS}"
echo

rc=0
run_step() {  # run_step <label> <cmd...>
    local label="$1"; shift
    echo "   -- $label"
    if ! "$@"; then
        echo "   !! FAILED: $label" >&2
        rc=1
    fi
}

# Per-dataset: metrics (04) -> excel (06) -> plots (05).
for d in "${DATASETS[@]}"; do
    echo "==================================================================="
    echo ">> Dataset: $d"
    echo "==================================================================="
    run_step "04 g-mean^2 metrics" \
        python "$S/04_compute_gmean2.py" --model "$MODEL_ID" --dataset "$d" --hint-type "$HINT_TYPE"
    run_step "06 Excel export" \
        python "$S/06_export_excel.py" --model "$MODEL_ID" --dataset "$d" --hint-type "$HINT_TYPE"
    if [[ "$SKIP_PLOTS" != "1" ]]; then
        run_step "05 plots" \
            python "$S/05_analyze_gmean2.py" --model "$MODEL_ID" --dataset "$d" --hint-type "$HINT_TYPE"
    fi
    echo
done

# Whole-model: CoT token stats (08) then Markdown reports (09).
if [[ "$SKIP_REPORTS" != "1" ]]; then
    echo "==================================================================="
    echo ">> Whole-model reports"
    echo "==================================================================="
    run_step "08 CoT token stats" \
        python "$S/08_cot_token_analysis.py" --model "$MODEL_ID" --datasets "${DATASETS[@]}" --hint-type "$HINT_TYPE"
    run_step "09 Markdown reports" \
        python "$S/09_generate_reports.py" --model "$MODEL_ID"
    echo
fi

MODEL_DIR="$(echo "$MODEL_ID" | tr '/' '_')"
if [[ $rc -eq 0 ]]; then
    echo ">> Done. Artifacts under:"
    echo "     outputs-and-results/monitorability-results/${MODEL_DIR}/<dataset>/${HINT_TYPE}/   (metrics, .xlsx, .md)"
    echo "     outputs-and-results/plots/${MODEL_DIR}/<dataset>/${HINT_TYPE}/                    (.png)"
else
    echo ">> Finished with one or more failures (see above)." >&2
fi
exit $rc
