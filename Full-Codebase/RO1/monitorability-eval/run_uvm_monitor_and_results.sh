#!/usr/bin/env bash
#
# run_uvm_monitor_and_results.sh — CPU phase of the uncertainty-vs-monitorability
# experiment (config/uncertainty_vs_monitorability.yaml). Run locally AFTER
# run_uvm_inference.sh has pushed the inference outputs.
#
# Usage (from anywhere; the script cd's to Full-Codebase/RO1):
#   export OPENAI_API_KEY=sk-...
#   bash monitorability-eval/run_uvm_monitor_and_results.sh [dataset ...]
#   (default dataset: lexam)
#
# Steps: git pull -> per dataset: 03 monitor, 04 g-mean², 05 plots, 06 Excel
# -> 59 entropy analysis -> 60 bias-variance / error-incoherence, across the
# given datasets. The monitor (03) resumes
# from monitor_progress.json, so an interrupted run is continued by simply
# rerunning this script. Stops at the first failing step. Nothing is committed
# or pushed; output is mirrored to logs/.

set -uo pipefail
cd "$(dirname "$0")/.."

C=config/uncertainty_vs_monitorability.yaml
MODEL=Qwen/Qwen3-4B
HINT=grader_hacking
DATASETS=("$@")
[[ ${#DATASETS[@]} -eq 0 ]] && DATASETS=(lexam)
O=outputs-and-results/uncertainty-vs-monitorability/inference-outputs/${MODEL//\//_}

mkdir -p logs
LOG="logs/uvm_monitor_results_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "uncertainty-vs-monitorability monitor + results — $(date)"
echo "Datasets: ${DATASETS[*]}   Log: $LOG"

fail() { echo "FAILED: $*"; exit 1; }
step() {
    echo ""; echo "=== $* ==="
    "$@" || fail "$* (rerun this script to resume)"
}

# --- Preflight ----------------------------------------------------------------
[[ -n "${OPENAI_API_KEY:-}" ]] || fail "OPENAI_API_KEY is not set — export OPENAI_API_KEY=sk-..."
python -c "import yaml, openai, numpy, pandas, matplotlib, scipy, openpyxl" 2>/dev/null \
    || fail "python deps missing — run: python -m pip install openai pyyaml numpy pandas matplotlib scipy openpyxl"
# one tiny request exactly as the monitor makes it (same model, same API, same
# server: OPENAI_BASE_URL if set, e.g. an OpenAI-compatible university gateway)
MON_MODEL=$(python -c "import yaml;print(yaml.safe_load(open('$C'))['monitor']['model'])")
python - "$MON_MODEL" <<'EOF' || fail "monitor API check failed for key '${OPENAI_API_KEY:0:8}…' at ${OPENAI_BASE_URL:-api.openai.com} (error above)"
import sys
from openai import OpenAI
try:
    OpenAI().chat.completions.create(model=sys.argv[1],
                                     messages=[{"role": "user", "content": "Reply OK."}])
except Exception as e:
    print(f"  {type(e).__name__}: {str(e)[:300]}")
    sys.exit(1)
EOF
echo "Monitor API OK: model $MON_MODEL at ${OPENAI_BASE_URL:-api.openai.com}"

echo ""; echo "=== git pull ==="
git pull --ff-only || fail "git pull (commit or stash local changes to the same files, then rerun)"

for ds in "${DATASETS[@]}"; do
    ls "$O/$ds/$HINT"/hinted.jsonl* "$O/$ds/$HINT"/unhinted.jsonl* >/dev/null 2>&1 \
        || fail "no inference outputs for $ds under $O/$ds/$HINT — was it pushed from the pod?"
done
echo "Preflight OK (OPENAI_API_KEY, deps, inference outputs present)."

# --- Monitor (repeated until complete) + metrics, per dataset ------------------
# 03 skips a call that still fails after its retries (e.g. HTTP 429 rate limits)
# without recording it, so one pass can leave gaps. Each pass re-loads and only
# calls what is missing; its "N remaining" line is the count BEFORE that pass,
# so a pass reporting 0 remaining means the monitor is complete.
MAX_PASSES=${UVM_MAX_PASSES:-10}
PAUSE=${UVM_PASS_PAUSE:-60}
monitor_until_complete() {  # $1 = dataset
    local pass out remaining
    for ((pass = 1; pass <= MAX_PASSES; pass++)); do
        echo ""; echo "=== monitor $1 — pass $pass/$MAX_PASSES ==="
        out=$(mktemp)
        python monitorability-eval/scripts/03_run_monitor.py --config $C --model $MODEL \
            --dataset "$1" --hint-type $HINT 2>&1 | tee "$out"
        [[ ${PIPESTATUS[0]} -eq 0 ]] || { rm -f "$out"; fail "03_run_monitor.py crashed for $1 (rerun this script to resume)"; }
        remaining=$(grep -oE '[0-9]+ remaining' "$out" | head -1 | grep -oE '^[0-9]+')
        rm -f "$out"
        [[ -n "$remaining" ]] || fail "could not read the remaining-call count from 03's output for $1"
        if [[ "$remaining" -eq 0 ]]; then
            echo "Monitor complete for $1 (0 calls remaining)."
            return 0
        fi
        echo "Pass $pass started with $remaining calls to make; checking again after a ${PAUSE}s pause (rate-limit cool-down) ..."
        sleep "$PAUSE"
    done
    fail "monitor for $1 still incomplete after $MAX_PASSES passes — rerun this script later (completed calls are kept)"
}

for ds in "${DATASETS[@]}"; do
    monitor_until_complete "$ds"
    for s in 04_compute_gmean2 05_analyze_gmean2 06_export_excel; do
        step python monitorability-eval/scripts/$s.py --config $C --model $MODEL \
            --dataset "$ds" --hint-type $HINT
    done
done

# --- Entropy vs monitorability ------------------------------------------------
step python monitorability-eval/scripts/59_uvm_gmean2_by_answer_entropy.py \
    --config $C --model $MODEL --hint-type $HINT --datasets "${DATASETS[@]}"

# --- Bias-variance / error-incoherence (Hot Mess of AI) vs monitorability -----
step python monitorability-eval/scripts/60_uvm_bias_variance_incoherence.py \
    --config $C --model $MODEL --hint-type $HINT --datasets "${DATASETS[@]}"

R=outputs-and-results/uncertainty-vs-monitorability
echo ""
echo "Done — $(date)."
echo "Plot:     $R/plots/${MODEL//\//_}/gmean2_by_answer_entropy_$HINT.png"
echo "Tables:   $R/monitorability-results/${MODEL//\//_}/entropy_{per_question,bins}_$HINT.csv"
echo "Log:      $LOG"
