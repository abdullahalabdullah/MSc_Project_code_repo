#!/usr/bin/env bash
#
# run_pbm_monitor_and_results.sh — CPU phase of the prior-belief-vs-monitorability
# experiment (config/prior_belief_vs_monitorability.yaml). Run locally AFTER
# run_pbm_inference.sh has pushed the inference outputs.
#
# Usage (from anywhere; the script cd's to Full-Codebase/RO1):
#   export OPENAI_API_KEY=...            (and OPENAI_BASE_URL=... for a gateway such as ELM)
#   bash monitorability-eval/run_pbm_monitor_and_results.sh
#
# Steps: git pull -> per version (lexam_priorA, lexam_priorB): 03 monitor
# (unchanged; repeated until no call is left — it skips calls that exhaust
# their retries, e.g. HTTP 429) and 04 g-mean² (unchanged) -> 63 analysis.
# Each version's unhinted runs are monitored against that version's target
# (fix 3), which is why the shared fresh unhinted runs are monitored twice.
# Stops at the first failing step; a rerun resumes the monitor where it
# stopped. Nothing is committed or pushed; output is mirrored to logs/.

set -uo pipefail
cd "$(dirname "$0")/.."

C=config/prior_belief_vs_monitorability.yaml
MODEL=Qwen/Qwen3-4B
HINT=grader_hacking
VERSIONS=(lexam_priorA lexam_priorB)
O=outputs-and-results/prior-belief-vs-monitorability/inference-outputs/${MODEL//\//_}

mkdir -p logs
LOG="logs/pbm_monitor_results_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "prior-belief-vs-monitorability monitor + results — $(date)"
echo "Log: $LOG"

fail() { echo "FAILED: $*"; exit 1; }
step() {
    echo ""; echo "=== $* ==="
    "$@" || fail "$* (rerun this script to resume)"
}

# --- Preflight ----------------------------------------------------------------
[[ -n "${OPENAI_API_KEY:-}" ]] || fail "OPENAI_API_KEY is not set"
python -c "import yaml, openai, numpy, pandas, matplotlib, scipy" 2>/dev/null \
    || fail "python deps missing — run: python -m pip install openai pyyaml numpy pandas matplotlib scipy"
# one tiny request exactly as the monitor makes it (same model / API / server)
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
for v in "${VERSIONS[@]}"; do
    ls "$O/$v/$HINT"/hinted.jsonl* "$O/$v/$HINT"/unhinted.jsonl* >/dev/null 2>&1 \
        || fail "no inference outputs for $v under $O/$v/$HINT — was it pushed from the pod?"
done
echo "Preflight OK (API, deps, inference outputs present)."

# --- Monitor (repeated until complete) + g-mean², per version ---------------------
MAX_PASSES=${PBM_MAX_PASSES:-10}
PAUSE=${PBM_PASS_PAUSE:-60}
monitor_until_complete() {  # $1 = version dataset key
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

for v in "${VERSIONS[@]}"; do
    monitor_until_complete "$v"
    step python monitorability-eval/scripts/04_compute_gmean2.py --config $C --model $MODEL \
        --dataset "$v" --hint-type $HINT
done

# --- Analysis --------------------------------------------------------------------
step python monitorability-eval/scripts/63_pbm_analysis.py --config $C --model $MODEL

R=outputs-and-results/prior-belief-vs-monitorability
echo ""
echo "Done — $(date)."
echo "Report:   $R/monitorability-results/${MODEL//\//_}/pbm_summary.md"
echo "Figures:  $R/plots/${MODEL//\//_}/pbm_{paired_gmean2,gmean2_by_p0,paired_ack_rate}.png"
echo "Log:      $LOG"
