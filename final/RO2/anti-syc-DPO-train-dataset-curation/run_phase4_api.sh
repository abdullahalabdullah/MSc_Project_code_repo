#!/usr/bin/env bash
# Phase 4 (API box, NO GPU): stages 08 (Filter-1 grading), 09 (pairs + balancing), then
# either the pilot tail (11: QA + gates) or the full tail (11 QA, 12 export).
#
# Usage:
#   X=500 PILOT=1 X_FULL=20000 bash run_phase4_api.sh outputs/pilot
#   X=20000 bash run_phase4_api.sh outputs/full_run
set -euo pipefail
cd "$(dirname "$0")"

if [ $# -lt 1 ]; then
    echo "Usage: X=<pool size> [PILOT=1 X_FULL=20000] bash run_phase4_api.sh <run_dir>"; exit 1
fi
export RUN_OUTPUT_DIR="$1"
export PYTHONUNBUFFERED=1
PY="${PY:-python}"
mkdir -p "$RUN_OUTPUT_DIR"
exec > >(tee -a "$RUN_OUTPUT_DIR/run_all.log") 2>&1

FORCE_FLAG=""
if [ "${FORCE:-0}" = "1" ]; then FORCE_FLAG="--force"; fi

$PY - <<'PYEOF'
import os, sys
try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass
if not os.environ.get("OPENAI_API_KEY"):
    sys.exit("ERROR: OPENAI_API_KEY not set. Set it in .env.")
print("OpenAI key present.")
PYEOF

echo "== Phase 4 (API) started: $(date '+%Y-%m-%d %H:%M:%S') | $RUN_OUTPUT_DIR | PILOT=${PILOT:-0} =="
echo "== Stage 08: Filter-1 grading (API) =="
$PY 08_filter1_grade.py $FORCE_FLAG
echo "== Stage 09: pair selection + balancing (CPU) =="
$PY 09_pairs.py $FORCE_FLAG
if [ "${PILOT:-0}" = "1" ]; then
    echo "== Stage 11: QA report + manifest + pilot gates =="
    $PY 11_qa_report.py
    echo "== PILOT done. Gates: $RUN_OUTPUT_DIR/pilot_gates.json =="
    echo "== Gate 1 needs the hand audit:  python audit_grader_sample.py --make $RUN_OUTPUT_DIR =="
else
    echo "== Stage 11: QA report + manifest =="
    $PY 11_qa_report.py
    echo "== Stage 12: export DatasetDict =="
    $PY 12_export_dpo_dataset.py $FORCE_FLAG
    echo "== Done. DPO dataset: $RUN_OUTPUT_DIR/dpo_dataset =="
fi
