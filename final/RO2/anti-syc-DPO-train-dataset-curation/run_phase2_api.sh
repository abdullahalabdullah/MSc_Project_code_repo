#!/usr/bin/env bash
# Phase 2 (API box, NO GPU): stages 04 (Filter-0 grading), 05 (wrong answers + eval-slice
# freeze), 06 (template assignment).
#
# Usage:  X=500 bash run_phase2_api.sh outputs/pilot
# Afterwards move the run dir back to a GPU box for phase 3.
set -euo pipefail
cd "$(dirname "$0")"

if [ $# -lt 1 ]; then
    echo "Usage: X=<pool size> bash run_phase2_api.sh <run_dir>"; exit 1
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
    # Explicit paths: parameterless load_dotenv() inspects the caller's file,
    # which does not exist for stdin scripts (heredoc) and crashes. cwd is
    # this module dir, so check here first, then the RO2 root .env.
    from dotenv import load_dotenv
    load_dotenv(".env")
    load_dotenv("../.env")
except Exception:
    pass
if not os.environ.get("OPENAI_API_KEY"):
    sys.exit("ERROR: OPENAI_API_KEY not set. Set it in .env.")
print("OpenAI key present.")
PYEOF

echo "== Phase 2 (API) started: $(date '+%Y-%m-%d %H:%M:%S') | $RUN_OUTPUT_DIR =="
echo "== Stage 04: Filter-0 grading (API) =="
$PY 04_filter0_grade.py $FORCE_FLAG
echo "== Stage 05: wrong answers + freeze eval slice (API) =="
$PY 05_wrong_answers.py $FORCE_FLAG
echo "== Stage 06: template assignment (CPU) =="
$PY 06_templates.py $FORCE_FLAG
echo "== Phase 2 done. Next: run_phase3_gpu.sh on a GPU box. =="
