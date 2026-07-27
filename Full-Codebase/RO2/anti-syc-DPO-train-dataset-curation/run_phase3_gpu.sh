#!/usr/bin/env bash
# Phase 3 (GPU box): stage 07 (Filter-1 generation) only.
#
# Usage:  X=500 bash run_phase3_gpu.sh outputs/pilot
#         X=20000 N_FILTER1=10 bash run_phase3_gpu.sh outputs/full_run
set -euo pipefail
cd "$(dirname "$0")"

if [ $# -lt 1 ]; then
    echo "Usage: X=<pool size> bash run_phase3_gpu.sh <run_dir>"; exit 1
fi
export RUN_OUTPUT_DIR="$1"
export PYTHONUNBUFFERED=1
PY="${PY:-python}"
mkdir -p "$RUN_OUTPUT_DIR"
exec > >(tee -a "$RUN_OUTPUT_DIR/run_all.log") 2>&1

FORCE_FLAG=""
if [ "${FORCE:-0}" = "1" ]; then FORCE_FLAG="--force"; fi

echo "== Phase 3 (GPU) started: $(date '+%Y-%m-%d %H:%M:%S') | $RUN_OUTPUT_DIR | N_FILTER1=${N_FILTER1:-10} =="
echo "== Stage 07: Filter-1 generation (GPU, vLLM) =="
$PY 07_filter1_generate.py $FORCE_FLAG
echo "== Phase 3 done. Next: backup_run_to_hf.py --push $RUN_OUTPUT_DIR, then"
echo "   run_phase4_api.sh on the API box. =="
