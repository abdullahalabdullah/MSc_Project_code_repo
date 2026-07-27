#!/usr/bin/env bash
# Phase 1 (GPU box): stages 01 (sampling), 02 (OOD eval file), 03 (Filter-0 generation).
# 01/02 are cheap CPU/network steps run here so the GPU session is self-contained; both
# are cached no-ops when already done.
#
# Usage:  X=500 bash run_phase1_gpu.sh outputs/pilot
#         X=20000 [DROPPED_TEMPLATES=T7] bash run_phase1_gpu.sh outputs/full_run
# Afterwards move the run dir to the API box:  python backup_run_to_hf.py --push <run_dir>
set -euo pipefail
cd "$(dirname "$0")"

if [ $# -lt 1 ]; then
    echo "Usage: X=<pool size> bash run_phase1_gpu.sh <run_dir>"; exit 1
fi
export RUN_OUTPUT_DIR="$1"
export PYTHONUNBUFFERED=1
PY="${PY:-python}"
mkdir -p "$RUN_OUTPUT_DIR"
exec > >(tee -a "$RUN_OUTPUT_DIR/run_all.log") 2>&1

FORCE_FLAG=""
if [ "${FORCE:-0}" = "1" ]; then FORCE_FLAG="--force"; fi

echo "== Phase 1 (GPU) started: $(date '+%Y-%m-%d %H:%M:%S') | $RUN_OUTPUT_DIR | X=${X:-5} =="
echo "== Stage 01: sample eval slice + training pool (CPU) =="
$PY 01_sample_pool.py $FORCE_FLAG
echo "== Stage 02: build OOD eval file (CPU) =="
$PY 02_build_eval_ood.py $FORCE_FLAG
echo "== Stage 03: Filter-0 generation (GPU, vLLM) =="
$PY 03_filter0_generate.py $FORCE_FLAG
echo "== Phase 1 done. Next: backup_run_to_hf.py --push $RUN_OUTPUT_DIR, then"
echo "   run_phase2_api.sh on the API box. =="
