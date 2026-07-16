#!/usr/bin/env bash
# Full anti-sycophancy curation pipeline (stages 01-12) on ONE box that has both a GPU
# and OPENAI_API_KEY — intended for the X=5 smoke test, or for running everything in a
# single GPU rental. For split GPU/API execution use run_phase{1..4}_{gpu,api}.sh.
#
# Every stage is cached (skipped when already done under the same config); re-running
# this script resumes wherever it stopped. Pass FORCE=1 to re-run all stages.
#
# Usage:
#   X=5 RUN_OUTPUT_DIR=outputs/smoke bash run_all.sh     # smoke test
#   X=500 PILOT=1 RUN_OUTPUT_DIR=outputs/pilot bash run_all.sh
#   X=20000 bash run_all.sh                              # full run, fresh timestamped dir
#   N_FILTER1=16 ... bash run_all.sh                     # owner-approved escalation only
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-python}"
export PYTHONUNBUFFERED=1

git config --global user.email "abd@example.com" 2>/dev/null || true
git config --global user.name "abd" 2>/dev/null || true

if [ -z "${RUN_OUTPUT_DIR:-}" ]; then
    RUN_OUTPUT_DIR="outputs/run_$(date +%Y%m%d_%H%M%S)"
fi
export RUN_OUTPUT_DIR
mkdir -p "$RUN_OUTPUT_DIR"
exec > >(tee -a "$RUN_OUTPUT_DIR/run_all.log") 2>&1

FORCE_FLAG=""
if [ "${FORCE:-0}" = "1" ]; then FORCE_FLAG="--force"; fi

echo "== Run started: $(date '+%Y-%m-%d %H:%M:%S') (log: $RUN_OUTPUT_DIR/run_all.log) =="
echo "== Output: $RUN_OUTPUT_DIR | X=${X:-5} | N_FILTER1=${N_FILTER1:-10} | PILOT=${PILOT:-0} =="

# Preflight: the API stages need OPENAI_API_KEY — fail before any GPU time.
$PY - <<'PYEOF'
import os, sys
try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass
if not os.environ.get("OPENAI_API_KEY"):
    sys.exit("ERROR: OPENAI_API_KEY not set (stages 04/05/08 need it). Set it in .env.")
print("OpenAI key present.")
PYEOF

echo "== Stage 01: sample eval slice + training pool (CPU) =="
$PY 01_sample_pool.py $FORCE_FLAG
echo "== Stage 02: build OOD eval file (CPU) =="
$PY 02_build_eval_ood.py $FORCE_FLAG
echo "== Stage 03: Filter-0 generation (GPU, vLLM) =="
$PY 03_filter0_generate.py $FORCE_FLAG
echo "== Stage 04: Filter-0 grading (API) =="
$PY 04_filter0_grade.py $FORCE_FLAG
echo "== Stage 05: wrong answers + freeze eval slice (API) =="
$PY 05_wrong_answers.py $FORCE_FLAG
echo "== Stage 06: template assignment (CPU) =="
$PY 06_templates.py $FORCE_FLAG
echo "== Stage 07: Filter-1 generation (GPU, vLLM) =="
$PY 07_filter1_generate.py $FORCE_FLAG
echo "== Stage 08: Filter-1 grading (API) =="
$PY 08_filter1_grade.py $FORCE_FLAG
echo "== Stage 09: pair selection + balancing (CPU) =="
$PY 09_pairs.py $FORCE_FLAG
if [ "${PILOT:-0}" = "1" ]; then
    echo "== PILOT: skipping stage 12 (export) =="
    echo "== Stage 11: QA report + manifest + pilot gates =="
    $PY 11_qa_report.py
else
    echo "== Stage 11: QA report + manifest =="
    $PY 11_qa_report.py
    echo "== Stage 12: export DatasetDict =="
    $PY 12_export_dpo_dataset.py $FORCE_FLAG
fi

echo "== Done. QA: $RUN_OUTPUT_DIR/qa_report.md | manifest: $RUN_OUTPUT_DIR/manifest.json =="
if [ "${PILOT:-0}" != "1" ]; then
    echo "== DPO dataset: $RUN_OUTPUT_DIR/dpo_dataset =="
fi
