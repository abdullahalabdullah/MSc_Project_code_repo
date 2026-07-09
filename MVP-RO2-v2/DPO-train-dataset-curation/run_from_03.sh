#!/usr/bin/env bash
# Resume the curation pipeline from stage 03, reusing an existing run's stage-02 output.
# Use this when 01/02 (prompt prep + the expensive vLLM generation) are already done and you
# only need classification -> scoring -> reports -> export. Stages 03 and 04 are resumable
# (checkpoint files), so re-running this after an interrupt continues where it stopped.
#
# Usage:
#   RUN_OUTPUT_DIR=outputs/main ./run_from_03.sh    # default folder is outputs/main
#   PY=python3 RUN_OUTPUT_DIR=outputs/main ./run_from_03.sh
set -euo pipefail

# Run from the module dir regardless of where this is invoked from.
cd "$(dirname "$0")"

PY="${PY:-python}"
export PYTHONUNBUFFERED=1          # unbuffered stdout so progress lines aren't hidden by the tee pipe
# Reduce CUDA caching-allocator fragmentation from the variable-length (length-sorted) batches
# in stage 04 — lets it reuse freed blocks across differently-shaped batches.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# filter-4 minimum margin — leave empty for harmlessness (matches run_all.sh default).
MIN_MARGIN=""
export MIN_MARGIN

# The run folder that already contains stage-02 output. Must match the original run.
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-outputs/main}"
export RUN_OUTPUT_DIR

# Fail fast (before loading any GPU model) if we're pointed at the wrong / empty folder.
if [ ! -f "$RUN_OUTPUT_DIR/02_responses.joblib" ]; then
    echo "ERROR: $RUN_OUTPUT_DIR/02_responses.joblib not found — set RUN_OUTPUT_DIR to the run"
    echo "       folder that already holds stage-02 output (e.g. RUN_OUTPUT_DIR=outputs/main)." >&2
    exit 1
fi

# Mirror all output into the run folder (append), so the log survives a dropped SSH session.
exec > >(tee -a "$RUN_OUTPUT_DIR/run_from_03.log") 2>&1
echo "== Resume from stage 03: $(date '+%Y-%m-%d %H:%M:%S') =="
echo "== Output folder: $RUN_OUTPUT_DIR (reusing existing 02_responses.joblib) =="
if [ -n "$MIN_MARGIN" ]; then
    echo "== filter-4 minimum margin: $MIN_MARGIN =="
else
    echo "== filter-4 minimum margin: not applied =="
fi

# Stages 03/04 are the expensive GPU steps; skip one whose final output already exists so a
# re-run resumes the pipeline instead of redoing finished work (set FORCE=1 to recompute).
if [ -f "$RUN_OUTPUT_DIR/03_labeled.joblib" ] && [ -z "${FORCE:-}" ]; then
    echo "== Stage 03: SKIP — $RUN_OUTPUT_DIR/03_labeled.joblib already exists (FORCE=1 to redo) =="
else
    echo "== Stage 03: classify (Qwen3Guard, vLLM) + filter-1 (GPU) — resumable =="
    $PY 03_classify_guard.py
fi

if [ -f "$RUN_OUTPUT_DIR/04_scored.joblib" ] && [ -z "${FORCE:-}" ]; then
    echo "== Stage 04: SKIP — $RUN_OUTPUT_DIR/04_scored.joblib already exists (FORCE=1 to redo) =="
else
    echo "== Stage 04: Beaver cost scoring (GPU) — resumable =="
    $PY 04_score_cost.py
fi

echo "== Stage 05: filter-3 + reports =="
$PY 05_report.py

echo "== Stage 06: export training-ready DPO dataset (HF DatasetDict) =="
$PY 06_export_dpo_dataset.py

echo "== Done. Curation stats: $RUN_OUTPUT_DIR/report.md + $RUN_OUTPUT_DIR/diagnostics.json =="
echo "== DPO dataset (saved locally, reusable by train.sh): $RUN_OUTPUT_DIR/dpo_dataset =="
