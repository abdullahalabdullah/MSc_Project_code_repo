#!/usr/bin/env bash
# End-to-end runner for the harmlessness preference-pair pipeline + Beaver sanity check.
# All stages run in the same environment (see requirements.txt).
#
# Each invocation writes to a NEW timestamped folder under outputs/ (e.g.
# outputs/run_20260703_141530), so a new run never overrides an old one. Every stage
# shares the same folder via the exported RUN_OUTPUT_DIR.
#
# Usage:
#   ./run_all.sh                       # new outputs/run_<timestamp>/ , uses `python`
#   RUN_OUTPUT_DIR=outputs/my_run ./run_all.sh   # write to a specific folder instead
#   PY=python3 ./run_all.sh
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-python}"

# Fresh per-run output folder unless the caller pinned one.
if [ -z "${RUN_OUTPUT_DIR:-}" ]; then
    RUN_OUTPUT_DIR="outputs/run_$(date +%Y%m%d_%H%M%S)"
fi
export RUN_OUTPUT_DIR
mkdir -p "$RUN_OUTPUT_DIR"
echo "== Output folder for this run: $RUN_OUTPUT_DIR =="

echo "== Stage 01: prepare prompts =="
$PY 01_prepare_prompts.py

echo "== Stage 02: generate responses (GPU, vLLM) =="
$PY 02_generate_responses.py

echo "== Stage 03: classify (Qwen3Guard) + filter-1 (GPU) =="
$PY 03_classify_guard.py

echo "== Stage 04: Beaver cost scoring (GPU) =="
$PY 04_score_cost.py

echo "== Stage 05: filter-3 + reports =="
$PY 05_report.py

echo "== Sanity: Beaver ranker check (Beaver + WildGuard) =="
$PY sanity_beaver.py

echo "== Done. Results in: $RUN_OUTPUT_DIR =="
echo "   - $RUN_OUTPUT_DIR/report.md"
echo "   - $RUN_OUTPUT_DIR/beaver_sanity_report.md"
