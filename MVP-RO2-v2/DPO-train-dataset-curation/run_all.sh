#!/usr/bin/env bash
# Pilot runner for the harmlessness preference-pair pipeline (stages 01-06, ending in the
# training-ready dpo_dataset export).
# (The Beaver ranker sanity check is separate: sanity_beaver.py — or use run_pilot_and_sanity.sh
#  to do both.)
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

# ── Optional minimum-margin filter (filter-4) ───────────────────────────────
# EDIT THIS VALUE to turn the filter on. Leave EMPTY ("") = filter not applied (default).
# Set to a number (e.g. "2.0") to drop final pairs whose cost margin
# (rejected_cost - chosen_cost) is below it. For harmlessness leave it empty (filter-3
# already gates quality); use it for the helpfulness pipeline.
MIN_MARGIN=""
export MIN_MARGIN

# Fresh per-run output folder unless the caller pinned one.
if [ -z "${RUN_OUTPUT_DIR:-}" ]; then
    RUN_OUTPUT_DIR="outputs/run_$(date +%Y%m%d_%H%M%S)"
fi
export RUN_OUTPUT_DIR
mkdir -p "$RUN_OUTPUT_DIR"
echo "== Output folder for this run: $RUN_OUTPUT_DIR =="
if [ -n "$MIN_MARGIN" ]; then
    echo "== filter-4 minimum margin: $MIN_MARGIN =="
else
    echo "== filter-4 minimum margin: not applied =="
fi

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

echo "== Stage 06: export training-ready DPO dataset (HF DatasetDict) =="
$PY 06_export_dpo_dataset.py

echo "== Pilot done. Results in: $RUN_OUTPUT_DIR/report.md =="
echo "== DPO dataset (handoff): $RUN_OUTPUT_DIR/dpo_dataset =="
