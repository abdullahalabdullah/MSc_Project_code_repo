#!/usr/bin/env bash
# End-to-end runner for the harmlessness preference-pair pipeline.
# All five stages run in the same environment (see requirements.txt).
#
# Usage:
#   ./run_all.sh              # uses `python` on PATH
#   PY=python3 ./run_all.sh
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-python}"

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

echo "== Done. See outputs/report.md =="
