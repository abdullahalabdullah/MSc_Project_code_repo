#!/usr/bin/env bash
# Run BOTH the pilot (run_all.sh: stages 01-05) and the Beaver ranker sanity check
# (sanity_beaver.py), sharing ONE fresh timestamped output folder so the sanity check
# reads the same run's data. A new folder each time — never overrides an old run.
#
# Usage:
#   ./run_pilot_and_sanity.sh
#   RUN_OUTPUT_DIR=outputs/my_run ./run_pilot_and_sanity.sh   # pin a specific folder
#   PY=python3 ./run_pilot_and_sanity.sh
set -euo pipefail
cd "$(dirname "$0")"

export PY="${PY:-python}"

# One shared run folder for both steps (created here so run_all.sh reuses it).
if [ -z "${RUN_OUTPUT_DIR:-}" ]; then
    export RUN_OUTPUT_DIR="outputs/run_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$RUN_OUTPUT_DIR"
echo "== Run folder (pilot + sanity): $RUN_OUTPUT_DIR =="

# 1) Pilot (stages 01-05) — writes prompts, generations, labels, pairs, report.md
./run_all.sh

# 2) Beaver ranker sanity check — reuses 01/03 outputs, writes beaver_sanity_report.md
echo "== Sanity: Beaver ranker check (Beaver + WildGuard) =="
$PY sanity_beaver.py

echo "== All done. Results in: $RUN_OUTPUT_DIR =="
echo "   - $RUN_OUTPUT_DIR/report.md"
echo "   - $RUN_OUTPUT_DIR/beaver_sanity_report.md"
echo "   - $RUN_OUTPUT_DIR/beaver_sanity_disagreements.jsonl"
