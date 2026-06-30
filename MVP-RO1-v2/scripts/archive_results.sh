#!/usr/bin/env bash
#
# Archive analysis/, outputs/, monitor_outputs/, and plots/ into a timestamped
# folder under Old-Results/, then clear the source directories (except .gitkeep).
#
# Usage:
#   ./scripts/archive_results.sh
#
# Run from the MVP-RO1/ directory (or anywhere — it cd's to the repo root itself).

set -euo pipefail

cd "$(dirname "$0")/.."

TS=$(date +%Y-%m-%d_%H-%M-%S)
DEST="Old-Results/$TS"

mkdir -p "$DEST"/{analysis,outputs,monitor_outputs,plots}

for d in analysis outputs monitor_outputs plots; do
  find "$d" -mindepth 1 -maxdepth 1 ! -name '.gitkeep' -exec mv {} "$DEST/$d/" \;
done

echo "Archived to $DEST"
