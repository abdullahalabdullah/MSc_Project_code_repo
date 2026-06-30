#!/usr/bin/env bash
#
# Snapshot the result dirs (analysis, outputs, monitor_outputs, plots) under
# outputs-and-results/ into a timestamped outputs-and-results/Old-Results/ folder,
# then clear the source dirs (except .gitkeep) for a fresh run.
#
# Usage:
#   ./scripts/archive_results.sh
#
# Run from anywhere — it cd's to the repo root itself.

set -euo pipefail

cd "$(dirname "$0")/.."

ROOT="outputs-and-results"
TS=$(date +%Y-%m-%d_%H-%M-%S)
DEST="$ROOT/Old-Results/$TS"

mkdir -p "$DEST"/{analysis,outputs,monitor_outputs,plots}

for d in analysis outputs monitor_outputs plots; do
  find "$ROOT/$d" -mindepth 1 -maxdepth 1 ! -name '.gitkeep' -exec mv {} "$DEST/$d/" \;
done

echo "Archived to $DEST"
