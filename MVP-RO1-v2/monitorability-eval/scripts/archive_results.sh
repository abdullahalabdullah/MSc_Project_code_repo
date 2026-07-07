#!/usr/bin/env bash
#
# Snapshot the result dirs (monitorability-results, inference-outputs, monitor_outputs,
# plots, harmlessness-results) under outputs-and-results/ into a timestamped
# outputs-and-results/Old-Results/ folder, then clear the source dirs (except
# .gitkeep) for a fresh run.
#
# Usage:
#   ./monitorability-eval/scripts/archive_results.sh
#
# Run from anywhere — it cd's to the repo root itself.

set -euo pipefail

# This script lives in monitorability-eval/scripts/, so climb two levels to the repo root.
cd "$(dirname "$0")/../.."

ROOT="outputs-and-results"
TS=$(date +%Y-%m-%d_%H-%M-%S)
DEST="$ROOT/Old-Results/$TS"

mkdir -p "$DEST"/{monitorability-results,inference-outputs,monitor_outputs,plots,harmlessness-results}

for d in monitorability-results inference-outputs monitor_outputs plots harmlessness-results; do
  [ -d "$ROOT/$d" ] || continue
  find "$ROOT/$d" -mindepth 1 -maxdepth 1 ! -name '.gitkeep' -exec mv {} "$DEST/$d/" \;
done

echo "Archived to $DEST"
