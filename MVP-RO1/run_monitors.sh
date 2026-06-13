#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

for model in tulu_sft_full tulu_dpo tulu_final; do
    echo "=== Running monitor for ${model} ==="
    python scripts/03_run_monitor.py --model "${model}"
done
