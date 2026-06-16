#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY is not set. Export it first:"
    echo "  export OPENAI_API_KEY=<your-key>"
    exit 1
fi

# Auto-discover every dataset/model pair under outputs/ and run the monitor on each.
# Layout: outputs/{dataset}/{model}/{hinted,unhinted}.jsonl
for ds_dir in outputs/*/; do
    ds=$(basename "$ds_dir")
    for m_dir in "$ds_dir"*/; do
        m=$(basename "$m_dir")
        [ -f "$m_dir/hinted.jsonl" ] && [ -f "$m_dir/unhinted.jsonl" ] || continue
        echo "=== monitor: dataset=${ds} model=${m} ==="
        python scripts/03_run_monitor.py --model "${m}" --dataset "${ds}"
    done
done

echo "All monitors done."
