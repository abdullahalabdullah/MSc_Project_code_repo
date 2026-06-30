#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY is not set. Export it first:"
    echo "  export OPENAI_API_KEY=<your-key>"
    exit 1
fi

# Datasets to monitor.
DATASETS=(mmlu)

# Run the monitor for every model that has outputs under each dataset.
# All datasets are namespaced: outputs/{dataset}/{model}/{hinted,unhinted}.jsonl
for ds in "${DATASETS[@]}"; do
    base="outputs/${ds}"

    for m_dir in "${base}"/*/; do
        m=$(basename "$m_dir")
        [ -f "${m_dir}/hinted.jsonl" ] && [ -f "${m_dir}/unhinted.jsonl" ] || continue
        echo "=== monitor: dataset=${ds} model=${m} ==="
        python scripts/03_run_monitor.py --model "${m}" --dataset "${ds}"
    done
done

echo "All monitors done."
