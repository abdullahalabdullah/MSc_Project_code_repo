#!/usr/bin/env bash
set -euo pipefail

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY is not set." >&2
    exit 1
fi

MODEL="qwen3_1_7b"

for ds in mmlu gpqa agieval wmdp; do
    echo "=== Running monitor: ${MODEL} / ${ds} ==="
    python scripts/03_run_monitor.py --model "${MODEL}" --dataset "${ds}"
done

echo "=== Monitor complete for ${MODEL} ==="
