#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

for model in tulu_sft_full tulu_dpo tulu_final; do
    echo "=== Running monitor for ${model} ==="
    python scripts/03_run_monitor.py --model "${model}"

    echo "=== CoT token-length analysis for ${model} ==="
    python scripts/08_cot_token_analysis.py --model "${model}" --datasets mmlu
done
