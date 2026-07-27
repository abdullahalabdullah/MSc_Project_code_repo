#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "Running monitor for tulu_sft_no_safety..."
python scripts/03_run_monitor.py --model tulu_sft_no_safety

echo "Running monitor for tulu_sft_full..."
python scripts/03_run_monitor.py --model tulu_sft_full

echo "Running monitor for tulu_dpo..."
python scripts/03_run_monitor.py --model tulu_dpo

echo "Running monitor for tulu_final..."
python scripts/03_run_monitor.py --model tulu_final

echo "Running CoT token-length analysis for all Tulu models..."
for model in tulu_sft_no_safety tulu_sft_full tulu_dpo tulu_final; do
    python scripts/08_cot_token_analysis.py --model "${model}" --datasets mmlu
done

echo "All done."