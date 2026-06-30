#!/usr/bin/env bash
set -euo pipefail

if [ -z "${HF_HOME:-}" ]; then
    echo "ERROR: HF_HOME is not set." >&2
    exit 1
fi

python scripts/02_run_inference.py --model tulu_sft_full
rm -rf "$HF_HOME/hub/models--allenai--Llama-3.1-Tulu-3-8B-SFT"

python scripts/02_run_inference.py --model tulu_dpo
rm -rf "$HF_HOME/hub/models--allenai--Llama-3.1-Tulu-3-8B-DPO"

python scripts/02_run_inference.py --model tulu_final
rm -rf "$HF_HOME/hub/models--allenai--Llama-3.1-Tulu-3-8B"

# CoT token-length analysis (overall + per-dataset + length buckets) per model.
for model in tulu_sft_full tulu_dpo tulu_final; do
    python scripts/08_cot_token_analysis.py --model "${model}" --datasets mmlu
done