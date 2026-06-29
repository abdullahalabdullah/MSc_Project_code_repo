#!/bin/bash
# Judge base Qwen3-1.7B responses (no GPU needed — API calls only).
# Run this locally after aqua_base_eval_compute.sh has finished on RunPod
# and the output joblib files have been pulled back.
#
# Resume-safe: skips get_responses_data if _data.joblib already exists,
# and skips already-judged responses inside eval_responses.py.

BASE_OUTPUT_PATH="outputs/aqua/qwen3-1.7B/base"
JUDGE_MODEL="gpt-5-mini"
MAX_WORKERS=16
SEED=0

date '+[%H:%M:%S-%d/%m/%y]'

PROJECT_HOME=${PWD}
source ${PROJECT_HOME}/my_env/bin/activate

BASE_EVAL_PATH=${BASE_OUTPUT_PATH}/seed_${SEED}/eval

for SPLIT in orig cf; do
    DATA_FILE="${BASE_EVAL_PATH}/cot_negative_${SPLIT}_test_sampling_data.joblib"

    if [ ! -f "${DATA_FILE}" ]; then
        python ${PROJECT_HOME}/get_responses_data.py \
            ${BASE_EVAL_PATH}/cot_negative_${SPLIT}_test_sampling.joblib \
            aqua \
            --seed ${SEED}
    else
        echo "Skipping get_responses_data (${DATA_FILE} already exists)"
    fi

    python ${PROJECT_HOME}/eval_responses.py \
        ${DATA_FILE} \
        --judge_model ${JUDGE_MODEL} \
        --max_workers ${MAX_WORKERS} \
        --seed ${SEED}

    date '+[%H:%M:%S-%d/%m/%y]'; echo -e "\n\n"
done
