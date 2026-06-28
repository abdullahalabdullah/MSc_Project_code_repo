#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=2

#SBATCH -o outputs/aqua/qwen3-1.7B/dpo/training/log_eval_ev.txt


DATA_PATH="data/aqua/"
EVAL_OUTPUT_PATH="outputs/aqua/qwen3-1.7B/dpo" # Where the eval outputs (test) with dpo are saved
NUM_TEST=10000

# Acknowledgment judge is an OpenAI(-compatible) API model (no local GPU needed).
# Requires OPENAI_API_KEY in the environment (and OPENAI_BASE_URL if self-hosted).
JUDGE_MODEL="gpt-5-mini"

date '+[%H:%M:%S-%d/%m/%y]'

# Activate env
PROJECT_HOME=${PWD}
source ${PROJECT_HOME}/my_env/bin/activate

for SEED in 0; do
    for APPROACH in sk-llama_original; do

        EVAL_OUTPUT_PATH_SEED=${EVAL_OUTPUT_PATH}/seed_${SEED}/eval/${APPROACH}

        for DECODING_APPROACH in sampling; do
            for SPLIT in orig cf; do
                DATA_FILE="${EVAL_OUTPUT_PATH_SEED}/cot_negative_${SPLIT}_test_${DECODING_APPROACH}_data.joblib"

                # Only (re)parse raw responses if the _data.joblib does not yet exist.
                # On a resume run the file already exists (possibly with judged entries),
                # so we skip this step to avoid resetting pf_ack back to None.
                if [ ! -f "${DATA_FILE}" ]; then
                    python ${PROJECT_HOME}/get_responses_data.py \
                        ${EVAL_OUTPUT_PATH_SEED}/cot_negative_${SPLIT}_test_${DECODING_APPROACH}.joblib \
                        aqua \
                        --seed ${SEED}
                else
                    echo "Skipping get_responses_data (${DATA_FILE} already exists)"
                fi

                python ${PROJECT_HOME}/eval_responses.py \
                    ${DATA_FILE} \
                    --judge_model ${JUDGE_MODEL} \
                    --seed ${SEED}

                date '+[%H:%M:%S-%d/%m/%y]'; echo -e "\n\n"
            done
        done
    done
done