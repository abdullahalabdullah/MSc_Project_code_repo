#!/bin/bash
#SBATCH --time=32:00:00
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1

#SBATCH -o outputs/aqua/qwen3-1.7B/dpo/log_data.txt

# --------------------------------------------------------

# Decide what to run
RUN_RESPONSES=true
RUN_EVAL_RESPONSES=true
RUN_REWARD=true
RUN_PREPARE_DATA=true

# Args
SAVE_PATH="outputs/aqua/qwen3-1.7B/dpo/"
DATA_PATH="data/aqua/"
NUM_EXAMPLES=2000
# Number of responses sampled per prompt to build DPO chosen/rejected pairs.
N_SAMPLES=8

#MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL="Qwen/Qwen3-1.7B"

# Single-config replication: DPO + RM annotated by SK-Llama-8B only.
REWARD_MODEL_2="Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"

date '+[%H:%M:%S-%d/%m/%y]'

# Activate env
PROJECT_HOME=${PWD}
source ${PROJECT_HOME}/my_env/bin/activate

for SEED in 0; do

    echo -e "\nSEED: ${SEED}\n"

    SAVE_PATH_SEED=${SAVE_PATH}/seed_${SEED}/data

    # --- Get responses ---
    if ${RUN_RESPONSES}; then
        echo -e "---------- Running responses ...\n"
        # Only the original prompt is needed for the original-RM DPO preference set.
        for SPLIT in orig; do
            for DATA in valid train; do
                python ${PROJECT_HOME}/get_responses_vllm.py \
                    ${MODEL} \
                    ${SAVE_PATH_SEED} \
                    ${DATA_PATH}/cot_negative.joblib \
                    ${SPLIT} \
                    ${DATA} \
                    ${N_SAMPLES} \
                    --seed ${SEED} \
                    --number_entries_used ${NUM_EXAMPLES}
                date '+[%H:%M:%S-%d/%m/%y]'; echo -e "\n\n\n"
            done
        done
    fi

    # --- Eval responses ---
    if ${RUN_EVAL_RESPONSES}; then
        echo -e "---------- Running eval responses ...\n"
            for SPLIT in orig; do
                for DATA in valid train; do
                    # We don't run the ack step in this case
                    python ${PROJECT_HOME}/get_responses_data.py \
                        ${SAVE_PATH_SEED}/cot_negative_${SPLIT}_${DATA}.joblib \
                        aqua \
                        --seed ${SEED}
                    date '+[%H:%M:%S-%d/%m/%y]'; echo -e "\n\n"
                done
            done
    fi

    # --- Rewards ---
    # Score the original responses with SK-Llama only (no augmented scoring, since
    # the original-RM DPO set does not need RM_D / RM_C).
    if ${RUN_REWARD}; then
        RW_MODEL=${REWARD_MODEL_2}
        # We strip <think> before scoring, so the RM sees short visible answers ->
        # a large batch fits easily on an 80GB GPU.
        BSIZE=16
        for DATA in valid train; do
            echo -e "---------- Running rewards ...\n"
            python -u ${PROJECT_HOME}/get_rewards_data.py \
                ${SAVE_PATH_SEED}/cot_negative_orig_${DATA}_data.joblib \
                ${RW_MODEL} \
                --batch_size ${BSIZE} \
                --seed ${SEED}
            date '+[%H:%M:%S-%d/%m/%y]'; echo -e "\n\n"
        done
    fi

    # --- Prepare DPO data ---
    if ${RUN_PREPARE_DATA}; then
        echo -e "---------- Running prepare data ...\n"
        for RW in sk-llama_original; do
            python dpo_prepare_dataset.py \
                ${SAVE_PATH_SEED}/cot_negative_orig_train_data_${RW}.joblib \
                ${SAVE_PATH_SEED}/cot_negative_orig_valid_data_${RW}.joblib \
                --seed ${SEED}
            date '+[%H:%M:%S-%d/%m/%y]'; echo -e "\n\n"
        done
    fi
done