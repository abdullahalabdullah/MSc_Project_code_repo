#!/bin/bash
#SBATCH --time=18:00:00
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1

#SBATCH -o outputs/aqua/qwen3-1.7B/base/log_eval_gen.txt

# Generate base Qwen3-1.7B responses on the AQUA test set (orig + cf splits).
# No adapter loading or merging needed — uses the HF base model directly.
# Run this on RunPod (GPU required for vLLM). After this completes,
# run aqua_base_eval_eval.sh locally to judge the responses.

DATA_PATH="data/aqua/"
BASE_OUTPUT_PATH="outputs/aqua/qwen3-1.7B/base"
NUM_TEST=10000
MODEL="Qwen/Qwen3-1.7B"
SEED=0

date '+[%H:%M:%S-%d/%m/%y]'

PROJECT_HOME=${PWD}
source ${PROJECT_HOME}/my_env/bin/activate

BASE_EVAL_PATH=${BASE_OUTPUT_PATH}/seed_${SEED}/eval

for SPLIT in orig cf; do
    python ${PROJECT_HOME}/get_responses_vllm.py \
        ${MODEL} \
        ${BASE_EVAL_PATH} \
        ${DATA_PATH}/cot_negative.joblib \
        ${SPLIT} \
        test \
        16 \
        --suffix _sampling \
        --seed ${SEED} \
        --number_entries_used ${NUM_TEST}

    date '+[%H:%M:%S-%d/%m/%y]'; echo -e "\n\n"
done
