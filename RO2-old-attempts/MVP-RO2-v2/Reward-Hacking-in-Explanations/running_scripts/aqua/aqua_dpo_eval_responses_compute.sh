#!/bin/bash
#SBATCH --time=18:00:00
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1

#SBATCH -o outputs/aqua/qwen3-1.7B/dpo/training/log_eval_gen.txt

DATA_PATH="data/aqua/"
SAVE_PATH="checkpoints/qwen3-1.7B/aqua_dpo"
EVAL_OUTPUT_PATH="outputs/aqua/qwen3-1.7B/dpo" # Where the eval outputs (test) with dpo are saved
NUM_TEST=10000

#MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL="Qwen/Qwen3-1.7B"

date '+[%H:%M:%S-%d/%m/%y]'

# Activate env
PROJECT_HOME=${PWD}
source ${PROJECT_HOME}/my_env/bin/activate

for SEED in 0; do
    for APPROACH in sk-llama_original; do

        EVAL_OUTPUT_PATH_SEED=${EVAL_OUTPUT_PATH}/seed_${SEED}/eval/${APPROACH}
        SAVE_PATH_SEED=${SAVE_PATH}/seed_${SEED}

        # Get ckpt path
        CKPT_PATH=${SAVE_PATH_SEED}/cot_negative_${APPROACH}/checkpoint-best
        echo "CKPT_PATH: ${CKPT_PATH}"

        # Merge model
        python ${PROJECT_HOME}/merge_peft_model.py \
            --base_model_name ${MODEL} \
            --adapter_model_name ${CKPT_PATH} \
            --output_name ${CKPT_PATH}_merged    

        date '+[%H:%M:%S-%d/%m/%y]'; echo -e "\n\n"

        # Decode sampling (Majority@16 is the evaluated config; greedy decoding is
        # discouraged for Qwen3 thinking mode and not on this metric's path).
        for SPLIT in orig cf; do
            python get_responses_vllm.py \
                ${CKPT_PATH}_merged \
                ${EVAL_OUTPUT_PATH_SEED} \
                ${DATA_PATH}/cot_negative.joblib \
                ${SPLIT} \
                test \
                16 \
                --suffix _sampling \
                --seed ${SEED} \
                --number_entries_used ${NUM_TEST}
        done
    
        # After decoding everything that is needed, remove merged model
        python delete_checkpoint.py ${CKPT_PATH}_merged

        date '+[%H:%M:%S-%d/%m/%y]'; echo -e "\n\n"
    done
done