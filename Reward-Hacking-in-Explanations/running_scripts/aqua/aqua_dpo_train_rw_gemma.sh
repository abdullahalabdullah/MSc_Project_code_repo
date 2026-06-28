#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1

#SBATCH -o outputs/aqua/qwen3-1.7B/dpo/training/log_train_llama.txt

DPO_DATA_PATH="outputs/aqua/qwen3-1.7B/dpo"
SAVE_PATH="checkpoints/qwen3-1.7B/aqua_dpo"

# Hub namespace (token owner). Each run is pushed to <token-owner>/${HUB_PREFIX}-...
HUB_PREFIX="aqua-dpo-qwen3-1.7B"

#MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL="Qwen/Qwen3-1.7B"

date '+[%H:%M:%S-%d/%m/%y]'

# Activate env
PROJECT_HOME=${PWD}
source ${PROJECT_HOME}/my_env/bin/activate

# Log W&B locally (no account / no interactive prompt that would hang a Runpod run).
# Use `wandb sync` later to upload, or inspect the local run files.
export WANDB_MODE=offline

for SEED in 0; do

    DPO_DATA_PATH_SEED=${DPO_DATA_PATH}/seed_${SEED}/data
    SAVE_PATH_SEED=${SAVE_PATH}/seed_${SEED}

    for RW in sk-llama_original; do

        echo -e "\n\n--- ${RW} ---\n\n"

        HUB_MODEL_ID="${HUB_PREFIX}-${RW}-seed-${SEED}"

        # Some hparams inspired by:
        # https://github.com/huggingface/alignment-handbook/blob/main/recipes/zephyr-7b-beta/dpo/config_qlora.yaml
        # https://github.com/huggingface/alignment-handbook/blob/main/recipes/zephyr-7b-gemma/dpo/config_full.yaml
        # https://www.philschmid.de/rl-with-llms-in-2025-dpo

        python -u ${PROJECT_HOME}/train_dpo.py \
            --dataset_path ${DPO_DATA_PATH_SEED}/cot_negative_${RW} \
            --model_name_or_path ${MODEL} \
            --output_dir ${SAVE_PATH_SEED}/cot_negative_${RW} \
            --run_name dpo_negative_${RW} \
            --num_train_epochs 1 \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps 16 \
            --per_device_eval_batch_size 1 \
            --gradient_checkpointing \
            --optim="paged_adamw_32bit" \
            --learning_rate 5.0e-6 \
            --weight_decay 0.01 \
            --lr_scheduler_type cosine \
            --warmup_ratio 0.1 \
            --max_length 10240 \
            --logging_steps 10 \
            --bf16 True \
            --tf32 True \
            --eval_strategy steps \
            --save_strategy steps \
            --save_only_model \
            --no_remove_unused_columns \
            --beta 0.1 \
            --seed ${SEED} \
            --run_name aqua_dpo_negative_${RW}_seed-${SEED} \
            --push_to_hub \
            --hub_model_id ${HUB_MODEL_ID} \
            --hub_strategy every_save \
            --hub_private_repo True \
            --use_peft \
            --lora_alpha 32 \
            --lora_r 16 \
            --lora_dropout 0.05 \
            --lora_target_modules "q_proj" "k_proj" "v_proj" "o_proj" "gate_proj" "up_proj" "down_proj"

        date '+[%H:%M:%S-%d/%m/%y]'
    done
done
