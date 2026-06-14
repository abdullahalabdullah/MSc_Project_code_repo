#!/bin/bash
#SBATCH --job-name=dpo_harmlessness
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

source ~/.bashrc
conda activate ro2_env          # or your venv activation command

# Run from the MVP-RO2 project root so relative paths (configs/, checkpoints/)
# and the .env file resolve correctly. Under sbatch, $0 points to a spooled copy
# of this script, so dirname "$0" would cd into the spool dir — use the submit
# directory instead (submit with `cd MVP-RO2 && sbatch train_harmlessness.sh`).
# Falls back to dirname "$0" when the script is run directly (e.g. for testing).
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" || exit 1

python train_dpo.py \
    --config configs/harmlessness.yaml

echo "Job finished: $(date)"
