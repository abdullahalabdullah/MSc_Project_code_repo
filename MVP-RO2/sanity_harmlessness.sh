#!/bin/bash
#SBATCH --job-name=dpo_harmlessness_sanity
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --time=01:00:00
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
# directory instead (submit with `cd MVP-RO2 && sbatch sanity_harmlessness.sh`).
# Falls back to dirname "$0" when the script is run directly (e.g. for testing).
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" || exit 1

# Sanity check: same pipeline as the full run, but on the tiny dry-run config
# (subsample 60 / val 10 / max_steps 50). Reads configs/harmlessness.yaml
# overlaid with configs/harmlessness_sanity.yaml internally — no --config arg.
python sanity_check_dpo.py

echo "Job finished: $(date)"
