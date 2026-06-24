#!/usr/bin/env bash
# =============================================================================
# RO2 MVP — Llama-3.2-3B-Instruct harmlessness SFT, end-to-end launcher
# (RunPod / single GPU).
# =============================================================================
# Usage (from the MVP-RO2 project root, inside the pod):
#   bash run_sft.sh                 # preflight -> sanity gate -> train -> push
#   bash run_sft.sh --skip-sanity   # skip the sanity gate (after it once passed)
#   bash run_sft.sh --train-only    # just the full training run
#
# Prereqs: .env present (HF_TOKEN), `pip install -r requirements.txt`, and the
# Llama-3.2 license accepted on HF for the token's account (gated model). See
# documentation.md.
# =============================================================================
set -euo pipefail

CONFIG="configs/sft_harmlessness.yaml"
SKIP_SANITY=0
TRAIN_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --skip-sanity) SKIP_SANITY=1 ;;
    --train-only)  TRAIN_ONLY=1 ;;
    --config=*)    CONFIG="${arg#*=}" ;;
    *) echo "Unknown arg: $arg" >&2; exit 2 ;;
  esac
done

# Run from this script's directory so relative paths (configs/, outputs/) resolve.
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

echo "=== $(date) | config=$CONFIG ==="
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

if [[ "$TRAIN_ONLY" -eq 0 ]]; then
  echo "=== [00] Sequence-length preflight ==="
  python scripts/00_check_seq_length.py --config "$CONFIG"

  if [[ "$SKIP_SANITY" -eq 0 ]]; then
    echo "=== [GATE] Sanity check ==="
    python scripts/sanity_check_sft.py
  fi
fi

echo "=== [01] Full SFT training ==="
python scripts/01_train_sft.py --config "$CONFIG"

echo "=== [02] Push adapters to the Hub ==="
python scripts/02_push_adapters.py --config "$CONFIG"

echo "=== DONE $(date) ==="
echo "Adapters pushed to <hf_username>/<repo_prefix>-epoch{1,2,3,4}."
echo "To merge an epoch into a standalone model for MVP-RO1 eval, e.g.:"
echo "  python scripts/04_merge_and_push.py --config $CONFIG --epoch 4"
