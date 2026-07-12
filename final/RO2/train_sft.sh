#!/usr/bin/env bash
#
# One-command SFT ablation arm: data prep + training (SFT_Arm_Developer_Spec.md).
#
# Runs the full METHOD=sft workflow, skipping anything already done:
#   1. Convert  outputs/<run>/dpo_dataset -> outputs/<run>_sft/sft_dataset
#      (convert_dpo_to_sft.py — the SFT arm gets its OWN run folder, never sharing the
#      DPO run's). SKIPPED if the sft_dataset already exists (provenance.json is the marker).
#   2. Verify   tokenization equality vs the DPO pathway (verify_sft_tokenization.py)
#      SKIPPED if verification already PASSED for this exact dataset content — a
#      .verify_passed marker inside sft_dataset stores the dataset hash it passed with,
#      so any regenerated/modified dataset is re-verified automatically.
#   3. Train    METHOD=sft bash train.sh <sft_dataset>
#
# Usage:
#   bash train_sft.sh                          # newest harmlessness outputs/*/dpo_dataset
#   bash train_sft.sh path/to/dpo_dataset      # explicit source dataset
#   PREP_ONLY=1 bash train_sft.sh              # convert+verify only (e.g. on a CPU box), no training
#
# All train.sh env knobs pass through unchanged (LEARNING_RATE, EPOCHS, EVAL_FRACTION,
# HUB_MODEL_ID, REPORT_TO, ...). Run inside tmux for long jobs: tmux new -s train
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Locate the source DPO dataset ───────────────────────────────────────────
DPO_DATASET="${1:-$(ls -dt harmlessness-DPO-train-dataset-curation/outputs/*/dpo_dataset 2>/dev/null | head -1 || true)}"
if [ -z "${DPO_DATASET}" ] || [ ! -d "${DPO_DATASET}" ]; then
  echo "ERROR: no exported dpo_dataset found."
  echo "  Run the curation pipeline first:  bash harmlessness-DPO-train-dataset-curation/run_all.sh"
  echo "  or pass an explicit path:          bash train_sft.sh path/to/dpo_dataset"
  exit 1
fi
# Own run folder for the SFT arm: outputs/<run>/dpo_dataset -> outputs/<run>_sft/sft_dataset
RUN_DIR="$(dirname "${DPO_DATASET}")"
SFT_DATASET="$(dirname "${RUN_DIR}")/$(basename "${RUN_DIR}")_sft/sft_dataset"
echo "== Source (DPO): ${DPO_DATASET}"
echo "== Target (SFT): ${SFT_DATASET}"

# ── 1. Convert (skip if already converted) ──────────────────────────────────
if [ -f "${SFT_DATASET}/provenance.json" ]; then
  echo "== Convert: SKIP — ${SFT_DATASET} already exists (provenance.json present)"
else
  echo "== Convert: dpo_dataset -> sft_dataset"
  python convert_dpo_to_sft.py "${DPO_DATASET}" --output "${SFT_DATASET}"
fi

# ── 2. Verify (skip only if this exact dataset content already passed) ──────
# The marker stores the derived_sha256 the verification passed with; convert_dpo_to_sft.py
# records the same hash in provenance.json, so a regenerated dataset invalidates the marker.
MARKER="${SFT_DATASET}/.verify_passed"
CURRENT_SHA="$(python - "${SFT_DATASET}" <<'PY'
import json, sys, os
with open(os.path.join(sys.argv[1], "provenance.json"), encoding="utf-8") as f:
    print(json.load(f)["derived_sha256"])
PY
)"
if [ -f "${MARKER}" ] && [ "$(cat "${MARKER}")" = "${CURRENT_SHA}" ]; then
  echo "== Verify: SKIP — tokenization equality already PASSED for this dataset (${CURRENT_SHA:0:12}…)"
else
  echo "== Verify: tokenization equality vs the DPO pathway (go/no-go gate)"
  python verify_sft_tokenization.py --dpo_dataset "${DPO_DATASET}" --sft_dataset "${SFT_DATASET}"
  printf '%s' "${CURRENT_SHA}" > "${MARKER}"   # only reached on PASS (verifier exits non-zero on failure)
fi

# ── 3. Train ────────────────────────────────────────────────────────────────
if [ "${PREP_ONLY:-0}" = "1" ]; then
  echo "== PREP_ONLY=1: data prep complete, skipping training."
  echo "   Train later with:  bash train_sft.sh ${DPO_DATASET}"
  exit 0
fi
METHOD=sft exec bash train.sh "${SFT_DATASET}"
