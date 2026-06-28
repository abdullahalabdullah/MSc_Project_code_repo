#!/bin/bash
# Generate base Qwen3-1.7B responses then commit and push the outputs.
# Run on RunPod with 50 GB container disk and vLLM template.

set -e  # exit on any error

PROJECT_HOME=${PWD}

# ── 1. Install git-lfs (needed to push .joblib files correctly) ────────────
if ! git lfs version &>/dev/null; then
    echo "[setup] Installing git-lfs ..."
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
    apt-get install -y git-lfs
fi
git lfs install

# ── 2. Set git identity (required for commits on a fresh pod) ─────────────
git config user.name  "Abdullah Alabdullah"
git config user.email "s2869453@sms.ed.ac.uk"

# ── 3. Generate base model responses ──────────────────────────────────────
echo "[run] Starting base model response generation ..."
bash ${PROJECT_HOME}/running_scripts/aqua/aqua_base_eval_compute.sh 2>&1 | tee logs/base_eval_compute.log

echo "[run] Generation complete."

# ── 4. Commit and push outputs ────────────────────────────────────────────
echo "[git] Staging outputs ..."
git add outputs/aqua/qwen3-1.7B/base/
git add logs/base_eval_compute.log

echo "[git] Committing ..."
git commit -m "Add base Qwen3-1.7B eval responses (seed 0, orig + cf, sampling)"

echo "[git] Pushing ..."
git push

echo "[done] All done."
