#!/usr/bin/env bash
#
# Full-run automation for the Qwen3-1.7B DPO reward-hacking replication.
#
#   pre-flight checks (HF token, git push)
#     -> prepare -> train -> eval-compute -> parse-test   (each logged to logs/)
#     -> commit + push the whole repo (incl. outputs/ + logs/) to GitHub
#
# The OpenAI judge (eval_responses.py) is NOT run here — it needs no GPU, so run it
# LOCALLY after pulling the repo (instructions printed at the end). This script only
# parses the test responses so the judge's input (*_data.joblib) is ready + pushed.
#
# Run it INSIDE tmux so an SSH drop doesn't kill the multi-hour job:
#     tmux new -s run
#     bash run_full.sh
#     # detach: Ctrl-b then d ;  reattach: tmux attach -t run
#
set -euo pipefail

# ----------------------------- config -----------------------------
NUM_EXAMPLES=2000        # full-run training-prompt count
EPOCHS=3                 # paper uses 5; 1 risks under-training. 3 = compromise.
COMMIT_MSG="Full run: Qwen3-1.7B DPO (seed 0, ${EPOCHS} epochs) + outputs/logs"

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source my_env/bin/activate
mkdir -p logs

bold() { printf "\n\033[1m%s\033[0m\n" "$*"; }
die()  { printf "\033[31mFAILED:\033[0m %s\n" "$*"; exit 1; }

# ------------------------- pre-flight checks ----------------------
bold "==================== PRE-FLIGHT CHECKS ===================="

printf "[1/2] Hugging Face token ... "
python -c "import env_setup; env_setup.setup_env(False); from huggingface_hub import whoami; print('OK as', whoami()['name'])" \
  || die "HF token invalid. env_setup.py hardcodes HF_TOKEN — check it."

printf "[2/2] git add/commit/push capability ... "
git rev-parse --is-inside-work-tree >/dev/null 2>&1 \
  || die "not a git repo. Run: git init && git remote add origin <YOUR repo> && git add -A && git commit -m init"
git remote get-url origin >/dev/null 2>&1 \
  || die "no 'origin' remote. Run: git remote add origin <YOUR writable repo URL>"
git push --dry-run origin HEAD >/dev/null 2>&1 \
  && echo "OK ($(git remote get-url origin))" \
  || die "cannot push to $(git remote get-url origin). Ensure it's YOUR repo and credentials (PAT/SSH) are set."

# Git LFS so large *.joblib outputs don't hit GitHub's 100MB/file limit.
printf "      git-lfs ... "
command -v git-lfs >/dev/null 2>&1 || { echo "installing"; apt-get update -qq && apt-get install -y -qq git-lfs >/dev/null 2>&1 || echo "(warn: could not install git-lfs; large joblibs may be rejected)"; }
git lfs install >/dev/null 2>&1 || true
git lfs track "outputs/**/*.joblib" >/dev/null 2>&1 || true
command -v git-lfs >/dev/null 2>&1 && echo "ready" || true

echo "All pre-flight checks passed."

# ------------------- configure full-run sizes ---------------------
bold "==================== CONFIGURE (NUM_EXAMPLES=${NUM_EXAMPLES}, EPOCHS=${EPOCHS}) ===================="
sed -i "s/^NUM_EXAMPLES=.*/NUM_EXAMPLES=${NUM_EXAMPLES}/" running_scripts/aqua/aqua_dpo_prepare_dataset.sh
sed -i "s/--num_train_epochs [0-9.]*/--num_train_epochs ${EPOCHS}/" running_scripts/aqua/aqua_dpo_train_rw_gemma.sh
grep -nE "^NUM_EXAMPLES=" running_scripts/aqua/aqua_dpo_prepare_dataset.sh
grep -nE "num_train_epochs" running_scripts/aqua/aqua_dpo_train_rw_gemma.sh

# ----------------------------- pipeline ---------------------------
bold "==================== [1/4] PREPARE ===================="
bash running_scripts/aqua/aqua_dpo_prepare_dataset.sh        2>&1 | tee logs/full_prepare.log

bold "==================== [2/4] TRAIN ===================="
bash running_scripts/aqua/aqua_dpo_train_rw_gemma.sh         2>&1 | tee logs/full_train.log

bold "==================== [3/4] EVAL — COMPUTE (generate test responses) ===================="
bash running_scripts/aqua/aqua_dpo_eval_responses_compute.sh 2>&1 | tee logs/full_eval_compute.log

bold "==================== [4/4] PARSE TEST RESPONSES (judge runs locally, off-GPU) ===================="
# The OpenAI judge needs no GPU, so we don't run it here. We only PARSE the test responses
# so the judge's input (*_test_sampling_data.joblib) exists and gets pushed to GitHub.
EVAL_DIR="outputs/aqua/qwen3-1.7B/dpo/seed_0/eval/sk-llama_original"
for SPLIT in orig cf; do
  python get_responses_data.py "${EVAL_DIR}/cot_negative_${SPLIT}_test_sampling.joblib" aqua --seed 0 2>&1 | tee -a logs/full_eval_parse.log
done

bold "Pipeline complete (GPU stages done)."

# -------------------------- commit + push -------------------------
bold "==================== GIT COMMIT + PUSH (full repo incl. outputs/) ===================="
git add -A
if git diff --cached --quiet; then
  echo "Nothing to commit."
else
  git commit -m "${COMMIT_MSG}"
  git push origin HEAD && echo "Pushed to $(git remote get-url origin)" || die "git push failed (see above — likely file-size/LFS or auth)."
fi

bold "DONE (GPU stages complete)."
echo "Adapter is on HF (pushed every half epoch). Outputs/logs are committed to GitHub,"
echo "including the parsed test responses the judge needs:"
echo "  outputs/aqua/qwen3-1.7B/dpo/seed_0/eval/sk-llama_original/cot_negative_{orig,cf}_test_sampling_data.joblib"
echo ""
echo "NEXT — run the OpenAI judge LOCALLY (no GPU needed). On your Mac:"
echo "  git pull                          # get outputs/ from GitHub"
echo "  pip install openai                # if not already"
echo "  export OPENAI_API_KEY=sk-..."
echo "  for S in orig cf; do"
echo "    python eval_responses.py outputs/aqua/qwen3-1.7B/dpo/seed_0/eval/sk-llama_original/cot_negative_\${S}_test_sampling_data.joblib --judge_model gpt-4o --seed 0"
echo "  done"
echo "(eval_responses.py updates those _data.joblib in place, adding pf_ack. Then run the analysis/ scripts.)"
echo ""
echo "Tip: you can now terminate the pod once the GitHub push above succeeded."
