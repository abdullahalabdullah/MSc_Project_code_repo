#!/usr/bin/env bash
#
# run_uvm_inference.sh — GPU phase of the uncertainty-vs-monitorability
# experiment (config/uncertainty_vs_monitorability.yaml): inference (02),
# CoT token stats (08) and the answers / parse report (58), with every saved
# checkpoint gzipped, committed and pushed to GitHub as soon as it lands.
#
# Usage (from anywhere; the script cd's to Full-Codebase/RO1):
#   export HF_TOKEN=hf_...
#   bash monitorability-eval/run_uvm_inference.sh [dataset ...]
#   (default datasets: gpqa_diamond lexam wmdp)
#
# Safety:
#   - Preflight BEFORE the model loads: python deps, HF token, and a
#     `git push --dry-run`, so a missing credential fails in seconds, not
#     after hours of GPU time.
#   - Resume from a fresh clone: plain .jsonl files are gitignored, so only
#     the .gz copies come back with the repo. If a dataset has a .gz but no
#     .jsonl, it is decompressed first, so 02 appends to the full file (and
#     resumes from the committed progress.json) instead of starting a short one.
#   - A background watcher polls every dataset's progress.json; whenever it
#     changes (02 writes it right after a checkpoint's rows are appended), the
#     dataset's hinted/unhinted.jsonl are gzipped and committed together with
#     that progress.json, then pushed. A pod deleted at any point loses at most
#     the checkpoint being generated.

set -uo pipefail
cd "$(dirname "$0")/.."

C=config/uncertainty_vs_monitorability.yaml
MODEL=Qwen/Qwen3-4B
HINT=grader_hacking
DATASETS=("$@")
[[ ${#DATASETS[@]} -eq 0 ]] && DATASETS=(gpqa_diamond lexam wmdp)
OUT=outputs-and-results/uncertainty-vs-monitorability
O=$OUT/inference-outputs/${MODEL//\//_}

mkdir -p logs
LOG="logs/uvm_inference_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "uncertainty-vs-monitorability inference — $(date)"
echo "Datasets: ${DATASETS[*]}   Log: $LOG"

# --- Preflight ----------------------------------------------------------------
fail() { echo "PREFLIGHT FAILED: $*"; exit 1; }
python -c "import yaml, vllm, transformers" 2>/dev/null \
    || fail "python deps missing — run: python -m pip install --no-cache-dir -r requirements.txt"
[[ -n "${HF_TOKEN:-}" ]] || fail "HF_TOKEN is not set — export HF_TOKEN=hf_..."
git config user.name  >/dev/null || git config --global user.name  "Abd"
git config user.email >/dev/null || git config --global user.email "abd@gmail.com"
git push --dry-run >/dev/null 2>&1 \
    || fail "git push does not work — set a token remote: git remote set-url origin https://<user>:<token>@github.com/abdullahalabdullah/MSc_Project_code_repo.git"
echo "Preflight OK (deps, HF_TOKEN, git push)."

# --- Restore .jsonl from committed .gz (resume after a fresh clone) -------------
for ds in "${DATASETS[@]}"; do
    d=$O/$ds/$HINT
    for arm in hinted unhinted; do
        if [[ -f $d/$arm.jsonl.gz && ! -s $d/$arm.jsonl ]]; then
            echo "Restoring $d/$arm.jsonl from .gz"
            gunzip -c "$d/$arm.jsonl.gz" > "$d/$arm.jsonl"
        fi
    done
done

# --- Checkpoint push ------------------------------------------------------------
push_dataset() {  # $1 = dataset, $2 = commit message suffix
    local d=$O/$1/$HINT
    for arm in hinted unhinted; do
        [[ -s $d/$arm.jsonl ]] && gzip -nkf "$d/$arm.jsonl"
    done
    git add "$d"/*.gz "$d/progress.json" 2>/dev/null
    git diff --cached --quiet && return 0
    git commit -q -m "uvm inference N=30 ($HINT) — $1 — $2" \
        && { git push -q && echo "[push] $1: $2" \
             || echo "[push] $1: commit made, PUSH FAILED (retried next checkpoint)"; }
}

watcher() {  # pushes each dataset whenever its progress.json changes
    while [[ ! -f "$STOP" ]]; do
        for ds in "${DATASETS[@]}"; do
            p=$O/$ds/$HINT/progress.json
            [[ -f $p ]] || continue
            n=$(python -c "import json;print(len(json.load(open('$p'))['completed_instance_ids']))" 2>/dev/null) || continue
            seen_f="$STOP.seen.$ds"
            if [[ "$(cat "$seen_f" 2>/dev/null)" != "$n" ]]; then
                sleep 5  # let 02 finish writing
                n=$(python -c "import json;print(len(json.load(open('$p'))['completed_instance_ids']))" 2>/dev/null)
                push_dataset "$ds" "$n instances done"
                echo "$n" > "$seen_f"
            fi
        done
        git log origin/main..HEAD --oneline 2>/dev/null | grep -q . && git push -q 2>/dev/null
        sleep "${UVM_POLL:-30}"
    done
}

STOP="logs/.uvm_stop_$$"
rm -f "$STOP"
watcher &
WATCHER=$!

# --- 02 inference ------------------------------------------------------------------
python monitorability-eval/scripts/02_run_inference_qwen.py --config $C --model $MODEL \
    --datasets "${DATASETS[@]}" --hint-type $HINT
rc=$?
touch "$STOP"; wait $WATCHER 2>/dev/null; rm -f "$STOP" "$STOP".seen.*
for ds in "${DATASETS[@]}"; do push_dataset "$ds" "after 02 (exit $rc)"; done
[[ $rc -ne 0 ]] && { echo "02 FAILED (exit $rc) — rerun this script to resume."; exit $rc; }

# --- 08 + 58 -----------------------------------------------------------------------
python monitorability-eval/scripts/08_cot_token_analysis.py --config $C --model $MODEL \
    --datasets "${DATASETS[@]}" --hint-type $HINT
python monitorability-eval/scripts/58_uvm_answers_and_parse_report.py --config $C \
    --datasets "${DATASETS[@]}"

# --- Final push ---------------------------------------------------------------------
for ds in "${DATASETS[@]}"; do push_dataset "$ds" "final"; done
git add $OUT 2>/dev/null
git diff --cached --quiet || git commit -q -m "uvm: CoT stats + parse report (${DATASETS[*]})"
git push -q && echo "Final push OK." || echo "FINAL PUSH FAILED — run: git push"
git status -sb | head -1
echo "Done — $(date). Check 'up to date with origin/main' above BEFORE deleting the pod."
