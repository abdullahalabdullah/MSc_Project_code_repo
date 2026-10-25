#!/usr/bin/env bash
#
# run_pbm_inference.sh — GPU phase of the prior-belief-vs-monitorability
# experiment (config/prior_belief_vs_monitorability.yaml): selection check (61),
# inference (62: fresh unhinted + hinted A + hinted B, 20 runs each) and the
# parse report (58, unchanged), with every saved checkpoint gzipped, committed
# and pushed to GitHub as soon as it lands. Same safety design as
# run_uvm_inference.sh.
#
# Usage (from anywhere; the script cd's to Full-Codebase/RO1):
#   export HF_TOKEN=hf_...
#   bash monitorability-eval/run_pbm_inference.sh
#
# Safety:
#   - Preflight BEFORE the model loads: python deps, HF token, a
#     `git push --dry-run`, and the step-61 selection being present (it is
#     committed; the script never re-selects on its own).
#   - Resume from a fresh clone: plain .jsonl files are gitignored, so only the
#     .gz copies come back with the repo; they are decompressed first so step 62
#     appends to the full files and resumes from the committed progress.json.
#   - A background watcher polls the version-A progress.json (step 62 writes
#     both versions' progress right after a checkpoint's rows); on a change it
#     gzips + commits + pushes BOTH version directories. Deleting the pod loses
#     at most the checkpoint being generated.

set -uo pipefail
cd "$(dirname "$0")/.."

C=config/prior_belief_vs_monitorability.yaml
MODEL=Qwen/Qwen3-4B
HINT=grader_hacking
VERSIONS=(lexam_priorA lexam_priorB)
OUT=outputs-and-results/prior-belief-vs-monitorability
O=$OUT/inference-outputs/${MODEL//\//_}

mkdir -p logs
LOG="logs/pbm_inference_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "prior-belief-vs-monitorability inference — $(date)"
echo "Log: $LOG"

# --- Preflight ----------------------------------------------------------------
fail() { echo "PREFLIGHT FAILED: $*"; exit 1; }
python -c "import yaml, vllm, transformers" 2>/dev/null \
    || fail "python deps missing — run: python -m pip install --no-cache-dir -r requirements.txt"
[[ -n "${HF_TOKEN:-}" ]] || fail "HF_TOKEN is not set — export HF_TOKEN=hf_..."
for v in "${VERSIONS[@]}"; do
    [[ -s $OUT/data/$v/dataset_index.json ]] \
        || fail "selection missing ($OUT/data/$v/dataset_index.json) — commit + push the step-61 output first"
done
git config user.name  >/dev/null || git config --global user.name  "Abd"
git config user.email >/dev/null || git config --global user.email "abd@gmail.com"
git push --dry-run >/dev/null 2>&1 \
    || fail "git push does not work — set a token remote: git remote set-url origin https://<user>:<token>@github.com/abdullahalabdullah/MSc_Project_code_repo.git"
echo "Preflight OK (deps, HF_TOKEN, selection, git push)."

# --- Restore .jsonl from committed .gz (resume after a fresh clone) -------------
for v in "${VERSIONS[@]}"; do
    d=$O/$v/$HINT
    for arm in hinted unhinted; do
        if [[ -f $d/$arm.jsonl.gz && ! -s $d/$arm.jsonl ]]; then
            echo "Restoring $d/$arm.jsonl from .gz"
            gunzip -c "$d/$arm.jsonl.gz" > "$d/$arm.jsonl"
        fi
    done
done

# --- Checkpoint push ------------------------------------------------------------
push_versions() {  # $1 = commit message suffix
    local v d
    for v in "${VERSIONS[@]}"; do
        d=$O/$v/$HINT
        for arm in hinted unhinted; do
            [[ -s $d/$arm.jsonl ]] && gzip -nkf "$d/$arm.jsonl"
        done
        git add "$d"/*.gz "$d/progress.json" 2>/dev/null
    done
    git diff --cached --quiet && return 0
    git commit -q -m "pbm inference (${HINT}, lexam A/B) — $1" \
        && { git push -q && echo "[push] $1" \
             || echo "[push] commit made, PUSH FAILED (retried next checkpoint): $1"; }
}

watcher() {
    local p=$O/${VERSIONS[0]}/$HINT/progress.json seen="" n
    while [[ ! -f "$STOP" ]]; do
        if [[ -f $p ]]; then
            n=$(python -c "import json;print(len(json.load(open('$p'))['completed_instance_ids']))" 2>/dev/null) || n=""
            if [[ -n "$n" && "$n" != "$seen" ]]; then
                sleep 5  # let step 62 finish writing the second version's progress
                n=$(python -c "import json;print(len(json.load(open('$p'))['completed_instance_ids']))" 2>/dev/null)
                push_versions "$n questions done"
                seen=$n
            fi
        fi
        git log origin/main..HEAD --oneline 2>/dev/null | grep -q . && git push -q 2>/dev/null
        sleep "${PBM_POLL:-30}"
    done
}

STOP="logs/.pbm_stop_$$"
rm -f "$STOP"
watcher &
WATCHER=$!

# --- 62 inference ------------------------------------------------------------------
python monitorability-eval/scripts/62_pbm_run_inference.py --config $C --model $MODEL
rc=$?
touch "$STOP"; wait $WATCHER 2>/dev/null; rm -f "$STOP"
push_versions "after 62 (exit $rc)"
[[ $rc -ne 0 ]] && { echo "62 FAILED (exit $rc) — rerun this script to resume."; exit $rc; }

# --- 58 parse report (unchanged script; min-valid 18 of 20 for its count column) ---
python monitorability-eval/scripts/58_uvm_answers_and_parse_report.py --config $C \
    --model $MODEL --hint-type $HINT --datasets "${VERSIONS[@]}" --min-valid 18

# --- Final push ---------------------------------------------------------------------
push_versions "final"
git add $OUT 2>/dev/null
git diff --cached --quiet || git commit -q -m "pbm: parse report (lexam A/B)"
git push -q && echo "Final push OK." || echo "FINAL PUSH FAILED — run: git push"
git status -sb | head -1
echo "Done — $(date). Check 'up to date with origin/main' above BEFORE deleting the pod."
