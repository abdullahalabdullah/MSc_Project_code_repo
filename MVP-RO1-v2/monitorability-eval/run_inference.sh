#!/usr/bin/env bash
#
# run_inference.sh — GPU phase: prepare datasets + hinted/unhinted inference +
# CoT token-length report. Run on the GPU box (e.g. RunPod). The monitor and
# results phase runs separately afterwards: see run_monitor_and_results.sh.
#
# Usage:
#   bash monitorability-eval/run_inference.sh <model_id> --dataset <d1 [d2 ...]|ALL> --hint <h1 [h2 ...]|ALL> [--N <int>] [--merge]
#
#   model_id   HF model ID (e.g. Qwen/Qwen3-4B) or a local model dir. FIRST arg.
#              All output dirs are namespaced by its folder-safe name ('/' -> '_').
#   --merge    OPTIONAL (default: off). Treat model_id as a LoRA ADAPTER repo
#              (adapter-only push, no config.json — vLLM can't load it directly):
#              step 2 merges it into the fixed base from the config merge: block
#              (always the same base) and runs inference on the local merged
#              model. Output dirs stay namespaced by the ADAPTER id.
#   --dataset  REQUIRED. One or more dataset keys from config/mvp_config.yaml.
#              Every token after --dataset (until --hint) is a dataset; tokens may
#              be space- and/or comma-separated (mmlu_pro lexam == mmlu_pro,lexam).
#              ALL selects every key.
#   --hint     REQUIRED. One or more hint types (professor, visual_pattern,
#              metadata, wrong_few_shot, grader_hacking, unethical); same
#              space/comma rules. ALL selects every type.
#   --N        OPTIONAL. N — samples per arm per instance (positive integer).
#              Overrides config qwen_inference.n_samples and the N_SAMPLES env
#              var. Example: --N 8
#   (--dataset, --hint, and --N may appear in any order.)
#
# Examples:
#   bash monitorability-eval/run_inference.sh Qwen/Qwen3-4B --dataset ALL --hint professor
#   bash monitorability-eval/run_inference.sh Qwen/Qwen3-4B --dataset mmlu_pro lexam --hint professor metadata
#   bash monitorability-eval/run_inference.sh Qwen/Qwen3-4B --dataset mmlu_pro,lexam --hint ALL
#   bash monitorability-eval/run_inference.sh Qwen/Qwen3-4B --dataset ALL --hint ALL --N 8
#
# Optional env knobs:
#   N_EXAMPLES=50        override each dataset's config n_examples budget
#   N_SAMPLES=8          override config qwen_inference.n_samples (N per arm);
#                        the --N flag takes precedence over this
#   FORCE_PREPARE=true   regenerate dataset indexes (implied when N_EXAMPLES is set)
#
# Fail-soft: a failing step is logged as FAIL and the run continues (no set -e).
# Progress is committed + pushed after each phase, and DURING inference each
# dataset's outputs are auto-committed + pushed as soon as that dataset finishes
# (background watcher on progress.json) — so a crash/spot reclaim loses at most
# the dataset currently generating. Output is mirrored to logs/ (gitignored).

set -uo pipefail

# cd to the repo root (this script lives in monitorability-eval/) so the shared
# config/, outputs-and-results/, and logs/ resolve as repo-root-relative paths.
cd "$(dirname "$0")/.."

USAGE="usage: bash monitorability-eval/run_inference.sh <model_id> --dataset <d1 [d2 ...]|ALL> --hint <h1 [h2 ...]|ALL> [--N <int>] [--data test|eval] [--merge]
       datasets/hints are space- and/or comma-separated, e.g. --dataset mmlu_pro,lexam --hint professor metadata
       --N <int>: samples per arm per instance (positive integer; overrides config + N_SAMPLES env var)
       --data test|eval: data split (default test). eval is sampled disjointly from test
                         (test index must exist and be final) and namespaced under {ds}/eval/
       --merge: model_id is a LoRA adapter repo — merge into the config merge.base before inference"

# First positional is the model id; --dataset / --hint follow (either order).
MODEL_ID="${1:-}"
if [[ -z "$MODEL_ID" || "$MODEL_ID" == --* ]]; then
    echo "ERROR: the first argument must be the model id."; echo "$USAGE"; exit 1
fi
shift

# Collect every token after --dataset into DATASET_ARGS and after --hint into
# HINT_ARGS, until the next flag. Tokens may themselves be comma-separated.
DATASET_ARGS=(); HINT_ARGS=(); MERGE=false; N_SAMPLES_CLI=""; DATA_SPLIT="test"
saw_dataset=false; saw_hint=false; mode=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset|--datasets)        mode="dataset"; saw_dataset=true ;;
        --hint|--hints|--hint-type)  mode="hint";    saw_hint=true ;;
        --N)
            if [[ $# -lt 2 || ! "$2" =~ ^[1-9][0-9]*$ ]]; then
                echo "ERROR: --N requires a positive integer (got '${2:-}')."; echo "$USAGE"; exit 1
            fi
            N_SAMPLES_CLI="$2"; shift ;;
        --data)
            if [[ $# -lt 2 || ! "$2" =~ ^(test|eval)$ ]]; then
                echo "ERROR: --data requires 'test' or 'eval' (got '${2:-}')."; echo "$USAGE"; exit 1
            fi
            DATA_SPLIT="$2"; shift ;;
        --merge)                     MERGE=true ;;
        --*) echo "ERROR: unknown flag '$1'."; echo "$USAGE"; exit 1 ;;
        *)
            IFS=',' read -r -a _parts <<< "$1"
            case "$mode" in
                dataset) DATASET_ARGS+=("${_parts[@]}") ;;
                hint)    HINT_ARGS+=("${_parts[@]}") ;;
                *) echo "ERROR: '$1' appears before --dataset/--hint."; echo "$USAGE"; exit 1 ;;
            esac ;;
    esac
    shift
done

# Both flags are REQUIRED and must carry at least one value.
if ! $saw_dataset || [[ ${#DATASET_ARGS[@]} -eq 0 ]]; then
    echo "ERROR: --dataset is required with at least one dataset (or ALL)."; echo "$USAGE"; exit 1
fi
if ! $saw_hint || [[ ${#HINT_ARGS[@]} -eq 0 ]]; then
    echo "ERROR: --hint is required with at least one hint type (or ALL)."; echo "$USAGE"; exit 1
fi

# Folder-safe model name — must mirror model_dir_name() in src/dataset.py.
MODEL_NAME="${MODEL_ID//\//_}"

# --- Expand/validate dataset + hint args against the config registries -------
ALL_DATASETS=($(python -c "import yaml; print(' '.join(yaml.safe_load(open('config/mvp_config.yaml'))['datasets']))"))
ALL_HINTS=($(python -c "import yaml; print(' '.join(yaml.safe_load(open('config/mvp_config.yaml'))['hint']['types']))"))

# ALL as any token expands to the full registry; otherwise validate each key.
# All given datasets are passed as ONE list to 02 so the model loads once.
if printf '%s\n' "${DATASET_ARGS[@]}" | grep -qx "ALL"; then
    DATASETS=("${ALL_DATASETS[@]}")
else
    DATASETS=("${DATASET_ARGS[@]}")
    for ds in "${DATASETS[@]}"; do
        if [[ " ${ALL_DATASETS[*]} " != *" $ds "* ]]; then
            echo "ERROR: unknown dataset '$ds'. Valid: ${ALL_DATASETS[*]} (or ALL)."; exit 1
        fi
    done
fi

if printf '%s\n' "${HINT_ARGS[@]}" | grep -qx "ALL"; then
    HINT_TYPES=("${ALL_HINTS[@]}")
else
    HINT_TYPES=("${HINT_ARGS[@]}")
    for hint in "${HINT_TYPES[@]}"; do
        if [[ " ${ALL_HINTS[*]} " != *" $hint "* ]]; then
            echo "ERROR: unknown hint type '$hint'. Valid: ${ALL_HINTS[*]} (or ALL)."; exit 1
        fi
    done
fi

# --data eval: some datasets have no eval split (config eval_exclude_datasets —
# pool too small for a disjoint set on top of the test split). Explicitly
# requesting one is an error; ALL just skips them with a note.
if [[ "$DATA_SPLIT" == "eval" ]]; then
    EVAL_EXCLUDE=($(python -c "import yaml; print(' '.join(yaml.safe_load(open('config/mvp_config.yaml')).get('eval_exclude_datasets') or []))"))
    if [[ ${#EVAL_EXCLUDE[@]} -gt 0 ]]; then
        _KEPT=()
        for ds in "${DATASETS[@]}"; do
            if [[ " ${EVAL_EXCLUDE[*]} " == *" $ds "* ]]; then
                if printf '%s\n' "${DATASET_ARGS[@]}" | grep -qx "$ds"; then
                    echo "ERROR: dataset '$ds' has no eval split (config eval_exclude_datasets: ${EVAL_EXCLUDE[*]})."; exit 1
                fi
                echo "NOTE: skipping '$ds' — no eval split (config eval_exclude_datasets)."
            else
                _KEPT+=("$ds")
            fi
        done
        DATASETS=("${_KEPT[@]}")
        if [[ ${#DATASETS[@]} -eq 0 ]]; then
            echo "ERROR: no datasets left after eval-split exclusions."; exit 1
        fi
    fi
fi

# --- Optional knobs -> CLI flags ----------------------------------------------
# Precedence for N: --N flag > N_SAMPLES env var > config default.
# (Line below still passes --n-samples: that's the 02 python script's flag name.)
N_EXAMPLES="${N_EXAMPLES:-}"
N_SAMPLES="${N_SAMPLES_CLI:-${N_SAMPLES:-}}"
FORCE_PREPARE="${FORCE_PREPARE:-false}"

PREP_FLAGS=" --data $DATA_SPLIT"
[[ -n "$N_EXAMPLES" ]] && PREP_FLAGS+=" --n-examples $N_EXAMPLES"
# A changed N_EXAMPLES needs a rebuild; honour an explicit FORCE_PREPARE too.
{ [[ -n "$N_EXAMPLES" ]] || [[ "$FORCE_PREPARE" == "true" ]]; } && PREP_FLAGS+=" --force"
INFER_FLAGS=" --data $DATA_SPLIT"
[[ -n "$N_SAMPLES" ]] && INFER_FLAGS+=" --n-samples $N_SAMPLES"
[[ "$MERGE" == "true" ]] && INFER_FLAGS+=" --merge"
# Split path segment mirroring resolve_paths(): "" for test, "/eval" for eval.
SPLIT_SEG=""
[[ "$DATA_SPLIT" == "eval" ]] && SPLIT_SEG="/eval"
# n/N suffix appended to commit messages for traceability.
NN_TAG=""
[[ -n "$N_EXAMPLES" ]] && NN_TAG+=", n=$N_EXAMPLES"
[[ -n "$N_SAMPLES" ]]  && NN_TAG+=", N=$N_SAMPLES"
[[ "$DATA_SPLIT" == "eval" ]] && NN_TAG+=", eval-split"
[[ "$MERGE" == "true" ]] && NN_TAG+=", merged"

# --- Fail-soft command runner --------------------------------------------------
SUMMARY=()

run() {
    echo ""
    echo "=================================================================="
    echo ">>> $*"
    echo "=================================================================="
    if "$@"; then
        echo "<<< OK: $*"
        SUMMARY+=("OK    | $*")
    else
        local rc=$?
        echo "<<< FAILED (exit $rc) — skipping and continuing: $*"
        SUMMARY+=("FAIL  | $*")
    fi
}

# Stage, commit, and push the current results. Fail-soft and tolerant of
# "nothing to commit": only push when the commit actually created something.
commit_push() {
    local msg="$1"
    run git add -A
    if git diff --cached --quiet; then
        echo "  (nothing to commit for: $msg)"
        SUMMARY+=("OK    | git commit (no changes) | $msg")
        return
    fi
    run git commit -m "$msg"
    run git push
}

# --- Per-dataset auto-push (phase 2) ---------------------------------------------
# 02 runs ALL datasets in one python process (model loads once), so bash can't
# commit between datasets directly. Instead a background watcher polls each
# dataset's progress.json and commits + pushes THAT dataset's output dir as soon
# as all its instances are done — a crash/spot reclaim then loses at most the
# dataset currently generating. The watcher stops via a stop-file (never killed
# mid-git, so no stale .git/index.lock is left behind); the phase-level
# commit_push after 02 stays as the catch-all for anything it missed.

_dataset_complete() {  # $1 = outputs dir, $2 = dataset key -> exit 0 iff all instances done
    python3 - "$1/progress.json" "monitorability-eval/data/$2$SPLIT_SEG/dataset_index.json" 2>/dev/null <<'PYEOF'
import json, sys
try:
    done = len(json.load(open(sys.argv[1])).get("completed_instance_ids", []))
    total = len(json.load(open(sys.argv[2])))
except Exception:
    sys.exit(1)
sys.exit(0 if total and done >= total else 1)
PYEOF
}

autopush_watcher() {  # $1 = stop file, $2 = hint type, $3.. = datasets
    local stop_file="$1" hint="$2"; shift 2
    local pushed=" " ds dir
    while [[ ! -f "$stop_file" ]]; do
        for ds in "$@"; do
            [[ "$pushed" == *" $ds "* ]] && continue
            # Mirrors resolve_paths() in src/dataset.py: outputs/{model}/{ds}[/eval]/{hint}.
            dir="outputs-and-results/inference-outputs/$MODEL_NAME/$ds$SPLIT_SEG/$hint"
            [[ -f "$dir/progress.json" ]] || continue
            _dataset_complete "$dir" "$ds" || continue
            echo "[auto-push] $ds complete — committing + pushing its outputs"
            git add "$dir"
            if git diff --cached --quiet; then
                echo "[auto-push] $ds: nothing new to commit"
                pushed+="$ds "
            elif git commit -q -m "$MODEL_NAME inference ($hint$NN_TAG) — $ds [auto-push]"; then
                git push -q && echo "[auto-push] $ds pushed" \
                            || echo "[auto-push] $ds committed; push failed (final push will retry)"
                pushed+="$ds "
            else
                echo "[auto-push] $ds: commit failed — retrying next pass"
            fi
        done
        sleep 20
    done
}

# Mirror all output to a timestamped log file (logs/ is gitignored).
mkdir -p logs
LOGFILE="logs/inference_${MODEL_NAME}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "Inference run started: $(date)"
echo "Model:       $MODEL_ID (dirs namespaced as '$MODEL_NAME')"
if [[ "$MERGE" == "true" ]]; then
    echo "Merge:       true — model_id is a LoRA adapter, merged into config merge.base before inference"
else
    echo "Merge:       false"
fi
echo "Datasets:    ${DATASETS[*]}"
echo "Hint types:  ${HINT_TYPES[*]}"
echo "Data split:  $DATA_SPLIT"
echo "N examples:  ${N_EXAMPLES:-config default} per dataset"
echo "N samples:   ${N_SAMPLES:-config default} per arm per instance"
echo "Force prep:  $FORCE_PREPARE"
echo "Log file:    $LOGFILE"

# --- Phase 0: git identity ------------------------------------------------------
# Ensure a committer identity exists so the per-phase commits never fail on a
# fresh box (e.g. RunPod) where git has no global user configured.
echo ""
echo "########## PHASE 0: git identity ##########"
run git config --global user.email "abd@gmail.com"
run git config --global user.name "Abd"

# --- Phase 1: prepare dataset indexes (shared by every hint type) ---------------
echo ""
echo "########## PHASE 1: prepare dataset indexes ##########"
for ds in "${DATASETS[@]}"; do
    run python monitorability-eval/scripts/01_prepare_dataset.py --dataset "$ds" $PREP_FLAGS
done
commit_push "dataset indexes${NN_TAG} — ${DATASETS[*]}"

# --- Phase 2: inference + CoT report, per hint type ------------------------------
# The model is loaded ONCE per hint type and reused across all datasets.
for hint in "${HINT_TYPES[@]}"; do
    echo ""
    echo "########## PHASE 2: inference — $MODEL_ID ($hint) ##########"
    AUTOPUSH_STOP="logs/.autopush_stop_$$"
    rm -f "$AUTOPUSH_STOP"
    autopush_watcher "$AUTOPUSH_STOP" "$hint" "${DATASETS[@]}" &
    AUTOPUSH_PID=$!
    run python monitorability-eval/scripts/02_run_inference_qwen.py --model "$MODEL_ID" --datasets "${DATASETS[@]}" --hint-type "$hint" $INFER_FLAGS
    touch "$AUTOPUSH_STOP"
    wait "$AUTOPUSH_PID" 2>/dev/null
    rm -f "$AUTOPUSH_STOP"
    commit_push "$MODEL_NAME inference ($hint$NN_TAG) — ${DATASETS[*]}"

    echo ""
    echo "########## PHASE 3: CoT token-length analysis — $MODEL_ID ($hint) ##########"
    run python monitorability-eval/scripts/08_cot_token_analysis.py --model "$MODEL_ID" --datasets "${DATASETS[@]}" --hint-type "$hint" --data "$DATA_SPLIT"
    commit_push "$MODEL_NAME CoT token-length analysis ($hint$NN_TAG)"
done

# --- Final push --------------------------------------------------------------------
# Catch-all: make sure EVERYTHING produced by this run is on GitHub, even if an
# intermediate per-phase push failed (fail-soft) or files were left unstaged.
echo ""
echo "########## FINAL PUSH ##########"
commit_push "$MODEL_NAME inference run complete (${HINT_TYPES[*]}$NN_TAG) — final push"

# --- Summary ---------------------------------------------------------------------
echo ""
echo "=================================================================="
echo "RUN SUMMARY ($(date))"
echo "=================================================================="
fail_count=0
for line in "${SUMMARY[@]}"; do
    echo "$line"
    [[ "$line" == FAIL* ]] && fail_count=$((fail_count + 1))
done
echo "------------------------------------------------------------------"
echo "Total commands: ${#SUMMARY[@]}  |  Failed: $fail_count"
echo "Full log: $LOGFILE"
_DS_CSV=$(IFS=,; echo "${DATASETS[*]}")
_HINT_CSV=$(IFS=,; echo "${HINT_TYPES[*]}")
_DATA_FLAG=""
[[ "$DATA_SPLIT" == "eval" ]] && _DATA_FLAG=" --data eval"
echo "Next: pull locally, then  bash monitorability-eval/run_monitor_and_results.sh $MODEL_ID --dataset $_DS_CSV --hint $_HINT_CSV$_DATA_FLAG"
