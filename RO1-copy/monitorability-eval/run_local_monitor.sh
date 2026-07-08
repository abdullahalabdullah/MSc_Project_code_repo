#!/usr/bin/env bash
#
# run_local_monitor.sh — GPU-pod driver for the LOCAL monitor backend (step 3
# only). Runs 03_run_monitor.py --backend local: the monitor model (config
# monitor.local, default Qwen/Qwen3-8B via vLLM) is loaded ONCE and reused
# across every dataset, so multi-dataset runs pay the model-load cost a single
# time. No OPENAI_API_KEY needed; needs a GPU (~18 GB VRAM minimum for the 8B
# monitor in bfloat16 — a 48 GB card like an A40 is comfortable).
#
# Unlike run_monitor_and_results.sh, dataset keys are NOT validated against the
# config `datasets:` registry — the monitor treats --dataset as a pure path
# namespace, so legacy output dirs (e.g. agieval/mmlu/gpqa from older runs) work.
#
# Usage:
#   bash monitorability-eval/run_local_monitor.sh <model_id> --dataset <d1 [d2 ...]> [--hint <h>] [--monitor <key>] [--data test|eval] [--no-push]
#
#   model_id   the SAME model ID used at inference (resolves the same dirs). FIRST arg.
#   --dataset  REQUIRED. One or more dataset keys (space- and/or comma-separated).
#   --hint     hint type (default: professor). Must match the inference run.
#   --monitor  which config monitor.local.models entry to run (e.g. qwen3_8b,
#              gemma_e4b). Default: config monitor.local.default. A non-default
#              choice writes to a {key}/ subdir of the monitor dir.
#   --data     data split (default: test). Must match the inference run.
#   --no-push  skip the git commit+push of the monitor outputs at the end.
#
# Examples:
#   bash monitorability-eval/run_local_monitor.sh qwen3_1_7b --dataset agieval gpqa mmlu wmdp --hint professor
#   bash monitorability-eval/run_local_monitor.sh qwen3_1_7b --dataset agieval gpqa mmlu wmdp --hint professor --monitor gemma_e4b
#   bash monitorability-eval/run_local_monitor.sh Qwen/Qwen3-4B --dataset mmlu_pro,lexam --hint professor
#
# Resumable: rerunning the same command skips already-monitored calls
# (monitor_progress.json). Logs to logs/ (gitignored).

set -uo pipefail

# cd to the repo root (this script lives in monitorability-eval/) so the shared
# config/, outputs-and-results/, and logs/ resolve as repo-root-relative paths.
cd "$(dirname "$0")/.."

USAGE="usage: bash monitorability-eval/run_local_monitor.sh <model_id> --dataset <d1 [d2 ...]> [--hint <h>] [--monitor <key>] [--data test|eval] [--no-push]"

MODEL_ID="${1:-}"
if [[ -z "$MODEL_ID" || "$MODEL_ID" == --* ]]; then
    echo "ERROR: the first argument must be the model id."; echo "$USAGE"; exit 1
fi
shift

DATASET_ARGS=(); HINT="professor"; MONITOR=""; DATA_SPLIT="test"; PUSH=true
mode=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset|--datasets)        mode="dataset" ;;
        --hint|--hints|--hint-type)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "ERROR: --hint requires a hint type."; echo "$USAGE"; exit 1
            fi
            HINT="$2"; shift ;;
        --monitor|--monitor-model)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "ERROR: --monitor requires a monitor.local.models key."; echo "$USAGE"; exit 1
            fi
            MONITOR="$2"; shift ;;
        --data)
            if [[ $# -lt 2 || ! "$2" =~ ^(test|eval)$ ]]; then
                echo "ERROR: --data requires 'test' or 'eval' (got '${2:-}')."; echo "$USAGE"; exit 1
            fi
            DATA_SPLIT="$2"; shift ;;
        --no-push) PUSH=false ;;
        --*) echo "ERROR: unknown flag '$1'."; echo "$USAGE"; exit 1 ;;
        *)
            IFS=',' read -r -a _parts <<< "$1"
            if [[ "$mode" == "dataset" ]]; then
                DATASET_ARGS+=("${_parts[@]}")
            else
                echo "ERROR: '$1' appears before --dataset."; echo "$USAGE"; exit 1
            fi ;;
    esac
    shift
done

if [[ ${#DATASET_ARGS[@]} -eq 0 ]]; then
    echo "ERROR: --dataset is required with at least one dataset."; echo "$USAGE"; exit 1
fi

# Folder-safe model name — must mirror model_dir_name() in src/dataset.py.
MODEL_NAME="${MODEL_ID//\//_}"

# --- Preflight: inference outputs must exist --------------------------------
echo "Checking inference outputs for $MODEL_ID ..."
missing=0
SPLIT_SEG=""
[[ "$DATA_SPLIT" == "eval" ]] && SPLIT_SEG="/eval"
for ds in "${DATASET_ARGS[@]}"; do
    for arm in hinted unhinted; do
        f="outputs-and-results/inference-outputs/${MODEL_NAME}/${ds}${SPLIT_SEG}/${HINT}/${arm}.jsonl"
        if [[ -f "$f" ]]; then
            echo "  OK  $f"
        else
            echo "  MISSING  $f"
            missing=$((missing + 1))
        fi
    done
done
if [[ $missing -gt 0 ]]; then
    echo "ERROR: $missing inference file(s) missing — run inference first (and pull)."
    exit 1
fi

# Mirror all output to a timestamped log file (logs/ is gitignored).
mkdir -p logs
LOGFILE="logs/local_monitor_${MODEL_NAME}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "Local monitor run started: $(date)"
echo "Model:      $MODEL_ID (dirs namespaced as '$MODEL_NAME')"
echo "Datasets:   ${DATASET_ARGS[*]}"
echo "Hint type:  $HINT"
echo "Data split: $DATA_SPLIT"
echo "Monitor:    ${MONITOR:-<config default>} (config monitor.local.models)"
echo "Backend:    local (vLLM)"
echo "Log file:   $LOGFILE"
echo ""

# ONE python invocation for all datasets = one monitor-model load.
MONITOR_FLAG=()
[[ -n "$MONITOR" ]] && MONITOR_FLAG=(--monitor-model "$MONITOR")
python monitorability-eval/scripts/03_run_monitor.py \
    --model "$MODEL_ID" \
    --dataset "${DATASET_ARGS[@]}" \
    --hint-type "$HINT" \
    --data "$DATA_SPLIT" \
    --backend local ${MONITOR_FLAG[@]+"${MONITOR_FLAG[@]}"}
rc=$?
if [[ $rc -ne 0 ]]; then
    echo "ERROR: 03_run_monitor.py exited with $rc (progress is checkpointed — rerun to resume)."
    exit $rc
fi

# --- Commit + push the monitor outputs (fail-soft) --------------------------
if $PUSH; then
    git config --global user.email "abd@gmail.com" || true
    git config --global user.name "Abd" || true
    git add -A
    if git diff --cached --quiet; then
        echo "(nothing to commit)"
    else
        git commit -m "local monitor (${HINT}${MONITOR:+, $MONITOR}) — ${MODEL_NAME}: ${DATASET_ARGS[*]}" \
            && git push || echo "WARNING: git commit/push failed — outputs are on disk."
    fi
fi

echo ""
echo "Done: $(date). Results in outputs-and-results/monitor_outputs/${MODEL_NAME}/{ds}${SPLIT_SEG}/${HINT}/"
