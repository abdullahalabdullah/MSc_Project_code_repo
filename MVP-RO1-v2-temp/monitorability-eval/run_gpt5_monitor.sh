#!/usr/bin/env bash
#
# run_gpt5_monitor.sh — run step 03 (the monitor) with a chosen monitor model
# on the Qwen/Qwen3-4B inference outputs, without permanently changing the
# tracked config.
#
# The monitor model is read from config/mvp_config.yaml (monitor.model); there
# is no CLI flag for it. This script backs up the config, points monitor.model
# at $MONITOR_MODEL for the duration of the run, then restores the original
# config on exit (even on error / Ctrl-C via the trap).
#
# Usage:
#   export OPENAI_API_KEY=sk-...
#   bash monitorability-eval/run_gpt5_monitor.sh [dataset ...]
#
#   With no args, runs the three datasets below. Pass dataset keys to override:
#     bash monitorability-eval/run_gpt5_monitor.sh mmlu_pro lexam
#
# Env overrides:
#   MONITOR_MODEL   monitor model id      (default: gpt-5)
#   MODEL_ID        inference model id    (default: Qwen/Qwen3-4B)
#   HINT_TYPE       hint type to monitor  (default: professor)

set -uo pipefail

# Repo root = parent of monitorability-eval/ (this script's dir).
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

MONITOR_MODEL="${MONITOR_MODEL:-gpt-5}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B}"
HINT_TYPE="${HINT_TYPE:-professor}"
CONFIG="config/mvp_config.yaml"

# Datasets: use CLI args if given, else default to the three present for this model.
if [[ $# -gt 0 ]]; then
    DATASETS=("$@")
else
    DATASETS=(mmlu_pro lexam medxpertqa)
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY is not set. export OPENAI_API_KEY=sk-... first." >&2
    exit 1
fi

if ! grep -qE '^  model: "gpt-5-mini"' "$CONFIG"; then
    echo "WARNING: config monitor.model is not the expected 'gpt-5-mini'." >&2
    echo "         Current value:" >&2
    grep -nE '^monitor:' -A 2 "$CONFIG" | sed 's/^/           /' >&2
fi

# Back up the config beside itself (same dir/filesystem — avoids /tmp sandbox
# surprises) and guarantee restore on any exit.
BACKUP="${CONFIG}.monitorbak"
cp "$CONFIG" "$BACKUP"
restore_config() {
    if [[ -f "$BACKUP" ]]; then
        cp "$BACKUP" "$CONFIG"
        rm -f "$BACKUP"
        echo ">> Restored original $CONFIG (monitor.model reverted)."
    else
        echo "!! Backup $BACKUP missing — could NOT auto-restore $CONFIG. Check monitor.model manually." >&2
    fi
}
trap restore_config EXIT

# Point monitor.model at the chosen model for the duration of this run.
sed -i '' -E "s/^(  model: )\".*\"/\1\"${MONITOR_MODEL}\"/" "$CONFIG"
echo ">> monitor.model set to \"${MONITOR_MODEL}\" for this run."
echo ">> model=${MODEL_ID}  hint=${HINT_TYPE}  datasets=${DATASETS[*]}"
echo

rc=0
for d in "${DATASETS[@]}"; do
    echo "==================================================================="
    echo ">> Monitoring dataset: $d"
    echo "==================================================================="
    if ! python monitorability-eval/scripts/03_run_monitor.py \
            --model "$MODEL_ID" --dataset "$d" --hint-type "$HINT_TYPE"; then
        echo "!! Monitor FAILED for dataset '$d' (continuing)." >&2
        rc=1
    fi
    echo
done

if [[ $rc -eq 0 ]]; then
    echo ">> Done. Results under outputs-and-results/monitor_outputs/$(echo "$MODEL_ID" | tr '/' '_')/<dataset>/${HINT_TYPE}/"
else
    echo ">> Finished with one or more failures (see above)." >&2
fi
exit $rc
