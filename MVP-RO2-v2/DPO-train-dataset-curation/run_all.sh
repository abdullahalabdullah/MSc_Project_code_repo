#!/usr/bin/env bash
# Runner for the harmlessness preference-pair pipeline (stages 01-06, ending in the
# training-ready dpo_dataset export).
# (The Beaver ranker sanity check is separate: sanity_beaver.py — or use run_pilot_and_sanity.sh
#  to do both.)
#
# By default the pipeline uses ALL unique WildJailbreak harmful prompts (vanilla_harmful +
# adversarial_harmful, train config). Set SAMPLE=<int> for a smaller seeded pilot run.
#
# Each invocation writes to a NEW timestamped folder under outputs/ (e.g.
# outputs/run_20260703_141530), so a new run never overrides an old one. Every stage
# shares the same folder via the exported RUN_OUTPUT_DIR. The training-ready dataset is
# saved to <run folder>/dpo_dataset — train.sh auto-discovers the newest one, so curation
# never needs re-running for future training runs.
#
# Usage:
#   ./run_all.sh                       # ALL prompts, new outputs/run_<timestamp>/
#   SAMPLE=300 ./run_all.sh            # seeded 300-prompt pilot instead
#   SEED=1 ./run_all.sh                # change the single seed used by every stage
#                                      # (same SEED env var train.sh reads; default 0)
#   EXCLUDE_RUNS=outputs/pilot ./run_all.sh      # don't re-sample prompts a previous run
#                                      # used; combine results later with merge_runs.py
#   RUN_OUTPUT_DIR=outputs/my_run ./run_all.sh   # write to a specific folder instead
#   PY=python3 ./run_all.sh
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-python}"

# ── Optional minimum-margin filter (filter-4) ───────────────────────────────
# EDIT THIS VALUE to turn the filter on. Leave EMPTY ("") = filter not applied (default).
# Set to a number (e.g. "2.0") to drop final pairs whose cost margin
# (rejected_cost - chosen_cost) is below it. For harmlessness leave it empty (filter-3
# already gates quality); use it for the helpfulness pipeline.
MIN_MARGIN=""
export MIN_MARGIN

# Fresh per-run output folder unless the caller pinned one.
if [ -z "${RUN_OUTPUT_DIR:-}" ]; then
    RUN_OUTPUT_DIR="outputs/run_$(date +%Y%m%d_%H%M%S)"
fi
export RUN_OUTPUT_DIR
mkdir -p "$RUN_OUTPUT_DIR"
echo "== Output folder for this run: $RUN_OUTPUT_DIR =="
echo "== Prompt selection: SAMPLE=${SAMPLE:-ALL} =="
echo "== Seed (all stages): SEED=${SEED:-0} =="
if [ -n "${EXCLUDE_RUNS:-}" ]; then
    echo "== Excluding prompts from previous run(s): ${EXCLUDE_RUNS} =="
fi
if [ -n "$MIN_MARGIN" ]; then
    echo "== filter-4 minimum margin: $MIN_MARGIN =="
else
    echo "== filter-4 minimum margin: not applied =="
fi

# ── Preflight: WildJailbreak is gated — fail here (before any GPU time) if the HF
# token is missing/invalid or the account hasn't accepted the dataset's terms ──────
echo "== Preflight: checking HF token + gated WildJailbreak access =="
$PY - <<'PYEOF'
import os, sys
try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass
token = os.environ.get("HF_TOKEN") or None
try:
    from huggingface_hub import HfApi
    info = HfApi(token=token).whoami()
    print(f"HF auth OK: user={info.get('name')}")
except Exception as e:
    sys.exit(f"ERROR: no valid HF credentials — WildJailbreak is gated. Set HF_TOKEN or run "
             f"`huggingface-cli login`. ({e})")
try:
    from huggingface_hub import auth_check
    auth_check("allenai/wildjailbreak", repo_type="dataset", token=token)
    print("Gated dataset access OK: allenai/wildjailbreak")
except Exception as e:
    sys.exit("ERROR: this HF account cannot read allenai/wildjailbreak — accept the terms on "
             f"https://huggingface.co/datasets/allenai/wildjailbreak first. ({e})")
PYEOF

echo "== Stage 01: prepare prompts =="
$PY 01_prepare_prompts.py

echo "== Stage 02: generate responses (GPU, vLLM) =="
$PY 02_generate_responses.py

echo "== Stage 03: classify (Qwen3Guard) + filter-1 (GPU) =="
$PY 03_classify_guard.py

echo "== Stage 04: Beaver cost scoring (GPU) =="
$PY 04_score_cost.py

echo "== Stage 05: filter-3 + reports =="
$PY 05_report.py

echo "== Stage 06: export training-ready DPO dataset (HF DatasetDict) =="
$PY 06_export_dpo_dataset.py

echo "== Done. Curation stats: $RUN_OUTPUT_DIR/report.md + $RUN_OUTPUT_DIR/diagnostics.json =="
echo "== DPO dataset (saved locally, reusable by train.sh): $RUN_OUTPUT_DIR/dpo_dataset =="
