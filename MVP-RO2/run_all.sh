#!/usr/bin/env bash
# =============================================================================
# RO2 MVP — full pipeline, one command (RunPod / single GPU).
#   token preflight -> seq preflight -> sanity gate -> base CoT baseline ->
#   full SFT training -> per-epoch CoT verification -> push.
#
# Usage (from the MVP-RO2 project root, inside the pod):
#   bash run_all.sh                       # everything
#   bash run_all.sh --skip-sanity         # skip the env gate (after it passed once)
#   bash run_all.sh --skip-baseline       # skip the base-model CoT baseline
#   bash run_all.sh --no-push             # train + verify, don't push to the Hub
#   bash run_all.sh --max-eval-samples=300
#   bash run_all.sh --config=configs/sft_harmlessness.yaml
#
# Prereq: a .env with HF_TOKEN (WRITE scope + PKU-SafeRLHF terms accepted), or
# `export HF_TOKEN=...` in this shell. See documentation.md.
# =============================================================================
set -euo pipefail

CONFIG="configs/sft_harmlessness.yaml"
SKIP_SANITY=0
SKIP_BASELINE=0
NO_PUSH=0
MAX_EVAL_SAMPLES=200

for arg in "$@"; do
  case "$arg" in
    --skip-sanity)        SKIP_SANITY=1 ;;
    --skip-baseline)      SKIP_BASELINE=1 ;;
    --no-push)            NO_PUSH=1 ;;
    --max-eval-samples=*) MAX_EVAL_SAMPLES="${arg#*=}" ;;
    --config=*)           CONFIG="${arg#*=}" ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $arg" >&2; exit 2 ;;
  esac
done

# Run from this script's directory so relative paths (configs/, outputs/) resolve.
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

log() { echo ""; echo "=== $(date '+%H:%M:%S') | $* ==="; }

# ---------------------------------------------------------------------------
# [00] TOKEN PREFLIGHT — fail fast, BEFORE any GPU work or multi-hour training.
# Verifies HF_TOKEN is present + valid, can READ the gated PKU dataset, and can
# WRITE (push) to the Hub. The write check is a real create+delete probe, so a
# read-only token is caught here instead of after training.
# ---------------------------------------------------------------------------
log "[00] Credential & token preflight"
REQUIRE_WRITE=1
[[ "$NO_PUSH" -eq 1 && "$SKIP_SANITY" -eq 1 ]] && REQUIRE_WRITE=0   # no push happens at all
CONFIG="$CONFIG" REQUIRE_WRITE="$REQUIRE_WRITE" python - <<'PY'
import os, sys, time, yaml
from dotenv import load_dotenv

# load_dotenv() does NOT override an existing env var: an HF_TOKEN already in the
# shell takes precedence over .env (matches sft/credentials.py). We test whatever
# token is actually active.
load_dotenv()
cfg = yaml.safe_load(open(os.environ["CONFIG"]))
require_write = os.environ.get("REQUIRE_WRITE", "1") == "1"

def fail(msg):
    print(f"\n[PREFLIGHT FAIL] {msg}", file=sys.stderr)
    sys.exit(1)

token = os.environ.get("HF_TOKEN")
if not token:
    fail("HF_TOKEN is not set. Put HF_TOKEN=hf_... (WRITE scope) in MVP-RO2/.env, "
         "or `export HF_TOKEN=...` in this shell.")

from huggingface_hub import HfApi
api = HfApi(token=token)

# 1) Token valid + identity
try:
    user = api.whoami().get("name")
except Exception as e:
    fail(f"HF_TOKEN is invalid / could not authenticate: {e}")
print(f"[ok] authenticated as: {user}")

# 2) Gated dataset readable (PKU-SafeRLHF)
ds = cfg["dataset"]
try:
    api.dataset_info(ds)
    print(f"[ok] can read gated dataset: {ds}")
except Exception as e:
    m = str(e).lower()
    if "403" in m or "gated" in m or "awaiting" in m or "access" in m:
        fail(f"Token cannot access gated dataset {ds}. Accept its terms while "
             f"logged in as {user}: https://huggingface.co/datasets/{ds}")
    print(f"[warn] could not pre-verify dataset access ({e}); the data load will confirm.")

# 3) WRITE capability — create + delete a throwaway repo (definitive push test)
if require_write:
    probe = f"{cfg['hf_username']}/_write_probe_{int(time.time())}"
    try:
        api.create_repo(probe, private=True, repo_type="model", exist_ok=True)
    except Exception as e:
        fail("HF_TOKEN cannot create/push repos — you need a **Write** token "
             "(https://huggingface.co/settings/tokens). NOTE: an HF_TOKEN already "
             "set in the shell environment overrides the one in .env. Detail: "
             f"{e}")
    try:
        api.delete_repo(probe, repo_type="model")
    except Exception:
        print(f"[warn] write OK, but could not delete probe repo {probe} — remove it manually.")
    print(f"[ok] WRITE access confirmed (user: {user})")
    if cfg.get("hf_username") and user and user != cfg["hf_username"]:
        print(f"[warn] config hf_username='{cfg['hf_username']}' != token user '{user}'. "
              "Pushes target hf_username — make sure that account is intended.")
else:
    print("[skip] write check (--no-push and --skip-sanity): nothing pushes this run")

# 4) OPENAI (downstream RO1 eval only — not needed for SFT)
if not os.environ.get("OPENAI_API_KEY"):
    print("[warn] OPENAI_API_KEY not set (fine for SFT; RO1 eval/monitor needs it later)")

print("[PREFLIGHT PASS] all required tokens present.")
PY

log "Environment"
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

# ---------------------------------------------------------------------------
# [01] Sequence-length preflight (max_length already pinned — informational)
# ---------------------------------------------------------------------------
log "[01] Sequence-length preflight"
python scripts/00_check_seq_length.py --config "$CONFIG"

# ---------------------------------------------------------------------------
# [GATE] Tiny end-to-end run + masking/loss/push assertions (must pass)
# ---------------------------------------------------------------------------
if [[ "$SKIP_SANITY" -eq 0 ]]; then
  log "[GATE] Sanity check"
  python scripts/sanity_check_sft.py
else
  log "[GATE] skipped (--skip-sanity)"
fi

# ---------------------------------------------------------------------------
# [BASE] Base-model CoT baseline on the eval set -> outputs/think_stats/base.json
# (the reference every trained checkpoint is compared against)
# ---------------------------------------------------------------------------
if [[ "$SKIP_BASELINE" -eq 0 ]]; then
  if [[ -f "outputs/think_stats/base.json" ]]; then
    log "[BASE] baseline exists (outputs/think_stats/base.json) — skipping"
    echo "      (delete it if you changed --max-eval-samples, so base & checkpoints match)"
  else
    log "[BASE] Base-model CoT baseline over the eval set"
    python scripts/05_think_token_eval.py --config "$CONFIG" \
      --checkpoint base --max-eval-samples "$MAX_EVAL_SAMPLES"
  fi
else
  log "[BASE] skipped (--skip-baseline)"
fi

# ---------------------------------------------------------------------------
# [02] Full SFT training (3 epochs -> 3 adapter checkpoints)
# ---------------------------------------------------------------------------
log "[02] Full SFT training"
python scripts/01_train_sft.py --config "$CONFIG"

# ---------------------------------------------------------------------------
# [03] Per-epoch CoT verification (quantitative + qualitative).
# Non-fatal: a verification hiccup must NOT lose the trained adapters / push.
# ---------------------------------------------------------------------------
OUTDIR="$(python -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['output_dir'])")"
log "[03] CoT verification over checkpoints in $OUTDIR"
mapfile -t CKPTS < <(ls -d "$OUTDIR"/checkpoint-* 2>/dev/null | sort -V)
if [[ ${#CKPTS[@]} -eq 0 ]]; then
  echo "[warn] no checkpoints found in $OUTDIR — skipping verification"
else
  for ckpt in "${CKPTS[@]}"; do
    echo "--- think-token count: $ckpt ---"
    python scripts/05_think_token_eval.py --config "$CONFIG" \
      --checkpoint "$ckpt" --max-eval-samples "$MAX_EVAL_SAMPLES" || \
      echo "[warn] think-token count failed for $ckpt (continuing)"
  done
  LAST="${CKPTS[-1]}"
  echo "--- generation sample (read the reasoning): $LAST ---"
  python scripts/03_sanity_generation.py --config "$CONFIG" --checkpoint "$LAST" || \
    echo "[warn] generation check failed for $LAST (continuing)"
fi

# ---------------------------------------------------------------------------
# [04] Push all epoch adapters to the Hub (verified writable in [00])
# ---------------------------------------------------------------------------
if [[ "$NO_PUSH" -eq 0 ]]; then
  log "[04] Push adapters to the Hub"
  python scripts/02_push_adapters.py --config "$CONFIG"
else
  log "[04] push skipped (--no-push)"
fi

log "DONE"
echo "CoT stats: outputs/think_stats/*.json — compare each epoch's mean to base.json."
echo "A healthy fix keeps each epoch near ~100% of base CoT; a collapse toward 0% = suppression."
