#!/usr/bin/env bash
# =============================================================================
# RO2 MVP — full pipeline, one command (RunPod / single GPU).
#   token preflight -> seq preflight -> sanity gate -> full SFT training -> push.
#
# Usage (from the MVP-RO2 project root, inside the pod):
#   bash run_all.sh                       # everything
#   bash run_all.sh --skip-sanity         # skip the env gate (after it passed once)
#   bash run_all.sh --no-push             # train, don't push to the Hub
#   bash run_all.sh --config=configs/sft_harmlessness.yaml
#
# Prereq: a .env with HF_TOKEN (WRITE scope + the Llama-3.2 license accepted for
# that account), or `export HF_TOKEN=...` in this shell. See documentation.md.
# =============================================================================
set -euo pipefail

CONFIG="configs/sft_harmlessness.yaml"
SKIP_SANITY=0
NO_PUSH=0

for arg in "$@"; do
  case "$arg" in
    --skip-sanity)        SKIP_SANITY=1 ;;
    --no-push)            NO_PUSH=1 ;;
    --config=*)           CONFIG="${arg#*=}" ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
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
# Verifies HF_TOKEN is present + valid, can access the gated Llama model, and can
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
# token is actually active. Pass an explicit path: bare load_dotenv() calls
# find_dotenv(), which inspects the caller's stack frame and asserts when Python
# runs from stdin (this `python - <<'PY'` heredoc).
load_dotenv(os.path.join(os.getcwd(), ".env"))
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

# 2) Gated MODEL accessible (Llama-3.2). hh-rlhf is public; the model is gated.
model_id = cfg["base_model"]
try:
    api.model_info(model_id)
    print(f"[ok] can access gated model: {model_id}")
except Exception as e:
    m = str(e).lower()
    if "403" in m or "gated" in m or "awaiting" in m or "access" in m:
        fail(f"Token cannot access gated model {model_id}. Accept its license while "
             f"logged in as {user}: https://huggingface.co/{model_id}")
    print(f"[warn] could not pre-verify model access ({e}); the model load will confirm.")

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
# [02] Full SFT training (4 epochs -> 4 adapter checkpoints)
# ---------------------------------------------------------------------------
log "[02] Full SFT training"
python scripts/01_train_sft.py --config "$CONFIG"

# ---------------------------------------------------------------------------
# [03] Push all epoch adapters to the Hub (verified writable in [00])
# ---------------------------------------------------------------------------
if [[ "$NO_PUSH" -eq 0 ]]; then
  log "[03] Push adapters to the Hub"
  python scripts/02_push_adapters.py --config "$CONFIG"
else
  log "[03] push skipped (--no-push)"
fi

log "DONE"
echo "Adapters: <hf_username>/<repo_prefix>-epoch{1,2,3,4}."
echo "Merge an epoch for MVP-RO1 eval: python scripts/04_merge_and_push.py --config $CONFIG --epoch 4"
