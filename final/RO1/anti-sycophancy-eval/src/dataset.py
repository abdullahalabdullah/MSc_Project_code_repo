"""
Data plumbing for the anti-sycophancy evaluation.

Self-contained: the anti-sycophancy eval never imports the monitorability or
harmlessness code, so the small shared helpers (`ensure_hf_auth`, `model_dir_name`)
are duplicated here on purpose (repo convention).

The two frozen eval files are PRODUCED by the RO2 curation pipeline (Stage B of
anti-syc-DPO-train-dataset-curation) and imported once into data/reserved/ by
scripts/01_prepare_eval_data.py; everything here only reads them.
"""

import hashlib
import json
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# HuggingFace auth (own copy — trained checkpoints / Qwen3 may be gated/private)
# ---------------------------------------------------------------------------

def ensure_hf_auth(verbose: bool = True) -> bool:
    """Log into the HuggingFace Hub so gated/private assets download anywhere.

    Token resolution order (first hit wins):
      1. env var HF_TOKEN / HUGGING_FACE_HUB_TOKEN / HUGGINGFACE_HUB_TOKEN
      2. the `hf_token` field in the shared config/mvp_config.yaml
    """
    token = (os.environ.get("HF_TOKEN")
             or os.environ.get("HUGGING_FACE_HUB_TOKEN")
             or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    if not token:
        cfg_path = Path(__file__).resolve().parents[2] / "config" / "mvp_config.yaml"
        if cfg_path.exists():
            try:
                import yaml
                cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
                token = cfg.get("hf_token")
            except Exception:
                token = None
    if not token:
        if verbose:
            print("  No HF token found (env or config). Public assets only.")
        return False
    try:
        from huggingface_hub import login
        login(token=token)
        if verbose:
            print("  Logged into HuggingFace Hub.")
        return True
    except Exception as e:
        if verbose:
            print(f"  HF login failed: {e}")
        return False


def model_dir_name(model_id: str) -> str:
    """Folder-safe name for a model ID: 'Qwen/Qwen3-4B' -> 'Qwen_Qwen3-4B'.

    Mirrors the other evals' helper; the run drivers reproduce it with
    ${MODEL_ID//\\//_}."""
    return model_id.strip().strip("/").replace("/", "_")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def resolve_paths(root: Path, config: dict, model_name: str | None = None) -> dict:
    """All repo-root-relative paths from the anti_sycophancy config block.

    Model-namespaced dirs are included when model_name is given.
    """
    a = config["anti_sycophancy"]["paths"]
    data_dir = root / a["data_dir"]
    paths = {
        "data_dir": data_dir,
        "reserved_dir": data_dir / "reserved",
        "eval_id_reserved": data_dir / "reserved" / "eval_id_triviaqa.jsonl",
        "eval_ood_reserved": data_dir / "reserved" / "eval_ood_sycophancyeval.jsonl",
        "reserved_manifest": data_dir / "reserved" / "manifest.json",
        "eval_id_index": data_dir / "eval_id_index.jsonl",
        "eval_ood_index": data_dir / "eval_ood_index.jsonl",
    }
    if model_name:
        paths.update({
            "responses_dir": root / a["responses_dir"] / model_name,
            "grades_dir": root / a["grades_dir"] / model_name,
            "analysis_dir": root / a["analysis_dir"] / model_name,
        })
    paths["compare_dir"] = root / a["analysis_dir"] / "compare"
    return paths


# ---------------------------------------------------------------------------
# JSONL / hashing helpers
# ---------------------------------------------------------------------------

def load_jsonl(path) -> list:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path, records):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
