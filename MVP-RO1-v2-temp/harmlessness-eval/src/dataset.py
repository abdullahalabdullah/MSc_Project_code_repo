"""
WildJailbreak eval-set preparation for the harmlessness evaluation.

Self-contained: the harmlessness eval never imports the monitorability-eval code,
so the small shared helpers (`ensure_hf_auth`, `model_dir_name`) are duplicated
here on purpose. It loads the gated `allenai/wildjailbreak` eval config, keeps the
adversarial prompt + its subset label (adversarial_harmful / adversarial_benign),
and writes a flat prompts index the rest of the pipeline consumes.
"""

import json
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# HuggingFace auth (own copy — WildJailbreak and Qwen3 are gated)
# ---------------------------------------------------------------------------

def ensure_hf_auth(verbose: bool = True) -> bool:
    """Log into the HuggingFace Hub so gated assets (WildJailbreak, WildGuard,
    gated target models) download in any environment.

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

    Mirrors monitorability-eval's helper (kept independent so the two evals never
    import each other); the run driver reproduces it with ${MODEL_ID//\\//_}."""
    return model_id.strip().strip("/").replace("/", "_")


# ---------------------------------------------------------------------------
# WildJailbreak eval loading
# ---------------------------------------------------------------------------

def load_wildjailbreak_eval(bench_cfg: dict, limit: int | None = None) -> list:
    """Load the WildJailbreak eval config and return a list of prompt records.

    Each record: {prompt_id, subset, prompt}. `subset` is the raw data_type
    (adversarial_harmful / adversarial_benign). The eval config ships as TSV, so
    the delimiter is passed through and NA-filling is disabled to keep prompt
    text verbatim.
    """
    from datasets import load_dataset

    ensure_hf_auth()

    load_kwargs = {"keep_default_na": False}
    if bench_cfg.get("delimiter"):
        load_kwargs["delimiter"] = bench_cfg["delimiter"]
    ds = load_dataset(bench_cfg["hf"], bench_cfg.get("config", "eval"), **load_kwargs)

    # The eval config is a single split (HF names it "train"); take whatever split exists.
    split = ds[next(iter(ds.keys()))] if hasattr(ds, "keys") else ds

    prompt_field = bench_cfg.get("prompt_field", "adversarial")
    subset_field = bench_cfg.get("subset_field", "data_type")

    records = []
    for idx, row in enumerate(split):
        prompt = (row.get(prompt_field) or "").strip()
        if not prompt:
            continue
        records.append({
            "prompt_id": idx,
            "subset": row.get(subset_field, "unknown"),
            "prompt": prompt,
        })

    if limit is not None:
        records = records[:limit]
    return records


def prepare_prompts(config: dict, output_path: Path, limit: int | None = None) -> list:
    """Load WildJailbreak eval, write the prompts index to output_path, return it."""
    h = config["harmlessness"]
    records = load_wildjailbreak_eval(h["benchmark"], limit=limit)
    if not records:
        raise ValueError("WildJailbreak eval produced no prompts — check the config/benchmark.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Report the subset balance so the run log shows the harmful/benign split.
    from collections import Counter
    counts = Counter(r["subset"] for r in records)
    print(f"  Wrote {len(records)} prompts to {output_path}")
    for subset, n in sorted(counts.items()):
        print(f"    {subset}: {n}")
    return records


def load_prompts(index_path: Path) -> list:
    """Load the prepared prompts index (list of {prompt_id, subset, prompt})."""
    records = []
    with open(index_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
