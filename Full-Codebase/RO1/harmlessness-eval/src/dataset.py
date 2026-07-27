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
import random
from collections import Counter, defaultdict
from pathlib import Path

# The WildJailbreak eval config is a single fixed pool; test and eval are two
# DISJOINT proportional-stratified samples of it (see prepare_split). Mirrors the
# monitorability eval's (test, eval) split names and `eval/` path namespacing.
SPLITS = ("test", "eval")


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
# Path namespacing (test / eval split)
# ---------------------------------------------------------------------------

def resolve_paths(root, config: dict, model_name: str | None = None,
                  split: str = "test") -> dict:
    """Resolve harmlessness on-disk paths for a (model, split) pair.

    `split` = "test" (default) keeps the existing flat layout; "eval" inserts an
    `eval/` segment after the dataset dir / model dir everywhere, so the two
    splits never overwrite each other (mirrors monitorability-eval.resolve_paths):

        data/wildjailbreak_eval/[eval/]prompts.jsonl
        responses/{model}/[eval/]responses.jsonl
        wildguard/{model}/[eval/]classifications.jsonl
        analysis/{model}/[eval/]

    `model_name` is required for every path except the prompts index (which is
    model-independent — both splits are sampled once, shared by all models)."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got '{split}'")
    root = Path(root)
    p = config["harmlessness"]["paths"]
    seg = () if split == "test" else ("eval",)
    paths = {
        "data_index": root.joinpath(p["data_dir"], "wildjailbreak_eval", *seg, "prompts.jsonl"),
    }
    if model_name:
        paths["responses"] = root.joinpath(p["responses_dir"], model_name, *seg, "responses.jsonl")
        paths["wildguard"] = root.joinpath(p["wildguard_dir"], model_name, *seg, "classifications.jsonl")
        paths["analysis_dir"] = root.joinpath(p["analysis_dir"], model_name, *seg)
    return paths


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


def _stratified_partition(records: list, n: int, rng: random.Random,
                          key: str = "subset") -> tuple[list, list]:
    """Proportional stratified sample of `n` records by `key` (default the
    adversarial_harmful / adversarial_benign subset), preserving each stratum's
    share of the pool so the harmful/benign ratio is kept. Returns
    (sample, remainder). Deterministic in `rng`; the fractional leftover is
    allocated by largest remainder (ties broken by stratum name). If
    n >= len(records) the whole pool is the sample (remainder empty)."""
    if n >= len(records):
        out = list(records)
        rng.shuffle(out)
        return out, []

    groups: dict[str, list] = defaultdict(list)
    for r in records:
        groups[r.get(key, "unknown")].append(r)
    for items in groups.values():
        rng.shuffle(items)

    total = len(records)
    alloc: dict[str, int] = {}
    frac: dict[str, float] = {}
    for g, items in groups.items():
        exact = n * len(items) / total
        alloc[g] = int(exact)
        frac[g] = exact - int(exact)
    assigned = sum(alloc.values())
    # Distribute the remaining slots to the largest fractional remainders.
    for g in sorted(frac, key=lambda k: (frac[k], k), reverse=True):
        if assigned >= n:
            break
        if alloc[g] < len(groups[g]):
            alloc[g] += 1
            assigned += 1

    sample: list = []
    remainder: list = []
    for g, items in groups.items():
        k = alloc[g]
        sample.extend(items[:k])
        remainder.extend(items[k:])
    rng.shuffle(sample)
    rng.shuffle(remainder)
    return sample, remainder


def prepare_split(config: dict, output_path: Path, split: str = "test",
                  test_records: list | None = None, n_examples: int | None = None,
                  limit: int | None = None) -> list:
    """Prepare a WildJailbreak prompts index for `split` and write it (JSONL).

    Partition semantics (mirrors the monitorability eval): the WildJailbreak eval
    config is a single fixed pool; test and eval are DISJOINT proportional
    stratified samples of it (harmful/benign ratio preserved), so eval never
    overlaps test.

      test  — default size = 2/3 of the pool (⌊2·|pool|/3⌋), which leaves a
              disjoint half for eval; override with `n_examples`.
      eval  — sampled from the pool MINUS the test split; default size =
              len(test)//2 (half the test set). Requires the FINAL test index,
              passed as `test_records`.

    `limit` caps the loaded pool first (smoke-test only). Deterministic in the
    config `harmlessness.random_seed` (default 88)."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got '{split}'")
    h = config["harmlessness"]
    rng = random.Random(h.get("random_seed", 88))
    pool = load_wildjailbreak_eval(h["benchmark"], limit=limit)
    if not pool:
        raise ValueError("WildJailbreak eval produced no prompts — check the config/benchmark.")

    if split == "test":
        n = n_examples if n_examples is not None else (2 * len(pool)) // 3
        records, _ = _stratified_partition(pool, n, rng)
    else:  # eval
        if not test_records:
            raise ValueError("eval split requires the FINAL test index (test_records).")
        used_ids = {r["prompt_id"] for r in test_records}
        used_prompts = {r["prompt"] for r in test_records}
        avail = [r for r in pool
                 if r["prompt_id"] not in used_ids and r["prompt"] not in used_prompts]
        if not avail:
            raise ValueError(
                f"no prompts left after excluding the {len(test_records)}-prompt "
                f"test split — the test index likely still covers the whole "
                f"{len(pool)}-prompt pool (a pre-split full run). Regenerate the "
                f"test split at the 2/3 partition size first: "
                f"01_prepare_wildjailbreak.py --data test --force")
        n = n_examples if n_examples is not None else len(test_records) // 2
        if n > len(avail):
            print(f"  NOTE: only {len(avail)} prompts left after excluding the "
                  f"{len(test_records)}-prompt test split; using all (below n={n}). "
                  f"Regenerate the test split at the 2/3 partition size "
                  f"(01_prepare_wildjailbreak.py --data test --force) if you expected more.")
            n = len(avail)
        records, _ = _stratified_partition(avail, n, rng)
        assert_splits_disjoint(test_records, records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Report the subset balance so the run log shows the harmful/benign split.
    counts = Counter(r["subset"] for r in records)
    print(f"  Wrote {len(records)} '{split}' prompts to {output_path}")
    for subset, n_sub in sorted(counts.items()):
        print(f"    {subset}: {n_sub}")
    return records


def assert_splits_disjoint(test_records: list, eval_records: list) -> None:
    """Hard-fail if the test and eval splits share any prompt (by prompt_id or
    prompt text). Guards against the eval index being sampled against a stale
    test index (e.g. test regenerated at a different size afterwards)."""
    test_ids = {r["prompt_id"] for r in test_records}
    test_prompts = {r["prompt"] for r in test_records}
    overlap = [r["prompt_id"] for r in eval_records
               if r["prompt_id"] in test_ids or r["prompt"] in test_prompts]
    if overlap:
        raise ValueError(
            f"eval split overlaps the test split on {len(overlap)} prompt(s) "
            f"(prompt_ids {overlap[:10]}...). The test index was probably "
            f"regenerated after the eval index was sampled. Regenerate the eval "
            f"index (01_prepare_wildjailbreak.py --data eval --force) from the "
            f"FINAL test index."
        )


def load_prompts(index_path: Path) -> list:
    """Load the prepared prompts index (list of {prompt_id, subset, prompt})."""
    records = []
    with open(index_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
