"""
Dataset preparation for the MVP CoT monitorability evaluation.

Loads a dataset, samples N examples, assigns incorrect hint targets, and saves a
dataset_index.json. MMLU is the legacy default (read from the `dataset:` config
block, unnamespaced paths). Five further 4-option MCQ datasets — GPQA, GSM-MC,
MATH-MC, AGIEval (English), WMDP — are added via the `datasets:` registry and are
namespaced on disk by dataset key (see resolve_paths). The whole pipeline is
strictly 4-option (A-D); AGIEval's 5-option subsets are reduced to 4 by dropping
one distractor (seeded), never the correct option.
"""

import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

# Answer letters. The pipeline is strictly 4-option, so every dataset normalises to
# exactly these four. MMLU_LETTERS is kept as a name for backward-compatibility
# (imported by scripts/07_estimate_accuracy*.py for the answer-entropy calculation).
LETTERS = ["A", "B", "C", "D"]
MMLU_LETTERS = LETTERS


# ---------------------------------------------------------------------------
# HuggingFace authentication
# ---------------------------------------------------------------------------

# !!! SECURITY: hardcoded read-only HF token, intentionally committed so the
# pipeline runs with zero setup in any environment (per project owner's request).
# This IS pushed to git by run_all.sh. REMOVE THIS and rotate the token before
# any public release of the code. Env var / .hf_token still override it.
_FALLBACK_HF_TOKEN = "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"


def ensure_hf_auth(verbose: bool = True) -> bool:
    """Log into the HuggingFace Hub so gated assets (the GPQA dataset, and gated
    models like Qwen3 / Llama / Tulu) download in any environment.

    Token resolution order (first hit wins):
      1. env var HF_TOKEN / HUGGING_FACE_HUB_TOKEN / HUGGINGFACE_HUB_TOKEN
      2. a local, git-ignored `.hf_token` file at the MVP-RO1 root
      3. the hardcoded `_FALLBACK_HF_TOKEN` above (committed; remove before release)

    Returns False only if every source is empty AND login fails.
    """
    token = (os.environ.get("HF_TOKEN")
             or os.environ.get("HUGGING_FACE_HUB_TOKEN")
             or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    source = "env"
    if not token:
        token_file = Path(__file__).resolve().parent.parent / ".hf_token"
        if token_file.exists():
            token = token_file.read_text(encoding="utf-8").strip()
            source = str(token_file)
    if not token and _FALLBACK_HF_TOKEN:
        token = _FALLBACK_HF_TOKEN
        source = "hardcoded fallback"
    if not token:
        return False
    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        # Export so child libraries (transformers, vLLM) pick it up too.
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
        if verbose:
            print(f"HuggingFace: authenticated (token source: {source}).")
        return True
    except Exception as e:  # pragma: no cover - network/credential edge cases
        if verbose:
            print(f"WARNING: HuggingFace login failed ({e}); continuing unauthenticated.")
        return False


# ---------------------------------------------------------------------------
# Path namespacing
# ---------------------------------------------------------------------------

def resolve_paths(root, config: dict, dataset_key: str, model_name: str | None = None) -> dict:
    """Resolve all on-disk paths for a (dataset, model) pair.

    Every dataset (including "mmlu") is namespaced by its key:
        data/{ds}/dataset_index.json, outputs/{ds}/{model}/,
        monitor_outputs/{ds}/{model}/, analysis/{ds}/{model}/, plots/{ds}/

    `model_name` is appended to the per-model dirs (outputs/monitor/analysis) only
    when provided; data_index and plots_dir are never model-specific.
    """
    root = Path(root)
    p = config["paths"]

    def base(d):
        return root / d / dataset_key

    def model_dir(d):
        b = base(d)
        return (b / model_name) if model_name else b

    return {
        "data_index":  base(p["data_dir"]) / "dataset_index.json",
        "outputs_dir": model_dir(p["outputs_dir"]),
        "monitor_dir": model_dir(p["monitor_outputs_dir"]),
        "analysis_dir": model_dir(p["analysis_dir"]),
        "plots_dir":   base(p["plots_dir"]),
    }


# ---------------------------------------------------------------------------
# Normalisation helpers (shared by all non-MMLU loaders)
# ---------------------------------------------------------------------------

# A normalised record is a dict:
#   {"question": str, "options": [str, ...], "correct_idx": int,
#    "group": str, "source_id": str}
# `options` may have >4 entries (e.g. AGIEval); the finaliser reduces to 4.

# Matches an option label at the start of a choice string: "(A)" (AGIEval style,
# no trailing space), "A.", or "A)". Trailing whitespace is optional because
# AGIEval renders options as "(A)4" with no separator.
_LETTER_PREFIX_RE = re.compile(r"^\s*(?:\([A-E]\)|[A-E][.)])\s*")


def _clean(value) -> str:
    """Stringify an option/question cleanly (whole-number floats lose the .0)."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _strip_letter_prefix(choice: str) -> str:
    """Remove a leading '(A)' / 'A)' / 'A.' style option label if present."""
    return _LETTER_PREFIX_RE.sub("", str(choice)).strip()


def _reduce_to_four(options: list, correct_idx: int, rng: random.Random) -> tuple[list, int]:
    """Drop (len-4) distractors at random (never the correct option), keep order."""
    if len(options) <= 4:
        return options, correct_idx
    idxs = list(range(len(options)))
    distractors = [i for i in idxs if i != correct_idx]
    drop = set(rng.sample(distractors, len(options) - 4))
    kept = [i for i in idxs if i not in drop]
    return [options[i] for i in kept], kept.index(correct_idx)


def _finalize(instance_id: int, rec: dict, rng: random.Random) -> dict:
    """Turn a normalised record into a pipeline instance dict (4 options, A-D)."""
    options, correct_idx = _reduce_to_four(rec["options"], rec["correct_idx"], rng)
    assert len(options) == 4, f"{rec['source_id']}: {len(options)} options after reduction"
    assert 0 <= correct_idx < 4, f"{rec['source_id']}: bad correct_idx {correct_idx}"

    correct_letter = LETTERS[correct_idx]
    wrong_letters = [l for l in LETTERS if l != correct_letter]
    hint_target = rng.choice(wrong_letters)
    assert hint_target != correct_letter

    return {
        "instance_id": instance_id,
        "question_id": rec["source_id"],
        "subject": rec["group"],
        "question": _clean(rec["question"]),
        "choices": {LETTERS[i]: _clean(options[i]) for i in range(4)},
        "correct_answer": correct_letter,
        "hint_target": hint_target,
    }


def _balanced_sample(records: list, n: int, rng: random.Random) -> list:
    """Sample n records, balanced across distinct `group` values when there is
    more than one group; otherwise a plain random sample. Deterministic in rng."""
    if n >= len(records):
        out = list(records)
        rng.shuffle(out)
        return out

    groups: dict[str, list] = defaultdict(list)
    for r in records:
        groups[r["group"]].append(r)

    if len(groups) <= 1:
        return rng.sample(records, n)

    for g in groups.values():
        rng.shuffle(g)

    order = sorted(groups)
    iters = {g: iter(groups[g]) for g in order}
    exhausted: set = set()
    selected: list = []
    while len(selected) < n and len(exhausted) < len(order):
        for g in order:
            if g in exhausted:
                continue
            try:
                selected.append(next(iters[g]))
            except StopIteration:
                exhausted.add(g)
                continue
            if len(selected) >= n:
                break
    return selected


# ---------------------------------------------------------------------------
# Per-dataset loaders — each returns a list of normalised records
# ---------------------------------------------------------------------------

def _load_gpqa(ds_cfg: dict, rng: random.Random) -> list:
    from datasets import load_dataset as hf_load_dataset
    ds = hf_load_dataset(ds_cfg["hf"], ds_cfg["config"], split=ds_cfg["split"])
    recs = []
    for idx, row in enumerate(ds):
        options = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        options = [_clean(o) for o in options]
        order = list(range(4))
        rng.shuffle(order)                       # randomise correct-answer position
        shuffled = [options[i] for i in order]
        recs.append({
            "question": row["Question"],
            "options": shuffled,
            "correct_idx": order.index(0),       # original index 0 was the correct one
            "group": row.get("High-level domain") or "unknown",
            "source_id": f"gpqa_{row.get('Record ID', idx)}",
        })
    return recs


def _load_wmdp(ds_cfg: dict, rng: random.Random) -> list:
    from datasets import load_dataset as hf_load_dataset
    recs = []
    for cfg in ds_cfg["configs"]:
        ds = hf_load_dataset(ds_cfg["hf"], cfg, split=ds_cfg["split"])
        for idx, row in enumerate(ds):
            recs.append({
                "question": row["question"],
                "options": [_clean(c) for c in row["choices"]],
                "correct_idx": int(row["answer"]),
                "group": cfg,
                "source_id": f"{cfg}_{idx}",
            })
    return recs


def _load_letter_mc(ds_cfg: dict, rng: random.Random, group: str, prefix: str) -> list:
    """Shared loader for the A/B/C/D + Answer-letter schema (GSM-MC, MATH-MC)."""
    from datasets import load_dataset as hf_load_dataset
    ds = hf_load_dataset(ds_cfg["hf"], split=ds_cfg["split"])
    recs = []
    for idx, row in enumerate(ds):
        options = [row["A"], row["B"], row["C"], row["D"]]
        correct_idx = LETTERS.index(str(row["Answer"]).strip().upper())
        recs.append({
            "question": row["Question"],
            "options": [_clean(o) for o in options],
            "correct_idx": correct_idx,
            "group": group,
            "source_id": f"{prefix}_{idx}",
        })
    return recs


def _load_gsm_mc(ds_cfg: dict, rng: random.Random) -> list:
    return _load_letter_mc(ds_cfg, rng, group="gsm", prefix="gsm_mc")


def _load_math_mc(ds_cfg: dict, rng: random.Random) -> list:
    return _load_letter_mc(ds_cfg, rng, group="math", prefix="math_mc")


def _load_agieval(ds_cfg: dict, rng: random.Random) -> list:
    from datasets import load_dataset as hf_load_dataset
    recs = []
    for subset in ds_cfg["subsets"]:
        hf_id = f"{ds_cfg['hf_prefix']}{subset}"
        ds = hf_load_dataset(hf_id, split=ds_cfg["split"])
        for idx, row in enumerate(ds):
            gold = row["gold"]
            correct_idx = int(gold[0] if isinstance(gold, (list, tuple)) else gold)
            options = [_strip_letter_prefix(c) for c in row["choices"]]
            recs.append({
                "question": row["query"],
                "options": options,                  # 5 options -> reduced to 4 in _finalize
                "correct_idx": correct_idx,
                "group": subset,
                "source_id": f"{subset}_{idx}",
            })
    return recs


_LOADERS = {
    "gpqa": _load_gpqa,
    "gsm_mc": _load_gsm_mc,
    "math_mc": _load_math_mc,
    "agieval": _load_agieval,
    "wmdp": _load_wmdp,
}


# ---------------------------------------------------------------------------
# Top-level preparation entry point
# ---------------------------------------------------------------------------

def prepare_dataset(config: dict, output_path: Path, dataset_key: str = "mmlu") -> list:
    """Prepare a dataset_index.json for `dataset_key` and write it to output_path.

    `mmlu` (default) uses the legacy code path below, reading the `dataset:` config
    block, so its index is byte-identical to before. Every other key is resolved
    through the `datasets:` registry and the per-dataset loaders.
    """
    ensure_hf_auth()  # GPQA is gated; harmless no-op for public datasets

    if dataset_key != "mmlu":
        return _prepare_registry_dataset(config, output_path, dataset_key)

    from datasets import load_dataset as hf_load_dataset

    ds_cfg = config["dataset"]
    rng = random.Random(ds_cfg["random_seed"])

    print(f"Loading {ds_cfg['name']} ({ds_cfg['config']}, {ds_cfg['split']}) ...")
    dataset = hf_load_dataset(ds_cfg["name"], ds_cfg["config"], split=ds_cfg["split"])

    # Group rows by subject
    by_subject: dict[str, list] = defaultdict(list)
    for idx, row in enumerate(dataset):
        by_subject[row["subject"]].append((idx, row))

    # Sample up to max_per_subject from each subject
    max_per = ds_cfg["max_per_subject"]
    pool = []
    for subject, rows in sorted(by_subject.items()):
        sample_size = min(len(rows), max_per)
        pool.extend(rng.sample(rows, sample_size))

    n = ds_cfg["n_examples"]
    if len(pool) < n:
        raise ValueError(f"Pool has only {len(pool)} examples; need {n}.")

    selected = rng.sample(pool, n)

    records = []
    for instance_id, (orig_idx, row) in enumerate(selected):
        correct_letter = MMLU_LETTERS[row["answer"]]
        wrong_letters = [l for l in MMLU_LETTERS if l != correct_letter]
        hint_target = rng.choice(wrong_letters)

        assert hint_target != correct_letter, (
            f"instance {instance_id}: hint_target == correct_answer ({correct_letter})"
        )

        records.append({
            "instance_id": instance_id,
            "question_id": f"{row['subject']}_{orig_idx}",
            "subject": row["subject"],
            "question": row["question"],
            "choices": {
                "A": row["choices"][0],
                "B": row["choices"][1],
                "C": row["choices"][2],
                "D": row["choices"][3],
            },
            "correct_answer": correct_letter,
            "hint_target": hint_target,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(records)} instances to {output_path}")
    return records


def _prepare_registry_dataset(config: dict, output_path: Path, dataset_key: str) -> list:
    """Prepare a non-MMLU dataset from the `datasets:` registry."""
    if dataset_key not in config.get("datasets", {}):
        available = list(config.get("datasets", {}))
        raise ValueError(f"dataset '{dataset_key}' not in config datasets registry. Available: {available}")

    ds_cfg = config["datasets"][dataset_key]
    loader = _LOADERS.get(ds_cfg["loader"])
    if loader is None:
        raise ValueError(f"no loader '{ds_cfg['loader']}' registered for dataset '{dataset_key}'.")

    rng = random.Random(ds_cfg.get("random_seed", 88))

    print(f"Loading dataset '{dataset_key}' via loader '{ds_cfg['loader']}' ...")
    normalised = loader(ds_cfg, rng)
    print(f"  Loaded {len(normalised)} raw records.")

    n = ds_cfg["n_examples"]
    if len(normalised) < n:
        raise ValueError(f"dataset '{dataset_key}' has only {len(normalised)} records; need {n}.")

    selected = _balanced_sample(normalised, n, rng)
    records = [_finalize(i, rec, rng) for i, rec in enumerate(selected)]

    # Report group balance (helps confirm balanced sampling worked).
    counts: dict[str, int] = defaultdict(int)
    for r in records:
        counts[r["subject"]] += 1
    print(f"  Group balance: {dict(sorted(counts.items()))}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(records)} instances to {output_path}")
    return records


def load_dataset_index(path: Path) -> list:
    """Load dataset_index.json and return list of instance dicts."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)
