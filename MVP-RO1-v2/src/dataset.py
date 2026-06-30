"""
Dataset preparation for the MVP CoT monitorability evaluation.

Loads a dataset, filters it, samples a balanced subset (single- or two-level
uniform stratified sampling), assigns an incorrect hint target, and writes a
dataset_index.json. Every dataset is driven by the `datasets:` registry in
config/mvp_config.yaml and namespaced on disk by its key (see resolve_paths).

The pipeline is multiple-choice with a **variable option count of 3–5 (A–E)**:
each question keeps its own number of options. The current suite:

  mmlu_redux     — HF edinburgh-dawg/mmlu-redux, error_type=="ok", uniform by subject (4 opt)
  gpqa_diamond   — HF Idavidrein/gpqa (gpqa_diamond), uniform by domain, all 198 (4 opt)
  mathbench_*    — MathBench release zip, English single_choice (theory+application),
                   two-level uniform: subject-area -> topic (4 opt)
  compmath       — HF biancaraimondi/CompMath-MCQ, uniform by subtopic (3 opt)
  logieval       — HF (gated), options in {3,4,5}, two-level: reasoning-type -> task-format
  wmdp           — HF cais/wmdp, uniform by subdomain (4 opt)
  codemmlu       — HF Fsoft-AIC/CodeMMLU, options in [3,5], uniform by subset (3-5 opt)
"""

import io
import json
import os
import random
import re
import ssl
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

# Answer letters. The pipeline supports 3–5 options, so a question uses LETTERS[:k].
LETTERS = ["A", "B", "C", "D", "E"]
MAX_OPTIONS = len(LETTERS)
MIN_OPTIONS = 3


# ---------------------------------------------------------------------------
# HuggingFace authentication
# ---------------------------------------------------------------------------

def ensure_hf_auth(verbose: bool = True) -> bool:
    """Log into the HuggingFace Hub so gated assets (GPQA, LogiEval, gated models)
    download in any environment.

    Token resolution order (first hit wins):
      1. env var HF_TOKEN / HUGGING_FACE_HUB_TOKEN / HUGGINGFACE_HUB_TOKEN
      2. the `hf_token` field in config/mvp_config.yaml (committed for zero-setup;
         read-only — rotate before any public release)

    Returns False only if every source is empty AND login fails.
    """
    token = (os.environ.get("HF_TOKEN")
             or os.environ.get("HUGGING_FACE_HUB_TOKEN")
             or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    source = "env"
    if not token:
        cfg_path = Path(__file__).resolve().parent.parent / "config" / "mvp_config.yaml"
        if cfg_path.exists():
            try:
                import yaml
                cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
                token = cfg.get("hf_token")
                source = str(cfg_path)
            except Exception:
                token = None
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

def resolve_paths(root, config: dict, dataset_key: str, model_name: str | None = None,
                  hint_type: str | None = None) -> dict:
    """Resolve all on-disk paths for a (dataset, model, hint_type) triple.

    Every dataset is namespaced by its key, and — when a `hint_type` is given —
    the result dirs are further namespaced by the hint type, so different hints
    never overwrite each other:
        data/{ds}/dataset_index.json
        outputs/{ds}/{model}/{hint}/, monitor_outputs/{ds}/{model}/{hint}/,
        analysis/{ds}/{model}/{hint}/, plots/{ds}/{hint}/

    `model_name` is appended to the per-model dirs (outputs/monitor/analysis) only
    when provided. `hint_type` is appended to those dirs and to plots_dir when
    provided. `data_index` is never model- or hint-specific (the hint_target is
    assigned once at prep time and shared by every hint type).
    """
    root = Path(root)
    p = config["paths"]

    def base(d):
        return root / d / dataset_key

    def model_dir(d):
        b = base(d)
        if model_name:
            b = b / model_name
        if hint_type:
            b = b / hint_type
        return b

    plots_dir = base(p["plots_dir"])
    if hint_type:
        plots_dir = plots_dir / hint_type

    return {
        "data_index":  base(p["data_dir"]) / "dataset_index.json",
        "outputs_dir": model_dir(p["outputs_dir"]),
        "monitor_dir": model_dir(p["monitor_outputs_dir"]),
        "analysis_dir": model_dir(p["analysis_dir"]),
        "plots_dir":   plots_dir,
    }


# ---------------------------------------------------------------------------
# Normalisation helpers (shared by all loaders)
# ---------------------------------------------------------------------------

# A normalised record is a dict:
#   {"question": str, "options": [str, ...], "correct_idx": int,
#    "group": str, "subgroup": str | None, "source_id": str}
# `group` is the primary stratum; `subgroup` (optional) is the second level used by
# the two-level uniform samplers (MathBench, LogiEval).

# Matches an option label at the start of a choice string: "(A)" (no trailing space),
# "A.", or "A)". Trailing whitespace optional (some sources render "(A)4").
_LETTER_PREFIX_RE = re.compile(r"^\s*(?:\([A-E]\)|[A-E][.)])\s*")


def _clean(value) -> str:
    """Stringify an option/question cleanly (whole-number floats lose the .0)."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _strip_letter_prefix(choice: str) -> str:
    """Remove a leading '(A)' / 'A)' / 'A.' style option label if present."""
    return _LETTER_PREFIX_RE.sub("", str(choice)).strip()


def _finalize(instance_id: int, rec: dict, rng: random.Random) -> dict:
    """Turn a normalised record into a pipeline instance dict (3–5 options, A–E)."""
    options = rec["options"]
    correct_idx = rec["correct_idx"]
    k = len(options)
    assert MIN_OPTIONS <= k <= MAX_OPTIONS, f"{rec['source_id']}: {k} options (must be 3–5)"
    assert 0 <= correct_idx < k, f"{rec['source_id']}: bad correct_idx {correct_idx} for {k} options"

    letters = LETTERS[:k]
    correct_letter = letters[correct_idx]
    hint_target = rng.choice([l for l in letters if l != correct_letter])
    assert hint_target != correct_letter

    return {
        "instance_id": instance_id,
        "question_id": rec["source_id"],
        "subject": rec["group"],
        "question": _clean(rec["question"]),
        "choices": {letters[i]: _clean(options[i]) for i in range(k)},
        "correct_answer": correct_letter,
        "hint_target": hint_target,
    }


def _balanced_order(records: list, keys: list, rng: random.Random) -> list:
    """Return all records in a balanced (round-robin) order across `keys`.

    With one key this round-robins across that field's groups; with two keys each
    group is itself balanced-ordered by the second key before the round-robin, so
    taking a prefix yields uniform-by-key1-then-uniform-by-key2 sampling. Records
    within the deepest level are shuffled. Deterministic in `rng`.
    """
    if not keys:
        out = list(records)
        rng.shuffle(out)
        return out

    key = keys[0]
    groups: dict[str, list] = defaultdict(list)
    for r in records:
        groups[r.get(key) or "unknown"].append(r)

    ordered = {g: _balanced_order(items, keys[1:], rng) for g, items in groups.items()}
    iters = {g: iter(ordered[g]) for g in ordered}
    order = sorted(ordered)
    exhausted: set = set()
    out: list = []
    while len(exhausted) < len(order):
        for g in order:
            if g in exhausted:
                continue
            try:
                out.append(next(iters[g]))
            except StopIteration:
                exhausted.add(g)
    return out


def _stratified_sample(records: list, n: int, rng: random.Random, keys: list) -> list:
    """Uniformly sample n records across the grouping `keys` (1 or 2 levels).

    If n >= len(records) all records are returned (shuffled)."""
    if n >= len(records):
        out = list(records)
        rng.shuffle(out)
        return out
    return _balanced_order(records, keys, rng)[:n]


# ---------------------------------------------------------------------------
# Per-dataset loaders — each returns a list of normalised records (post-filter)
# ---------------------------------------------------------------------------

def _load_mmlu_redux(ds_cfg: dict, rng: random.Random) -> list:
    """MMLU-Redux: 30 subject configs, keep only error_type == 'ok'. 4 options."""
    from datasets import load_dataset as hf_load_dataset, get_dataset_config_names
    hf = ds_cfg["hf"]
    subjects = ds_cfg.get("subjects") or get_dataset_config_names(hf)
    split = ds_cfg.get("split", "test")
    recs = []
    for subject in subjects:
        ds = hf_load_dataset(hf, subject, split=split)
        for idx, row in enumerate(ds):
            if row.get("error_type") != "ok":
                continue
            recs.append({
                "question": row["question"],
                "options": [_clean(c) for c in row["choices"]],
                "correct_idx": int(row["answer"]),
                "group": subject,
                "subgroup": None,
                "source_id": f"{subject}_{idx}",
            })
    return recs


def _load_gpqa(ds_cfg: dict, rng: random.Random) -> list:
    """GPQA (e.g. gpqa_diamond): 4 options, grouped by High-level domain."""
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
            "subgroup": None,
            "source_id": f"gpqa_{row.get('Record ID', idx)}",
        })
    return recs


def _load_wmdp(ds_cfg: dict, rng: random.Random) -> list:
    """WMDP: bio/chem/cyber configs, 4 options, grouped by subdomain."""
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
                "subgroup": None,
                "source_id": f"{cfg}_{idx}",
            })
    return recs


def _load_compmath(ds_cfg: dict, rng: random.Random) -> list:
    """CompMath-MCQ: 3 options, grouped by subtopic."""
    from datasets import load_dataset as hf_load_dataset
    ds = hf_load_dataset(ds_cfg["hf"], split=ds_cfg.get("split", "test"))
    recs = []
    for idx, row in enumerate(ds):
        recs.append({
            "question": row["question"],
            "options": [_clean(o) for o in row["options"]],
            "correct_idx": int(row["correct_label"]),
            "group": row["subtopic"],
            "subgroup": None,
            "source_id": f"compmath_{idx}",
        })
    return recs


def _load_codemmlu(ds_cfg: dict, rng: random.Random) -> list:
    """CodeMMLU: 9 subsets, keep questions with 3–5 options, grouped by subset.

    The dataset's `answer` is a letter; a handful of items have 6–7 options (and
    answers F/G) — those fall outside the A–E pipeline and are dropped along with
    1–2 option questions."""
    from datasets import load_dataset as hf_load_dataset
    recs = []
    for sub in ds_cfg["subsets"]:
        ds = hf_load_dataset(ds_cfg["hf"], sub, split=ds_cfg.get("split", "test"))
        for idx, row in enumerate(ds):
            choices = row["choices"]
            k = len(choices)
            if not (MIN_OPTIONS <= k <= MAX_OPTIONS):
                continue
            correct_idx = ord(str(row["answer"]).strip().upper()) - ord("A")
            if not (0 <= correct_idx < k):       # answer letter beyond kept options
                continue
            recs.append({
                "question": row["question"],
                "options": [_clean(c) for c in choices],
                "correct_idx": correct_idx,
                "group": sub,
                "subgroup": None,
                "source_id": f"{sub}_{idx}",
            })
    return recs


def _load_logieval(ds_cfg: dict, rng: random.Random) -> list:
    """LogiEval (gated HF): keep options in {3,4,5}; two-level groups
    reasoning-type -> task-format.

    Field names are config-driven (LogiEval's gated schema names are confirmed at
    access time) with sensible defaults. `gold` may be a letter, an index, or the
    answer string — all three are handled."""
    from datasets import load_dataset as hf_load_dataset
    f = ds_cfg.get("fields", {})
    f_q = f.get("question", "question")
    f_opts = f.get("options", "options")
    f_ans = f.get("answer", "answer")
    f_reason = f.get("reasoning_type", "reasoning type")
    f_task = f.get("task_format", "task format")

    load_args = {"split": ds_cfg.get("split", "test")}
    if ds_cfg.get("config"):
        ds = hf_load_dataset(ds_cfg["hf"], ds_cfg["config"], **load_args)
    else:
        ds = hf_load_dataset(ds_cfg["hf"], **load_args)

    recs = []
    for idx, row in enumerate(ds):
        options = [_clean(o) for o in row[f_opts]]
        k = len(options)
        if k not in (3, 4, 5):
            continue
        correct_idx = _resolve_answer_idx(row[f_ans], options)
        if correct_idx is None or not (0 <= correct_idx < k):
            continue
        recs.append({
            "question": row[f_q],
            "options": options,
            "correct_idx": correct_idx,
            "group": str(row.get(f_reason, "unknown")),
            "subgroup": str(row.get(f_task, "unknown")),
            "source_id": f"logieval_{idx}",
        })
    return recs


def _resolve_answer_idx(gold, options: list) -> int | None:
    """Map a gold answer (letter 'A'.., int index, or the answer text) to an index."""
    if isinstance(gold, (list, tuple)):
        gold = gold[0] if gold else None
    if isinstance(gold, bool):
        return None
    if isinstance(gold, int):
        return gold
    s = str(gold).strip()
    if len(s) == 1 and s.upper() in LETTERS:
        return LETTERS.index(s.upper())
    if s.isdigit():
        return int(s)
    cleaned = [_strip_letter_prefix(o) for o in options]
    if s in options:
        return options.index(s)
    if s in cleaned:
        return cleaned.index(s)
    return None


# --- MathBench (downloaded release zip) ------------------------------------

def _mathbench_root(ds_cfg: dict) -> Path:
    """Download + extract mathbench_v1.zip once into a cache dir; return its path."""
    cache = ds_cfg.get("cache_dir")
    cache_dir = Path(cache) if cache else (Path(__file__).resolve().parent.parent
                                           / "data" / "_mathbench_raw")
    cache_dir.mkdir(parents=True, exist_ok=True)
    root = cache_dir / "mathbench_v1"
    if root.is_dir() and any(root.iterdir()):
        return root
    url = ds_cfg["url"]
    print(f"  Downloading MathBench archive from {url} ...")
    # Use certifi's CA bundle when available — the macOS python.org build ships
    # without system roots, so a bare urlopen() fails SSL verification.
    try:
        import certifi
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ssl_ctx = None
    with urllib.request.urlopen(url, context=ssl_ctx) as resp:
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(cache_dir)
    return root


def _load_mathbench(ds_cfg: dict, rng: random.Random) -> list:
    """One MathBench stage: English single_choice from both the application
    ({stage}) and theory ({stage}_knowledge) folders. The `topic` field is
    'Stage--SubjectArea--Topic'; group = subject area, subgroup = topic. 4 options."""
    root = _mathbench_root(ds_cfg)
    stage = ds_cfg["stage"]
    recs = []
    for folder in (stage, f"{stage}_knowledge"):
        path = root / folder / "single_choice_en.jsonl"
        if not path.exists():
            continue
        for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            options = [_clean(o) for o in row["options"]]
            correct_idx = LETTERS.index(str(row["answer"]).strip().upper())
            parts = [p.strip() for p in str(row.get("topic", "")).split("--")]
            subject_area = parts[1] if len(parts) > 1 else "unknown"
            topic = parts[2] if len(parts) > 2 else (parts[1] if len(parts) > 1 else "unknown")
            recs.append({
                "question": row["question"],
                "options": options,
                "correct_idx": correct_idx,
                "group": subject_area,
                "subgroup": topic,
                "source_id": f"mathbench_{stage}_{folder}_{idx}",
            })
    return recs


_LOADERS = {
    "mmlu_redux": _load_mmlu_redux,
    "gpqa": _load_gpqa,
    "wmdp": _load_wmdp,
    "compmath": _load_compmath,
    "codemmlu": _load_codemmlu,
    "logieval": _load_logieval,
    "mathbench": _load_mathbench,
}


# ---------------------------------------------------------------------------
# Top-level preparation entry point
# ---------------------------------------------------------------------------

def prepare_dataset(config: dict, output_path: Path, dataset_key: str,
                    n_examples: int | None = None) -> list:
    """Prepare a dataset_index.json for `dataset_key` and write it to output_path.

    All datasets are resolved through the `datasets:` registry. Sampling is
    uniform; two-level (group -> subgroup) when the loader sets `subgroup`.

    `n_examples` overrides the config `n_examples` budget for this dataset when
    given (e.g. a smaller 50-example run); otherwise the config value is used."""
    ensure_hf_auth()  # GPQA / LogiEval are gated; harmless no-op for public datasets

    datasets_cfg = config.get("datasets", {})
    if dataset_key not in datasets_cfg:
        raise ValueError(
            f"dataset '{dataset_key}' not in config datasets registry. "
            f"Available: {list(datasets_cfg)}"
        )

    ds_cfg = datasets_cfg[dataset_key]
    loader = _LOADERS.get(ds_cfg["loader"])
    if loader is None:
        raise ValueError(f"no loader '{ds_cfg['loader']}' registered for dataset '{dataset_key}'.")

    rng = random.Random(ds_cfg["random_seed"])

    print(f"Loading dataset '{dataset_key}' via loader '{ds_cfg['loader']}' ...")
    normalised = loader(ds_cfg, rng)
    print(f"  Loaded {len(normalised)} records (post-filter).")

    n = n_examples if n_examples is not None else ds_cfg["n_examples"]
    if len(normalised) == 0:
        raise ValueError(f"dataset '{dataset_key}': no records after filtering.")
    if len(normalised) < n:
        print(f"  NOTE: only {len(normalised)} records available; using all "
              f"(below the n_examples={n} budget).")

    # Two-level uniform when the loader supplies subgroups; otherwise single-level.
    keys = ["group"] + (["subgroup"] if any(r.get("subgroup") is not None for r in normalised) else [])
    selected = _stratified_sample(normalised, n, rng, keys)
    records = [_finalize(i, rec, rng) for i, rec in enumerate(selected)]

    # Report balance (confirms uniform sampling worked).
    by_group: dict[str, int] = defaultdict(int)
    for r in records:
        by_group[r["subject"]] += 1
    opt_counts: dict[int, int] = defaultdict(int)
    for r in records:
        opt_counts[len(r["choices"])] += 1
    print(f"  Group balance: {dict(sorted(by_group.items()))}")
    print(f"  Option-count distribution: {dict(sorted(opt_counts.items()))}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(records)} instances to {output_path}")
    return records


def load_dataset_index(path: Path) -> list:
    """Load dataset_index.json and return list of instance dicts."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)
