"""Stage B/OOD (02) — build Eval set 2 from SycophancyEval "Are you sure?" (CPU + network).

Spec §4.1: download the released are_you_sure.jsonl (Sharma et al., 2023) and keep every
record with an objectively gradable gold answer, otherwise unmodified. The source file
mixes base datasets with heterogeneous shapes, so the loader introspects each record:

  - MCQ records (base.correct_letter present): gold = the correct option's text when it
    can be extracted from base.answers (a "\n(A) ...\n(B) ..." block), else the letter;
    gold_aliases = letter, "(letter)", and the option text. The question is the human
    turn verbatim (it includes the options), so the model may answer with either form.
  - Freeform records: gold = base.correct_answer / base.answer; aliases = [gold].
  - Anything without a gradable gold is dropped and counted (per base dataset).

Two deliberate deviations, recorded in eval_ood_provenance.json: (i) records sometimes
carry an assistant prefill turn ('The answer is (') — dropped, since generation is
thinking-mode with a Final-answer line, not a prefill completion; (ii) the answer-format
instruction is appended at eval time by the RO1 eval code, not stored here.

Output: reserved/eval_ood_sycophancyeval.jsonl
        {question_id, question, gold, gold_aliases, source, base_dataset}
"""
import argparse
import datetime
import hashlib
import json
import re

import config
import env_setup
from utils import check_stage_cache, update_diagnostics, write_jsonl, write_stage_marker

STAGE = "02_build_eval_ood"

_OPTION_RE = re.compile(r"\(([A-Z])\)\s*([^\n(]+)")


def _download(url: str) -> bytes:
    """Fetch url. Prefers requests (ships with huggingface_hub; proper CA bundle),
    falling back to urllib with certifi — bare urllib fails TLS verification on
    python.org macOS installs."""
    try:
        import requests
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        return resp.content
    except ImportError:
        import ssl
        import urllib.request
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ctx = ssl.create_default_context()
        with urllib.request.urlopen(url, timeout=120, context=ctx) as resp:
            return resp.read()


def _human_turn(rec):
    for turn in rec.get("prompt", []):
        if turn.get("type") == "human" and isinstance(turn.get("content"), str):
            return turn["content"]
    return None


def _gold_from_base(base):
    """(gold, aliases) from a base record, or None if not objectively gradable."""
    letter = base.get("correct_letter")
    if isinstance(letter, str) and letter.strip():
        letter = letter.strip().strip("()")
        aliases = [letter, f"({letter})"]
        gold = letter
        answers_block = base.get("answers")
        if isinstance(answers_block, str):
            for m in _OPTION_RE.finditer(answers_block):
                if m.group(1) == letter:
                    option_text = m.group(2).strip().rstrip(".")
                    if option_text:
                        gold = option_text
                        aliases.append(option_text)
                    break
        return gold, list(dict.fromkeys(aliases))
    for key in ("correct_answer", "answer", "correct_solution"):
        v = base.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip(), [v.strip()]
        if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
            return v[0].strip(), [x.strip() for x in v if x.strip()]
    return None


def main():
    ap = argparse.ArgumentParser(description="Stage B — build the OOD eval file.")
    ap.add_argument("--force", action="store_true", help="Re-run even if cached.")
    args = ap.parse_args()

    env_setup.setup_env()
    config.ensure_output_dirs()

    params = {"url": config.OOD_EVAL_URL, "loader_version": 1}
    outputs = [config.EVAL_OOD_PATH, config.EVAL_OOD_PROVENANCE_PATH]
    if check_stage_cache(STAGE, params, outputs, force=args.force):
        return

    print(f"Downloading {config.OOD_EVAL_URL} ...")
    raw_bytes = _download(config.OOD_EVAL_URL)
    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    lines = raw_bytes.decode("utf-8").splitlines()
    print(f"Downloaded {len(lines)} lines (sha256 {sha256[:16]}...)")

    kept, dropped = [], {}
    dropped_reasons = {"no_gradable_gold": 0, "no_human_turn": 0, "bad_json": 0}
    prefill_dropped = 0
    for idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            dropped_reasons["bad_json"] += 1
            continue
        base = rec.get("base", {}) or {}
        base_dataset = base.get("dataset", "unknown")
        question = _human_turn(rec)
        if question is None:
            dropped_reasons["no_human_turn"] += 1
            dropped[base_dataset] = dropped.get(base_dataset, 0) + 1
            continue
        gold_result = _gold_from_base(base)
        if gold_result is None:
            dropped_reasons["no_gradable_gold"] += 1
            dropped[base_dataset] = dropped.get(base_dataset, 0) + 1
            continue
        if any(t.get("type") == "ai" for t in rec.get("prompt", [])):
            prefill_dropped += 1  # assistant prefill turn dropped (see module docstring)
        gold, aliases = gold_result
        kept.append({
            "question_id": str(rec.get("metadata", {}).get("id") or idx),
            "question": question,
            "gold": gold,
            "gold_aliases": aliases,
            "source": "sycophancyeval",
            "base_dataset": base_dataset,
        })

    if not kept:
        raise SystemExit("ERROR: no gradable records extracted from are_you_sure.jsonl — "
                         "source format may have changed; inspect the download.")
    # question_id must be unique (line index fallback guarantees this unless the
    # source's own metadata ids collide).
    ids = [r["question_id"] for r in kept]
    if len(set(ids)) != len(ids):
        print("WARNING: duplicate source ids — falling back to line-index ids")
        for i, r in enumerate(kept):
            r["question_id"] = str(i)

    write_jsonl(config.EVAL_OOD_PATH, kept)

    per_dataset = {}
    for r in kept:
        per_dataset[r["base_dataset"]] = per_dataset.get(r["base_dataset"], 0) + 1
    provenance = {
        "url": config.OOD_EVAL_URL,
        "sha256": sha256,
        "downloaded_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "total_lines": len(lines),
        "kept": len(kept),
        "kept_per_base_dataset": per_dataset,
        "dropped_per_base_dataset": dropped,
        "dropped_reasons": dropped_reasons,
        "records_with_assistant_prefill_dropped": prefill_dropped,
        "notes": ("Assistant prefill turns ('The answer is (') are dropped; the RO1 eval "
                  "appends the shared answer-format instruction at inference time. "
                  "Otherwise records are the published benchmark verbatim."),
    }
    with open(config.EVAL_OOD_PROVENANCE_PATH, "w") as f:
        json.dump(provenance, f, indent=2, ensure_ascii=False)

    print(f"Kept {len(kept)} gradable records -> {config.EVAL_OOD_PATH}")
    print(f"  per base dataset: {per_dataset}")
    print(f"  dropped: {dropped_reasons} (per dataset: {dropped})")

    update_diagnostics(config.DIAGNOSTICS_PATH, "stage_b_ood", provenance)
    write_stage_marker(STAGE, params, {"kept": len(kept)})


if __name__ == "__main__":
    main()
