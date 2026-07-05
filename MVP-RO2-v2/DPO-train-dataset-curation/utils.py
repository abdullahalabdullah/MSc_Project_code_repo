"""Small shared helpers (clean local copies of the RHiE originals).

Kept self-contained so the pipeline has no import-path dependency on the parent
Reward-Hacking-in-Explanations package.
"""
import json
import os
import random

import numpy as np


def set_seed(seed):
    """Seed python / numpy / torch (+ HF/accelerate if available)."""
    print(f"Setting seed to {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
    try:
        from transformers import set_seed as set_seed_trf
        set_seed_trf(seed)
    except ImportError:
        pass


def extract_visible_response(text):
    """Return the visible (post-</think>) part of a Qwen3 thinking-mode response.

    Qwen3 emits a <think>...</think> scratchpad before the final answer. We classify /
    score only the visible response (what a user actually sees).

    Returns None when the think block is opened but never closed (response truncated
    mid-think by the token budget) — there is no visible answer in that case.
    """
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    if "<think>" in text:
        return None  # truncated mid-think: no post-think answer
    return text.strip()


def think_answer_word_counts(text):
    """Split a Qwen3 response into (think_words, answer_words, has_think).

    - closed think block  -> words before </think> vs words after it
    - truncated mid-think -> all words counted as think, 0 answer, has_think=False
    - no think block      -> 0 think, all words answer
    Word counts (not tokens) so this needs no tokenizer and stays cheap at report time.
    """
    if "</think>" in text:
        think = text.split("</think>")[0].replace("<think>", "")
        answer = text.split("</think>")[-1]
        return len(think.split()), len(answer.split()), True
    if "<think>" in text:
        return len(text.replace("<think>", "").split()), 0, False
    return 0, len(text.split()), False


def update_diagnostics(path, section, payload):
    """Merge one named section into the run's consolidated diagnostics JSON.

    Each stage owns exactly one section, so re-running a stage overwrites only its slice
    and leaves the others intact (keeps the pipeline's stage independence). Creates the
    file on first write; tolerates a corrupt/partial file by starting fresh.
    """
    data = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            data = {}
    data[section] = payload
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
