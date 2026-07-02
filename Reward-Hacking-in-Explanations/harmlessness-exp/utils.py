"""Small shared helpers (clean local copies of the RHiE originals).

Kept self-contained so the pipeline has no import-path dependency on the parent
Reward-Hacking-in-Explanations package.
"""
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
