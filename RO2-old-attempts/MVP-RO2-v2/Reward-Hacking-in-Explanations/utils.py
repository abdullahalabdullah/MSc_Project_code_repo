import os
import random

import numpy as np
import torch
from transformers import set_seed as set_seed_trf
from accelerate.utils import set_seed as set_seed_acc

from transformers import enable_full_determinism

def set_seed(seed, full_determinism=False):
    print(f"Setting seed to {seed}")
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    set_seed_acc(seed)
    set_seed_trf(seed)

    if full_determinism:
        enable_full_determinism(seed)


def extract_visible_response(text):
    """Return the visible (post-</think>) part of a thinking-mode response.

    Qwen3 emits a <think>...</think> scratchpad before the final answer. We score /
    judge only the visible response (what a user sees), since reward models and the
    acknowledgment judge are not trained on long, backtracking <think> traces.

    Returns None when the think block is opened but never closed (response truncated
    mid-think by the token budget) — there is no visible answer in that case.
    """
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    if "<think>" in text:
        return None  # truncated mid-think: no post-think answer
    return text.strip()
