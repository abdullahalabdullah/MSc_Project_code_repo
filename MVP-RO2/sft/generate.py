"""Thinking-mode generation sanity check (spec §9.1).

Load base model + a LoRA adapter checkpoint and generate with enable_thinking=True.
Confirms the model still emits a non-trivial <think>...</think> block — if SFT has
collapsed the reasoning behaviour, the CoT block disappears or becomes tiny, and
we should not hand that checkpoint to evaluation.
"""

import logging
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

DEFAULT_PROMPTS = [
    "What is 17 × 23? Show your reasoning.",
    "A train travels 60 km in 45 minutes. What is its average speed in km/h?",
]


def generation_check(
    base_model_id: str,
    adapter_dir: str,
    tokenizer,
    logger: logging.Logger,
    prompts=None,
    max_new_tokens: int = 512,
    min_think_tokens: int = 20,
    do_sample: bool = False,
    temperature: float = 0.6,
    top_p: float = 0.95,
) -> dict:
    """Generate for a few prompts; report whether <think> blocks are preserved.

    `do_sample=True` (temp/top_p) matches the RO1 Qwen3 eval settings and avoids
    greedy repetition loops; the default greedy is fine for a quick yes/no.
    """
    prompts = prompts or DEFAULT_PROMPTS
    base_only = adapter_dir is None or str(adapter_dir).lower() in {"base", "none", ""}
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if base_only:
        logger.info("Loading BASE model %s only (no adapter) for reference check ...",
                    base_model_id)
        model = base
        adapter_dir = "BASE (no adapter)"
    else:
        logger.info("Loading base %s + adapter %s for generation check ...",
                    base_model_id, adapter_dir)
        model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()

    results = []
    all_ok = True
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
        if do_sample:
            gen_kwargs.update(temperature=temperature, top_p=top_p)
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        decoded = tokenizer.decode(gen_ids, skip_special_tokens=False)

        # Prefer a closed <think>...</think>; if the block was truncated by
        # max_new_tokens (open <think> with no close), count what's there so we
        # don't falsely report "suppressed" for a model that thinks a lot.
        m = _THINK_RE.search(decoded)
        if m:
            think_text = m.group(1).strip()
        else:
            m_open = re.search(r"<think>(.*)$", decoded, re.DOTALL)
            think_text = m_open.group(1).strip() if m_open else ""
        has_think = "<think>" in decoded
        n_think_tokens = len(tokenizer(think_text, add_special_tokens=False)["input_ids"]) if think_text else 0
        ok = n_think_tokens >= min_think_tokens
        all_ok = all_ok and ok

        logger.info("-" * 70)
        logger.info("PROMPT: %s", prompt)
        logger.info("has <think>: %s | think tokens: %d | ok(>=%d): %s",
                    has_think, n_think_tokens, min_think_tokens, ok)
        logger.info("GENERATION:\n%s", decoded)
        results.append({"prompt": prompt, "has_think": has_think,
                        "n_think_tokens": n_think_tokens, "ok": ok})

    logger.info("=" * 70)
    logger.info("THINKING-MODE CHECK: %s (adapter=%s)",
                "PASS" if all_ok else "WARN — CoT may be suppressed", adapter_dir)
    logger.info("=" * 70)
    return {"adapter_dir": adapter_dir, "all_ok": all_ok, "per_prompt": results}


def load_tokenizer_for_gen(base_model_id: str):
    """Convenience tokenizer loader for the standalone generation script."""
    return AutoTokenizer.from_pretrained(base_model_id)
