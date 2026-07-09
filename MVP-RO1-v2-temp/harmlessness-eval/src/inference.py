"""
Free-form Qwen3 (thinking-mode) generation for the harmlessness eval.

Self-contained: `load_qwen_model` and `split_thinking` are copied verbatim from
monitorability-eval's inference so the two evals stay decoupled (no cross-imports).
Unlike the monitorability path there is no MCQ / hint / answer-parsing — each
WildJailbreak prompt gets a single free-form response, and we keep the visible
answer (post-`</think>`) for the WildGuard classifier.
"""

_SYSTEM_PROMPT = "You are a helpful assistant."

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


# ---------------------------------------------------------------------------
# Model loading (copied from monitorability-eval/src/inference_qwen.py)
# ---------------------------------------------------------------------------

def load_qwen_model(model_id: str, max_model_len: int = 24576,
                    gpu_memory_utilization: float = 0.90):
    """Load tokenizer + vLLM engine for a Qwen3 model. Returns (tokenizer, llm).

    max_model_len 24576 = 16384 (max_new_tokens) + 8192 prompt budget. Prefix
    caching is a pure throughput win (shared system-prompt prefix) with no effect
    on generated tokens."""
    from vllm import LLM
    from transformers import AutoTokenizer

    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(f"Loading model via vLLM: {model_id} "
          f"(max_model_len={max_model_len}, gpu_memory_utilization={gpu_memory_utilization})")
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=True,
    )
    return tokenizer, llm


# ---------------------------------------------------------------------------
# Thinking / answer split (copied from monitorability-eval/src/inference_qwen.py)
# ---------------------------------------------------------------------------

def split_thinking(raw_output: str) -> tuple[str, str]:
    """Return (thinking, answer) — thinking is the <think> trace, answer is the
    visible text after </think>. Handles the truncated / no-open / no-tag shapes."""
    close_idx = raw_output.find(_THINK_CLOSE)
    if close_idx == -1:
        stripped = raw_output.lstrip()
        if stripped.startswith(_THINK_OPEN):
            return stripped[len(_THINK_OPEN):].strip(), ""
        return "", raw_output.strip()

    before = raw_output[:close_idx]
    after = raw_output[close_idx + len(_THINK_CLOSE):]
    open_idx = before.find(_THINK_OPEN)
    thinking = before[open_idx + len(_THINK_OPEN):] if open_idx != -1 else before
    return thinking.strip(), after.strip()


# ---------------------------------------------------------------------------
# Prompt construction + generation
# ---------------------------------------------------------------------------

def build_prompt(tokenizer, user_prompt: str, enable_thinking: bool = True) -> str:
    """Apply the Qwen3 chat template to a raw user message. No MCQ/hint content."""
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def generate_responses(tokenizer, llm, prompts: list, gen_cfg: dict,
                       sampling_cfg: dict) -> list:
    """Generate responses for a list of prompt records.

    `prompts`     — list of {prompt_id, subset, prompt}.
    `gen_cfg`     — config["harmlessness"]["generation"] (enable_thinking, n_samples).
    `sampling_cfg`— config["qwen_inference"] (temperature, top_p, top_k, max_new_tokens).

    Returns one record per (prompt, sample): {prompt_id, subset, prompt,
    sample_idx, raw_output, thinking, answer, token_count, truncated, seed}.
    The batch is issued in a single llm.generate call so vLLM continuous batching
    keeps the GPU saturated.
    """
    from vllm import SamplingParams

    enable_thinking = gen_cfg.get("enable_thinking", True)
    n_samples = gen_cfg.get("n_samples", 1)

    reqs = []          # (record, seed) metadata parallel to the prompt list
    prompt_strs = []
    sampling = []
    for rec in prompts:
        templated = build_prompt(tokenizer, rec["prompt"], enable_thinking)
        for s in range(n_samples):
            seed = 1000 + s      # fixed per sample index, reused across prompts
            prompt_strs.append(templated)
            sampling.append(SamplingParams(
                n=1,
                temperature=sampling_cfg["temperature"],
                top_p=sampling_cfg["top_p"],
                top_k=sampling_cfg.get("top_k", -1),
                max_tokens=sampling_cfg["max_new_tokens"],
                seed=seed,
            ))
            reqs.append((rec, s, seed))

    outputs = llm.generate(prompt_strs, sampling)

    results = []
    for (rec, s, seed), out in zip(reqs, outputs):
        completion = out.outputs[0]     # n=1
        raw = completion.text
        thinking, answer = split_thinking(raw)
        results.append({
            "prompt_id":   rec["prompt_id"],
            "subset":      rec["subset"],
            "prompt":      rec["prompt"],
            "sample_idx":  s,
            "raw_output":  raw,
            "thinking":    thinking,
            "answer":      answer,
            "token_count": len(completion.token_ids),
            "truncated":   completion.finish_reason == "length",
            "seed":        seed,
        })
    return results
