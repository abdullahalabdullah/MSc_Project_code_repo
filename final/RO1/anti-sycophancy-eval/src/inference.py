"""
Qwen3 thinking-mode generation for the anti-sycophancy eval.

`load_qwen_model` / `split_thinking` follow the harmlessness eval's copies (offline
vLLM engine, prefix caching). One deliberate difference from the harmlessness eval:
NO system prompt and a single user turn — the RO2 curation pipeline renders its
prompts that way, and M1/M2 are an in-distribution manipulation check, so the eval
must sample from exactly the same prompt distribution.

Sampling comes from config["anti_sycophancy"]["inference"] (the curation spec's §2
config: temp 1.0 / top_p 0.95 / top_k 20 / max_new_tokens 16384), NOT from
qwen_inference — see the config block comment.
"""

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_qwen_model(model_id: str, inf_cfg: dict):
    """Load tokenizer + vLLM engine. Returns (tokenizer, llm)."""
    from vllm import LLM
    from transformers import AutoTokenizer

    max_model_len = inf_cfg.get("max_model_len", 18432)
    gpu_mem = inf_cfg.get("gpu_memory_utilization", 0.90)
    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(f"Loading model via vLLM: {model_id} "
          f"(max_model_len={max_model_len}, gpu_memory_utilization={gpu_mem})")
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem,
        enable_prefix_caching=True,
    )
    return tokenizer, llm


# ---------------------------------------------------------------------------
# Thinking / answer split (verbatim from harmlessness-eval/src/inference.py)
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
    """Single user turn, no system prompt (matches the curation pipeline)."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def build_challenge_prompt(tokenizer, turn1_prompt: str, turn1_answer: str,
                           challenge_user_msg: str, enable_thinking: bool = True) -> str:
    """Two-turn "Are you sure?" prompt (M3): the model's own turn-1 VISIBLE answer
    (post-</think>) is replayed as assistant history — Qwen3's template strips
    assistant-history think blocks anyway, so passing the visible text makes that
    explicit — then the challenge turn is appended."""
    messages = [
        {"role": "user", "content": turn1_prompt},
        {"role": "assistant", "content": turn1_answer},
        {"role": "user", "content": challenge_user_msg},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def generate_batch(llm, prompt_strs: list, sampling_cfg: dict, seeds: list) -> list:
    """One continuous-batched llm.generate over pre-rendered prompts.

    seeds is parallel to prompt_strs (per-request seed -> reproducible and identical
    across base/trained models, the paired-eval requirement). Returns per prompt:
    {raw_output, thinking, answer, token_count, truncated, seed}.
    """
    from vllm import SamplingParams

    sampling = [SamplingParams(
        n=1,
        temperature=sampling_cfg["temperature"],
        top_p=sampling_cfg["top_p"],
        top_k=sampling_cfg.get("top_k", -1),
        max_tokens=sampling_cfg["max_new_tokens"],
        seed=seed,
    ) for seed in seeds]

    outputs = llm.generate(prompt_strs, sampling)

    results = []
    for seed, out in zip(seeds, outputs):
        completion = out.outputs[0]     # n=1
        raw = completion.text
        thinking, answer = split_thinking(raw)
        results.append({
            "raw_output": raw,
            "thinking": thinking,
            "answer": answer,
            "token_count": len(completion.token_ids),
            "truncated": completion.finish_reason == "length",
            "seed": seed,
        })
    return results
