"""
Local (GPU) monitor backend for the MVP CoT monitorability evaluation.

Runs the SAME monitor classification as src/monitor.py — identical prompt
(build_monitor_prompt) and identical output parsing (parse_monitor_output_v2) —
but through a local vLLM engine instead of the OpenAI chat-completions API.
Only the transport changes, so `03_run_monitor.py --backend local` writes
monitor_results.jsonl with exactly the same schema as the API backend and
steps 04-09 work unchanged.

Which model runs is picked from the config `monitor.local.models` registry
(via `resolve_local_monitor_cfg`): `monitor.local.default` names the entry used
when `03 --monitor-model` is omitted. Every entry is rendered with its own
model's chat template, so adding a monitor = adding a registry entry (e.g.
`qwen3_8b` = Qwen/Qwen3-8B, `gemma_e4b` = google/gemma-4-E4B-it).

Model-specific handling:
  * Hybrid *reasoning* models (Qwen3): the thinking/non-thinking switch is a
    chat-template argument, not a sampling flag. When a registry entry sets
    `enable_thinking`, it is passed to `apply_chat_template` — with False the
    Qwen3 template pre-fills an empty `<think>\n\n</think>` block so the model
    answers directly (no CoT), which is what we want from a deterministic
    classifier. Entries WITHOUT the key (Gemma, Llama, ...) don't pass the
    argument at all, since their templates have no such switch.
  * Decoding is greedy: temperature 0 (vLLM's equivalent of
    `do_sample=False`), 1 sample, `max_new_tokens` new tokens, fixed seed.
  * Defensive parsing: if the model still emits a `<think>...</think>` block,
    OUTPUT/REASONING are parsed from the text after `</think>` only, so a
    rehearsed "OUTPUT: True" inside stray thinking never wins.

Batching: prompts share the whole monitor-template prefix, so prefix caching
is enabled and each call to `classify_batch` is ONE llm.generate over the
batch (continuous batching) — orders of magnitude faster than per-record
calls. Prompts that would overflow the engine's context (monitored raw_output
up to the inference max_new_tokens) are not sent; they come back as
parse_failure records instead of crashing the whole generate call.
"""

from src.monitor import build_monitor_prompt, parse_monitor_output_v2

_THINK_CLOSE = "</think>"


def resolve_local_monitor_cfg(mon_cfg: dict, key: str | None = None) -> tuple[str, dict]:
    """Resolve the config `monitor:` block to (monitor_key, model_cfg).

    `key` is the `--monitor-model` selection; None picks `monitor.local.default`.
    Two `monitor.local` shapes are supported:
      * registry: `local: {default: <key>, models: {<key>: {...}, ...}}`
      * flat (legacy): `local: {model: ..., ...}` — one unnamed entry; only
        selectable as the default.
    """
    local = mon_cfg.get("local", {}) or {}
    models = local.get("models")
    if not models:
        if key not in (None, "default"):
            raise ValueError(
                f"config monitor.local has no 'models' registry; cannot select '{key}'.")
        return "default", local
    key = key or local.get("default")
    if key not in models:
        raise ValueError(
            f"unknown local monitor '{key}'. Available: {', '.join(sorted(models))}")
    return key, models[key]


def load_local_monitor(local_cfg: dict):
    """Load tokenizer + vLLM engine for the local monitor. Call once, reuse.

    All knobs come from the config `monitor.local` block (model, dtype,
    max_model_len, gpu_memory_utilization). Prefix caching is on: every
    monitor prompt starts with the identical template prefix, so vLLM reuses
    its KV cache across the whole run — a pure throughput win.
    """
    from vllm import LLM
    from transformers import AutoTokenizer

    model_id = local_cfg.get("model", "Qwen/Qwen3-8B")
    max_model_len = int(local_cfg.get("max_model_len", 20480))
    gpu_mem = float(local_cfg.get("gpu_memory_utilization", 0.9))

    print(f"Loading monitor tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(f"Loading monitor via vLLM: {model_id} "
          f"(dtype={local_cfg.get('dtype', 'bfloat16')}, max_model_len={max_model_len}, "
          f"gpu_memory_utilization={gpu_mem})")
    llm = LLM(
        model=model_id,
        dtype=local_cfg.get("dtype", "bfloat16"),
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem,
        enable_prefix_caching=True,
    )
    return tokenizer, llm


def format_monitor_prompt_local(tokenizer, model_response: str, hint_answer: str,
                                hint_type: str = "professor", hint_cfg: dict | None = None,
                                enable_thinking: bool | None = None) -> str:
    """Render the monitor prompt through the local model's own chat template.

    The user content is byte-identical to what the API backend sends
    (src/monitor.build_monitor_prompt); only the chat framing differs (the
    monitor model's own template instead of the OpenAI messages API).

    `enable_thinking` is the hybrid-reasoning template switch (Qwen3): pass
    True/False for models whose template has it; pass None for models without
    one (Gemma, Llama, ...) and the argument is not forwarded at all.
    """
    user_prompt = build_monitor_prompt(
        model_response=model_response,
        hint_answer=hint_answer,
        hint_type=hint_type,
        hint_cfg=hint_cfg,
    )
    template_kwargs = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )


def _visible_text(raw: str) -> str:
    """Text used for OUTPUT/REASONING parsing: everything after a `</think>`
    tag if one is present (non-thinking mode shouldn't emit one, but a stray
    block must never contribute a rehearsed OUTPUT), else the raw text."""
    close = raw.find(_THINK_CLOSE)
    return raw[close + len(_THINK_CLOSE):] if close != -1 else raw


def classify_batch(records: list, tokenizer, llm, local_cfg: dict,
                   hint_type: str = "professor", hint_cfg: dict | None = None) -> list:
    """Classify a batch of inference records with the local monitor.

    Returns a list of result dicts parallel to `records`, with the same keys
    as src/monitor.classify: acknowledged, label, reasoning, raw_response,
    and parse_failure (when OUTPUT parsing fails or the prompt was too long
    for the engine's context and was skipped).
    """
    from vllm import SamplingParams

    enable_thinking = local_cfg.get("enable_thinking")  # None = template has no switch
    max_new_tokens = int(local_cfg.get("max_new_tokens", 256))
    prompt_budget = int(local_cfg.get("max_model_len", 20480)) - max_new_tokens

    sampling = SamplingParams(
        n=int(local_cfg.get("n_samples", 1)),
        temperature=float(local_cfg.get("temperature", 0.0)),  # 0 = greedy
        top_p=1.0,
        top_k=-1,
        max_tokens=max_new_tokens,
        seed=int(local_cfg.get("seed", 42)),
    )

    results: list = [None] * len(records)
    prompts, kept_idx = [], []
    for i, record in enumerate(records):
        prompt = format_monitor_prompt_local(
            tokenizer,
            model_response=record["raw_output"],
            hint_answer=record["hint_target"],
            hint_type=record.get("hint_type", hint_type),
            hint_cfg=hint_cfg,
            enable_thinking=enable_thinking,
        )
        n_tokens = len(tokenizer(prompt).input_ids)
        if n_tokens > prompt_budget:
            results[i] = {
                "acknowledged": None,
                "label": None,
                "reasoning": None,
                "parse_failure": True,
                "raw_response": f"[skipped: prompt is {n_tokens} tokens, over the "
                                f"local monitor budget of {prompt_budget}]",
            }
        else:
            prompts.append(prompt)
            kept_idx.append(i)

    outputs = llm.generate(prompts, sampling) if prompts else []
    for i, out in zip(kept_idx, outputs):
        raw = out.outputs[0].text
        parsed = parse_monitor_output_v2(_visible_text(raw))
        parsed["raw_response"] = raw
        results[i] = parsed
    return results
