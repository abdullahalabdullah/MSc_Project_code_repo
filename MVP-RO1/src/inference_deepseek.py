"""
Inference pipeline for the MVP CoT monitorability evaluation — DeepSeek-R1-Distill variant.

`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` is a *reasoning* model distilled from
DeepSeek-R1 onto a Qwen2.5-1.5B base. Like the Qwen3 thinking path, it emits its
chain-of-thought inside a `<think>...</think>` block before the visible answer.
The `<think>`-handling logic is therefore reused verbatim from `src/inference_qwen.py`.

What is DIFFERENT from both `src/inference.py` (Llama) and `src/inference_qwen.py`
(Qwen3), and why this module exists rather than reusing either:

  1. **No system prompt.** The DeepSeek-R1 model card explicitly recommends placing
     all instructions in the *user* turn and using no system prompt. The Llama and
     Qwen paths both prepend `_SYSTEM_PROMPT = "You are a helpful assistant."`; here
     the messages list contains only the user turn.
  2. **Forced opening `<think>`.** The DeepSeek-R1 distilled models sometimes skip
     the opening `<think>` tag and start reasoning (or answering) directly, which
     degrades the reasoning quality and breaks CoT extraction. The model card
     recommends enforcing the response to begin with `<think>`. We therefore append
     `<think>\n` to the prompt (unless the chat template already opened the block).
     Because that opening tag then lives in the *prompt*, the model's completion
     begins *inside* the block; we prepend the tag back onto the stored `raw_output`
     so the record is self-contained and contains the full `<think>...</think>` —
     exactly like the Qwen path, so the monitor reads the complete reasoning.
  3. **No `enable_thinking` flag.** That kwarg is Qwen3-specific; the DeepSeek chat
     template does not use it, so `format_prompt_qwen` is not reused.
  4. **Sampling.** Reads `config["deepseek_inference"]` (temp 0.6, top_p 0.95, large
     `max_new_tokens`) — the DeepSeek-R1 recommended thinking-mode settings.

The JSONL schema is identical to the Qwen path (a superset of the Llama path's,
adding the `thinking` field), so scripts 03–06 consume DeepSeek records unchanged.
"""

import json
import sys
from pathlib import Path

# Model-agnostic prompt body (the question block + Stanford-professor hint line are
# byte-for-byte identical across all models) comes from the base module. The
# <think>-block handling is identical to Qwen3, so it is reused as-is.
from src.inference import build_user_content
from src.inference_qwen import (
    _THINK_OPEN,
    split_thinking,
    extract_cot_qwen,
    parse_answer_qwen,
    is_no_cot,  # noqa: F401  (kept for parity with the Qwen module's public surface)
)


# ---------------------------------------------------------------------------
# Prompt construction (DeepSeek-R1-Distill thinking mode)
# ---------------------------------------------------------------------------

def format_prompt_deepseek(item: dict, include_hint: bool, tokenizer) -> str:
    """Apply the DeepSeek-R1-Distill chat template and return the prompt string.

    Two DeepSeek-specific choices vs. the Llama/Qwen paths:
      * **No system prompt** — the messages list contains only the user turn
        (DeepSeek-R1 model-card recommendation).
      * **Forced opening `<think>`** — `<think>\\n` is appended to the prompt so the
        model always opens a reasoning block (the distilled models otherwise skip
        it). If the chat template already ends with an open `<think>`, it is left
        alone to avoid a doubled tag.

    The user content (question + optional hint) is byte-for-byte identical to the
    Llama/Qwen pipelines, so the only cross-model differences are the template, the
    absence of a system prompt, and the forced think tag.
    """
    messages = [
        {"role": "user", "content": build_user_content(item, include_hint)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if not prompt.rstrip().endswith(_THINK_OPEN):
        prompt = prompt + _THINK_OPEN + "\n"
    return prompt


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------

def run_inference_deepseek(config: dict, model_cfg: dict, dataset: list, base_seed: int = 88):
    """
    Run hinted and unhinted inference for one DeepSeek-R1-Distill model.

    Reads sampling parameters from `config["deepseek_inference"]` (NOT the shared
    `inference:` block, which is for the Llama/Tulu pipeline, nor `qwen_inference:`).

    Writes:
      outputs/{model_name}/hinted.jsonl
      outputs/{model_name}/unhinted.jsonl
      outputs/{model_name}/progress.json  (resume checkpoint)

    Each JSONL line carries the same fields as the Qwen path:
      instance_id, question_id, arm, sample_idx, hint_target, correct_answer,
      full_prompt, raw_output, thinking, cot, parsed_answer, followed_hint,
      token_count, truncated, seed

    `raw_output` is the model completion with the forced opening `<think>` tag
    prepended back, so it contains the full `<think>...</think>` block.
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    model_name = model_cfg["name"]
    hf_id = model_cfg["hf_id"]
    inf = config["deepseek_inference"]
    n_samples = inf["n_samples"]
    checkpoint_every = inf["checkpoint_every"]

    out_dir = Path(config["paths"]["outputs_dir"]) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    hinted_path = out_dir / "hinted.jsonl"
    unhinted_path = out_dir / "unhinted.jsonl"
    progress_path = out_dir / "progress.json"

    # Resume: load already-completed instance IDs
    completed: set = set()
    if progress_path.exists():
        with open(progress_path) as f:
            completed = set(json.load(f).get("completed_instance_ids", []))
        print(f"Resuming {model_name}: {len(completed)} instances already done.")

    print(f"Loading tokenizer: {hf_id}")
    tokenizer = AutoTokenizer.from_pretrained(hf_id)

    print(f"Loading model via vLLM: {hf_id}")
    llm = LLM(
        model=hf_id,
        dtype="bfloat16",
        trust_remote_code=True,
    )

    todo = [item for item in dataset if item["instance_id"] not in completed]
    print(f"{len(todo)} instances to process for {model_name} (DeepSeek-R1-Distill, forced <think>).")

    null_count = 0
    truncated_count = 0
    total_generated = 0

    # Seed scheme matches src/inference.py: the hinted and unhinted arms draw from
    # DISJOINT lanes (arm_offset 0 vs 1_000_000) so the two arms are independent by
    # construction (the TE = p1 - p0 analysis assumes Cov(p0, p1) = 0). The
    # accuracy-estimation run (scripts/07_*) uses higher lanes (+2_000_000 /
    # +3_000_000) so it never collides on the same model + dataset.
    _ARM_OFFSET = {"unhinted": 0, "hinted": 1_000_000}

    def _seed(item, s, arm):
        return base_seed + _ARM_OFFSET[arm] + item["instance_id"] * n_samples + s

    def _sampling_params(item, s, arm):
        return SamplingParams(
            n=1,
            temperature=inf["temperature"],
            top_p=inf["top_p"],
            max_tokens=inf["max_new_tokens"],
            seed=_seed(item, s, arm),
        )

    # The forced opening tag lives in the prompt, so each completion starts INSIDE
    # the think block. Prepend it back so raw_output is self-contained.
    _RAW_PREFIX = _THINK_OPEN + "\n"

    for batch_start in range(0, len(todo), checkpoint_every):
        batch = todo[batch_start : batch_start + checkpoint_every]

        hinted_prompts = [
            format_prompt_deepseek(item, include_hint=True, tokenizer=tokenizer)
            for item in batch
        ]
        unhinted_prompts = [
            format_prompt_deepseek(item, include_hint=False, tokenizer=tokenizer)
            for item in batch
        ]

        with open(hinted_path, "a", encoding="utf-8") as hf, \
             open(unhinted_path, "a", encoding="utf-8") as uf:

            for s in range(n_samples):
                hinted_sp   = [_sampling_params(item, s, "hinted")   for item in batch]
                unhinted_sp = [_sampling_params(item, s, "unhinted") for item in batch]

                print(
                    f"  Batch [{batch_start}:{batch_start+len(batch)}] "
                    f"— sample {s+1}/{n_samples} — hinted arm ..."
                )
                hinted_outputs = llm.generate(hinted_prompts, hinted_sp)
                print(
                    f"  Batch [{batch_start}:{batch_start+len(batch)}] "
                    f"— sample {s+1}/{n_samples} — unhinted arm ..."
                )
                unhinted_outputs = llm.generate(unhinted_prompts, unhinted_sp)

                for item, h_prompt, u_prompt, h_out, u_out in zip(
                    batch, hinted_prompts, unhinted_prompts, hinted_outputs, unhinted_outputs
                ):
                    for arm_label, arm_prompt, arm_out, arm_file in [
                        ("hinted",   h_prompt, h_out, hf),
                        ("unhinted", u_prompt, u_out, uf),
                    ]:
                        completion = arm_out.outputs[0]  # n=1
                        # Prepend the forced opening tag so raw_output holds the full
                        # <think>...</think> block (the tag itself was in the prompt).
                        raw = _RAW_PREFIX + completion.text
                        parsed = parse_answer_qwen(raw)
                        trunc = completion.finish_reason == "length"
                        token_count = len(completion.token_ids)

                        if parsed is None:
                            null_count += 1
                        if trunc:
                            truncated_count += 1
                            print(
                                f"  ERROR: truncated output — "
                                f"instance {item['instance_id']}, {arm_label}, sample {s}",
                                file=sys.stderr,
                            )

                        thinking_text, _ = split_thinking(raw)
                        cot_text = extract_cot_qwen(raw)
                        record = {
                            "instance_id":   item["instance_id"],
                            "question_id":   item["question_id"],
                            "arm":           arm_label,
                            "sample_idx":    s,
                            "hint_target":   item["hint_target"],
                            "correct_answer": item["correct_answer"],
                            "full_prompt":   arm_prompt,
                            "raw_output":    raw,
                            "thinking":      thinking_text,
                            "cot":           cot_text,
                            "parsed_answer": parsed,
                            "followed_hint": (parsed == item["hint_target"]) if parsed else False,
                            "token_count":   token_count,
                            "truncated":     trunc,
                            "seed":          _seed(item, s, arm_label),
                        }
                        arm_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                        total_generated += 1

        completed.update(item["instance_id"] for item in batch)

        with open(progress_path, "w") as f:
            json.dump({"completed_instance_ids": sorted(completed)}, f)
        print(f"  Checkpoint saved: {len(completed)}/{len(dataset)} instances complete.")

    hinted_total = len(dataset) * n_samples
    null_rate = null_count / max(hinted_total * 2, 1)  # both arms
    if null_rate > 0.05:
        print(
            f"WARNING: answer parse failure rate {null_rate:.1%} > 5% threshold. "
            "If high, increase deepseek_inference.max_new_tokens (truncated thinking) "
            "or check the answer-format instruction in the prompt.",
            file=sys.stderr,
        )

    print(
        f"Inference complete for {model_name}. "
        f"Total outputs: {total_generated}. "
        f"Parse failures: {null_count} ({null_rate:.1%}). "
        f"Truncated: {truncated_count}."
    )
