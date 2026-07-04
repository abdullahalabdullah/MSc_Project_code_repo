"""Stage 02 — Generate N Qwen3-4B responses per prompt (thinking mode ON).

Runs on GPU (vLLM). Prompts are processed in batches of GEN_BATCH_PROMPTS so that
truncation (finish_reason == "length") is reported to the terminal *as it happens*
rather than only at the very end.

Output: config.RESPONSES_PATH = list[SAMPLE] of list[N] dicts
        {prompt, full_response, finish_reason, num_tokens, truncated}
"""
import joblib
from vllm import LLM, SamplingParams

import config
import env_setup
from utils import set_seed


def main():
    env_setup.setup_env()
    config.ensure_output_dir()
    set_seed(config.SEED)

    prompts = joblib.load(config.PROMPTS_PATH)
    print(f"Loaded {len(prompts)} prompts from {config.PROMPTS_PATH}")

    # Qwen3 thinking-mode preset; n=N generates all samples per prompt in one pass.
    sampling_params = SamplingParams(
        temperature=config.TEMPERATURE,
        top_p=config.TOP_P,
        top_k=config.TOP_K,
        max_tokens=config.MAX_TOKENS,
        n=config.N,
        seed=config.SEED,
    )

    # Engine tuning for throughput on a large-VRAM GPU (mirrors MVP-RO1-v2):
    #   - bfloat16 weights
    #   - max_model_len capped to MAX_TOKENS + prompt budget -> higher KV concurrency
    #   - gpu_memory_utilization 0.90 -> more VRAM for the KV cache
    #   - prefix caching: every request shares the identical chat-template prefix, and the
    #     N samples of a prompt share its full prefix, so vLLM reuses that cached KV.
    llm = LLM(
        model=config.GEN_MODEL,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=config.MAX_MODEL_LEN,
        gpu_memory_utilization=config.GPU_MEMORY_UTILIZATION,
        enable_prefix_caching=True,
    )
    tokenizer = llm.get_tokenizer()

    # Render each prompt through the Qwen3 chat template with thinking enabled.
    rendered = []
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        rendered.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        )

    all_responses = [None] * len(prompts)
    total_samples = 0
    total_truncated = 0

    bs = config.GEN_BATCH_PROMPTS
    for start in range(0, len(prompts), bs):
        end = min(start + bs, len(prompts))
        batch_rendered = rendered[start:end]
        print(f"\n>> Generating prompts {start}..{end - 1} "
              f"({config.N} samples each, max_tokens={config.MAX_TOKENS}) — "
              f"one continuous-batched call")
        # Single generate call for the whole batch: vLLM continuous-batches all
        # (prompts x N) sequences together, so the GPU stays saturated and we pay one
        # decode drain-tail per batch. use_tqdm shows live progress within the batch.
        outputs = llm.generate(batch_rendered, sampling_params, use_tqdm=True)

        for local_ix, output in enumerate(outputs):
            global_ix = start + local_ix
            per_prompt = []
            for s, sample in enumerate(output.outputs):
                truncated = sample.finish_reason == "length"
                num_tokens = len(sample.token_ids)
                total_samples += 1
                if truncated:
                    total_truncated += 1
                    # Real-time truncation warning.
                    print(f"  ERROR: truncated — prompt {global_ix}, sample {s} "
                          f"({num_tokens} tok, finish_reason=length)")
                per_prompt.append({
                    "prompt": prompts[global_ix],
                    "full_response": sample.text,
                    "finish_reason": sample.finish_reason,
                    "num_tokens": num_tokens,
                    "truncated": truncated,
                })
            all_responses[global_ix] = per_prompt

        pct = 100.0 * total_truncated / max(total_samples, 1)
        print(f"  running truncated: {total_truncated} / {total_samples} ({pct:.1f}%)")

    joblib.dump(all_responses, config.RESPONSES_PATH, compress=True)
    print(f"\nSaved -> {config.RESPONSES_PATH}")
    pct = 100.0 * total_truncated / max(total_samples, 1)
    print(f"FINAL truncated: {total_truncated} / {total_samples} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
