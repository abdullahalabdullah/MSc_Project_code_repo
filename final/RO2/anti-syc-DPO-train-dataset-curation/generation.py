"""Shared vLLM generation core for the two GPU stages (03 Filter-0, 07 Filter-1).

Mirrors the harmlessness module's stage 02 engine setup exactly (offline LLM engine,
bfloat16, capped max_model_len, prefix caching, big continuous-batched generate calls)
with the spec §2 sampling config, plus mid-stage resume: each prompt's N samples are
flushed to a fingerprint-guarded .ckpt.jsonl as their batch completes, so an
interrupted run resumes instead of regenerating (input/config changes archive the
checkpoint as .stale and start fresh).
"""
import json
import os

import config
from utils import (finalize_ckpt, init_ckpt, input_fingerprint, load_ckpt,
                   update_diagnostics, write_jsonl)


def generate_samples(items, key_field, prompt_field, n_samples,
                     ckpt_path, out_path, diag_section, stage_params):
    """Generate n_samples completions per item and write them to out_path (JSONL).

    items: list of dicts, each with a unique `key_field` and the fully rendered
    user-message text in `prompt_field`. Output records: {key_field, "samples":
    [{full_text, finish_reason, num_tokens, truncated} x n_samples]}, in input order.
    """
    ids = [it[key_field] for it in items]
    fingerprint = input_fingerprint(ids, stage_params)
    done = load_ckpt(ckpt_path, fingerprint, key_field)
    pending = [it for it in items if it[key_field] not in done]
    print(f"{len(items)} items: {len(done)} already generated, {len(pending)} pending")

    total_samples = total_truncated = total_tokens = max_tokens = 0
    min_tokens = None

    if pending:
        # GPU imports only when there is real work — a fully resumed/cached invocation
        # of the calling stage never needs vLLM (or a GPU) at all.
        from vllm import LLM, SamplingParams

        sampling_params = SamplingParams(
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            top_k=config.TOP_K,
            max_tokens=config.MAX_NEW_TOKENS,
            n=n_samples,
            seed=config.GEN_SEED,
        )
        llm = LLM(
            model=config.GEN_MODEL,
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=config.MAX_MODEL_LEN,
            gpu_memory_utilization=config.GPU_MEMORY_UTILIZATION,
            max_num_seqs=config.MAX_NUM_SEQS,
            enable_prefix_caching=True,
        )
        tokenizer = llm.get_tokenizer()

        rendered = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": it[prompt_field]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,   # spec §2: thinking mode ON
            )
            for it in pending
        ]

        ckpt_f = init_ckpt(ckpt_path, fingerprint)
        try:
            bs = config.GEN_BATCH_PROMPTS
            for start in range(0, len(pending), bs):
                end = min(start + bs, len(pending))
                print(f"\n>> Generating items {start}..{end - 1} of {len(pending)} pending "
                      f"({n_samples} samples each, max_tokens={config.MAX_NEW_TOKENS}) — "
                      f"one continuous-batched call")
                outputs = llm.generate(rendered[start:end], sampling_params, use_tqdm=True)

                for local_ix, output in enumerate(outputs):
                    item = pending[start + local_ix]
                    samples = []
                    for s, sample in enumerate(output.outputs):
                        truncated = sample.finish_reason == "length"
                        num_tokens = len(sample.token_ids)
                        total_samples += 1
                        total_tokens += num_tokens
                        min_tokens = num_tokens if min_tokens is None else min(min_tokens, num_tokens)
                        max_tokens = max(max_tokens, num_tokens)
                        if truncated:
                            total_truncated += 1
                            print(f"  WARNING: truncated — {item[key_field]}, sample {s} "
                                  f"({num_tokens} tok, finish_reason=length)")
                        samples.append({
                            "full_text": sample.text,
                            "finish_reason": sample.finish_reason,
                            "num_tokens": num_tokens,
                            "truncated": truncated,
                        })
                    record = {key_field: item[key_field], "samples": samples}
                    done[item[key_field]] = record
                    ckpt_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                ckpt_f.flush()
                os.fsync(ckpt_f.fileno())
                pct = 100.0 * total_truncated / max(total_samples, 1)
                print(f"  running truncated: {total_truncated} / {total_samples} ({pct:.1f}%)")
        finally:
            ckpt_f.close()

    # Final output in input order; checkpoint removed once this is safely on disk.
    records = [done[i] for i in ids]
    write_jsonl(out_path, records)
    finalize_ckpt(ckpt_path)
    print(f"\nSaved {len(records)} records -> {out_path}")

    all_tokens = [s["num_tokens"] for r in records for s in r["samples"]]
    all_trunc = sum(1 for r in records for s in r["samples"] if s["truncated"])
    n_total = len(all_tokens)
    update_diagnostics(config.DIAGNOSTICS_PATH, diag_section, {
        **config.sampling_config(),
        "n_items": len(records),
        "samples_per_item": n_samples,
        "total_samples": n_total,
        "truncated": all_trunc,
        "truncated_pct": round(100.0 * all_trunc / max(n_total, 1), 3),
        "response_tokens_mean": round(sum(all_tokens) / max(n_total, 1), 1),
        "response_tokens_min": min(all_tokens) if all_tokens else None,
        "response_tokens_max": max(all_tokens) if all_tokens else None,
        "newly_generated_samples": total_samples,
    })
    print(f"Diagnostics -> {config.DIAGNOSTICS_PATH} [{diag_section}]")
    return records
