"""
Inference pipeline for the MVP CoT monitorability evaluation — Qwen3 variant.

Qwen3 is a hybrid *reasoning* model. This pipeline always runs it in thinking
mode, so it emits its chain-of-thought inside a `<think>...</think>` block before
the visible answer. This is structurally different from the Llama/Tulu checkpoints
handled by `src/inference.py`, so this module exists as a parallel implementation
rather than modifying the original.

Differences vs. `src/inference.py`:
  1. `format_prompt_qwen` enables thinking on `apply_chat_template`
     (ChatML template, `<|im_start|>/<|im_end|>`).
  2. The chain-of-thought is the content of the `<think>` block; `extract_cot`
     returns that block's text (tags stripped) and the full block is also stored
     in a new `thinking` field.
  3. The final answer is parsed preferentially from the post-`</think>` text so a
     stray "the answer is (X)" inside the reasoning never wins over the model's
     actual conclusion.
  4. Sampling uses the Qwen3-recommended thinking-mode settings (temp 0.6,
     top_p 0.95, top_k 20, min_p 0) and a much larger `max_new_tokens`, all read
     from the `qwen_inference:` block of the config.

The JSONL schema is a superset of `src/inference.py`'s — every field the
Llama pipeline writes is written here too (so scripts 03–06 work unchanged),
plus one extra `thinking` field. `raw_output` is the verbatim model output,
INCLUDING the `<think>...</think>` tags, so the monitor (03_run_monitor.py)
reads the full reasoning exactly as the model produced it.
"""

import gzip
import json
import shutil
import sys
from pathlib import Path

# Reuse the model-agnostic prompt construction and the OLMES answer parser /
# anchor-based CoT splitter from src/inference.py (shared helpers; no heavy
# vLLM/torch imports at module level, so importing is cheap and safe).
from src.inference import (
    _SYSTEM_PROMPT,
    build_user_content,
    extract_cot as _base_extract_cot,
    parse_answer as _base_parse_answer,
)


# ---------------------------------------------------------------------------
# Prompt construction (Qwen3 thinking mode)
# ---------------------------------------------------------------------------

def format_prompt_qwen(item: dict, include_hint: bool, tokenizer,
                       hint_type: str = "professor", hint_cfg: dict | None = None,
                       dataset: list | None = None) -> str:
    """Apply Qwen3's chat template (thinking mode) and return the prompt string.

    The user/system content is byte-for-byte identical to the Llama pipeline
    (`build_user_content`) so the only cross-model difference is the chat
    template itself. `enable_thinking=True` lets the model emit `<think>...</think>`.

    `hint_type`/`hint_cfg`/`dataset` are forwarded to `build_user_content` to
    select the hinted-arm hint (see src/hints.py); `dataset` is only used by the
    `wrong_few_shot` hint to sample example questions.
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": build_user_content(
            item, include_hint, hint_type=hint_type, hint_cfg=hint_cfg, dataset=dataset)},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


# ---------------------------------------------------------------------------
# Thinking-block handling
# ---------------------------------------------------------------------------

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def split_thinking(raw_output: str) -> tuple[str, str]:
    """Split a Qwen3 completion into (thinking_text, post_think_text).

    - `thinking_text`: content of the `<think>...</think>` block, tags stripped.
    - `post_think_text`: everything after `</think>` — the visible answer.

    Robust to the common shapes:
      * `<think> ... </think>\n\nanswer`      → normal thinking output
      * `... </think>\n\nanswer`               → some vLLM/template versions emit the
                                                 opening `<think>` as part of the
                                                 prompt, so the output starts inside
                                                 the block
      * `<think> ...` (no close)               → truncated mid-thought: all thinking
      * no `<think>` at all                    → non-thinking output: no thinking
    """
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


def extract_cot_qwen(raw_output: str) -> str:
    """Return the chain-of-thought for a Qwen3 completion.

    The CoT is the `<think>` block content. If there is no thinking block
    (non-thinking output), fall back to the anchor-based splitter from the
    Llama pipeline over the visible text.
    """
    thinking, post = split_thinking(raw_output)
    if thinking:
        return thinking
    return _base_extract_cot(post or raw_output)


def parse_answer_qwen(raw_output: str) -> str | None:
    """Parse the final answer letter, preferring the post-`</think>` answer.

    Qwen3 often rehearses candidate answers inside `<think>`; the model's true
    conclusion is the text after `</think>`. Parse that first; only fall back to
    the whole output (e.g. when the block never closed) if it yields nothing.
    """
    _, post = split_thinking(raw_output)
    if post:
        ans = _base_parse_answer(post)
        if ans is not None:
            return ans
    return _base_parse_answer(raw_output)


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------

def load_qwen_model(model_cfg: dict, max_model_len: int = 24576, gpu_memory_utilization: float = 0.95):
    """Load tokenizer and vLLM engine for a Qwen3 model. Call once; reuse across datasets.

    max_model_len: hard cap on prompt_tokens + output_tokens per sequence. Set to
        24576 = 16384 (max_new_tokens) + 8192 prompt budget. MCQ prompts are typically
        1–3k tokens so this gives ample headroom with zero truncation risk, while raising
        KV-cache concurrency from ~8x (native 40960 limit) to ~14x on a 48GB GPU.
        Do NOT reduce below 16384 + observed_max_prompt_len or outputs will be silently
        truncated (finish_reason="length"), which would be a confound.
    gpu_memory_utilization: fraction of VRAM given to vLLM. 0.95 is safe for a dedicated
        inference pod with no other GPU workloads; the headroom above the weights goes
        to KV cache, so this directly raises how many sequences decode concurrently.

    Prefix caching is enabled: every prompt shares the identical system-prompt prefix
    (and, within an instance, the hinted/unhinted prompts share a long common prefix),
    so vLLM reuses the cached KV for those shared tokens. This is a pure throughput win
    with no effect on the generated tokens — sampling still starts at the first
    non-shared position with each request's own seed/params.
    """
    from vllm import LLM
    from transformers import AutoTokenizer

    hf_id = model_cfg["hf_id"]
    print(f"Loading tokenizer: {hf_id}")
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    print(f"Loading model via vLLM: {hf_id} "
          f"(max_model_len={max_model_len}, gpu_memory_utilization={gpu_memory_utilization})")
    llm = LLM(
        model=hf_id,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=True,
    )
    return tokenizer, llm


def unhinted_arm_complete(path: Path, dataset: list, n_samples: int) -> bool:
    """True iff `path` (an unhinted.jsonl, optionally .jsonl.gz — large runs are
    gzipped in git for the GitHub 100MB limit) covers every (instance, sample)
    pair of `dataset` at this N. Used to decide whether the unhinted arm can be
    reused instead of regenerated — valid because the unhinted prompt is
    byte-identical for every hint type (see src/inference.build_user_content)
    and unhinted seeds don't depend on the hint type either."""
    if not path.exists():
        return False
    opener = gzip.open if path.suffix == ".gz" else open
    seen = set()
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                return False
            seen.add((r["instance_id"], r["sample_idx"]))
    need = {(item["instance_id"], s) for item in dataset for s in range(n_samples)}
    return need <= seen


def run_inference_qwen(config: dict, model_cfg: dict, dataset: list, base_seed: int = 88,
                       tokenizer=None, llm=None, hint_type: str = "professor",
                       shared_unhinted: Path | None = None):
    """
    Run hinted and unhinted inference for one Qwen3 model over the full dataset.

    Reads sampling parameters from `config["qwen_inference"]` (NOT the shared
    `inference:` block, which is for the Llama/Tulu pipeline).

    `tokenizer` and `llm` may be pre-loaded (e.g. when looping over multiple
    datasets) to avoid reloading weights on every call. If None, they are loaded
    here (single-dataset / backward-compatible path).

    `hint_type` selects the hinted-arm hint (see src/hints.HINT_TYPES); the
    wording is read from the config `hint:` block. Outputs are namespaced by hint
    type so different hints never collide.

    Writes hinted.jsonl, unhinted.jsonl, and progress.json (resume checkpoint)
    directly into `config["paths"]["outputs_dir"]`. The caller is responsible for
    passing an already model/dataset/hint-namespaced directory (via resolve_paths),
    i.e. outputs/{model_name}/{dataset}/{hint_type}/.

    Each JSONL line contains the same fields as src/inference.py plus `thinking`
    and `hint_type`:
      instance_id, question_id, arm, sample_idx, hint_target, correct_answer,
      hint_type, full_prompt, raw_output, thinking, cot, parsed_answer,
      followed_hint, token_count, truncated, seed
    """
    from vllm import SamplingParams

    model_name = model_cfg["name"]
    inf = config["qwen_inference"]
    hint_cfg = config.get("hint", {})
    n_samples = inf["n_samples"]
    checkpoint_every = inf["checkpoint_every"]

    if tokenizer is None or llm is None:
        tokenizer, llm = load_qwen_model(model_cfg)

    # outputs_dir is already namespaced model/dataset/hint by the caller
    # (resolve_paths); write straight into it — do NOT re-append model/hint.
    out_dir = Path(config["paths"]["outputs_dir"])
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

    # Unhinted-arm reuse: the unhinted prompt and seeds are hint-type-invariant,
    # so a complete unhinted.jsonl from another hint's run is the same data.
    # Copy it in (only onto a fresh arm — never over partial own generations)
    # and skip unhinted generation entirely while it covers the whole dataset.
    # A .jsonl.gz source (gzipped in git for the 100MB limit) is decompressed.
    if shared_unhinted is not None and not unhinted_path.exists():
        if shared_unhinted.suffix == ".gz":
            with gzip.open(shared_unhinted, "rb") as src, open(unhinted_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
        else:
            shutil.copyfile(shared_unhinted, unhinted_path)
        print(f"Reusing unhinted arm (hint-type-invariant) from: {shared_unhinted}")
    skip_unhinted = unhinted_arm_complete(unhinted_path, dataset, n_samples)
    if skip_unhinted:
        print("Unhinted arm already complete — generating the hinted arm only.")

    todo = [item for item in dataset if item["instance_id"] not in completed]
    print(f"{len(todo)} instances to process for {model_name} / {out_dir.name} "
          f"(thinking mode).")

    null_count = 0
    truncated_count = 0
    total_generated = 0

    # Seed scheme: each of the N samples in an arm uses a distinct, fixed seed,
    # reused across all instances (different prompts + same seed still yield
    # independent outputs). The two arms use DISJOINT seed sets so hinted and
    # unhinted stay independent (the TE = p1 - p0 analysis assumes Cov(p0,p1)=0):
    # the hinted arm uses powers of 2 and the unhinted arm powers of 3, which can
    # never coincide (sample s -> 2**(s+1) / 3**(s+1)). This scales to ANY N (no
    # fixed cap) and reproduces the original [2,4,8,16] / [3,9,27,81] for N<=4.
    _ARM_SEEDS = {
        "hinted":   [2 ** (s + 1) for s in range(n_samples)],
        "unhinted": [3 ** (s + 1) for s in range(n_samples)],
    }

    def _seed(item, s, arm):
        return _ARM_SEEDS[arm][s]

    def _sampling_params(item, s, arm):
        return SamplingParams(
            n=1,
            temperature=inf["temperature"],
            top_p=inf["top_p"],
            top_k=inf.get("top_k", -1),
            max_tokens=inf["max_new_tokens"],
            seed=_seed(item, s, arm),
        )

    for batch_start in range(0, len(todo), checkpoint_every):
        batch = todo[batch_start : batch_start + checkpoint_every]

        hinted_prompts = [
            format_prompt_qwen(item, include_hint=True, tokenizer=tokenizer,
                               hint_type=hint_type, hint_cfg=hint_cfg, dataset=dataset)
            for item in batch
        ]
        unhinted_prompts = [
            format_prompt_qwen(item, include_hint=False, tokenizer=tokenizer,
                               hint_type=hint_type, hint_cfg=hint_cfg, dataset=dataset)
            for item in batch
        ]

        # Flatten every (sample, instance, arm) into ONE request list and issue a
        # single llm.generate call for the whole checkpoint. vLLM's continuous
        # batching then schedules all of them together and we pay one decode
        # drain-tail per checkpoint instead of one per (sample × arm) sub-call
        # (previously 2*n_samples generate calls, each leaving the GPU idle while
        # its few longest <think> traces finished). This is purely a scheduling
        # change: each request keeps its own prompt, per-arm/sample seed, and
        # sampling params, so the sampling distribution is identical — nothing
        # about the generations or token budget changes.
        arms_per_instance = 1 if skip_unhinted else 2
        reqs = []          # parallel to prompts/sampling: (item, sample_idx, arm_label, prompt)
        prompts = []
        sampling = []
        for s in range(n_samples):
            for idx, item in enumerate(batch):
                arm_pairs = [("hinted", hinted_prompts[idx])]
                if not skip_unhinted:
                    arm_pairs.append(("unhinted", unhinted_prompts[idx]))
                for arm_label, arm_prompt in arm_pairs:
                    reqs.append((item, s, arm_label, arm_prompt))
                    prompts.append(arm_prompt)
                    sampling.append(_sampling_params(item, s, arm_label))

        print(
            f"  Checkpoint [{batch_start}:{batch_start+len(batch)}] — generating "
            f"{len(prompts)} requests ({len(batch)} instances × {n_samples} samples × "
            f"{arms_per_instance} arm{'s' if arms_per_instance > 1 else ' (unhinted reused)'}) "
            f"in one batch ..."
        )
        outputs = llm.generate(prompts, sampling)

        with open(hinted_path, "a", encoding="utf-8") as hf, \
             open(unhinted_path, "a", encoding="utf-8") as uf:

            for (item, s, arm_label, arm_prompt), arm_out in zip(reqs, outputs):
                arm_file = hf if arm_label == "hinted" else uf
                completion = arm_out.outputs[0]  # n=1
                raw = completion.text
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
                    "hint_type":     hint_type,
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
    null_rate = null_count / max(hinted_total * (1 if skip_unhinted else 2), 1)
    if null_rate > 0.05:
        print(
            f"WARNING: answer parse failure rate {null_rate:.1%} > 5% threshold. "
            "If high, increase qwen_inference.max_new_tokens (truncated thinking) "
            "or check the answer-format instruction in the prompt.",
            file=sys.stderr,
        )

    print(
        f"Inference complete for {model_name}. "
        f"Total outputs: {total_generated}. "
        f"Parse failures: {null_count} ({null_rate:.1%}). "
        f"Truncated: {truncated_count}."
    )
