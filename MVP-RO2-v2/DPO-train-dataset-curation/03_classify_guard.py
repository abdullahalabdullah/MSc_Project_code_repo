"""Stage 03 — Split CoT/answer, classify with Qwen3Guard, apply filter-1.

Labeler: **Qwen/Qwen3Guard-Gen-4B**, a chat model that frames safety classification as an
instruction-following task (see model card). Given a two-turn conversation (user prompt +
assistant response) it generates:
    Safety: Safe|Unsafe|Controversial
    Categories: ...
    Refusal: Yes|No
Qwen3Guard's Safety verdict is 3-way; we COLLAPSE Controversial -> Unsafe so `guard_label`
stays binary Safe/Unsafe (what filter-1/filter-3 need). "Refusal" is stored as `refusal` (for
the future benign axis). Request-only harmfulness ("Harmful request" in the old WildGuard
sense) needs a *separate* prompt-only generation (Qwen3Guard has no single-pass three-verdict
mode); we run that once per UNIQUE prompt (not once per response) and store it as
`harmful_request`.

For each response: strip everything up to and including </think>; the remainder is the
`answer`. Responses truncated mid-think (no </think>) have no visible answer -> invalid.

filter-1: drop a prompt if all its VALID responses share one label (or if <2 valid) — such
prompts cannot yield a chosen/rejected pair.

Inference runs on **vLLM** (like stage 02): greedy decoding, continuous batching, per-sequence
EOS stop. Inputs are chat-templated to token ids and left-truncated to GUARD_MAX_INPUT_TOKENS
(keeping the answer + assessment cue). Both classification passes are **checkpointed** every
GUARD_CHECKPOINT_EVERY pairs to <OUTPUT_DIR>/03_guard_*.ckpt.jsonl, so an interrupted run
resumes from where it stopped instead of re-classifying from scratch. A fingerprint of the
inputs invalidates a stale checkpoint if 02_responses.joblib changed.

Runs on GPU. Output: config.LABELED_PATH (survivors) + config.ALL_LABELED_PATH (all, debug).
"""
import hashlib
import json
import os
import re

import joblib
from vllm import LLM, SamplingParams

import config
import env_setup
from utils import (
    set_seed, extract_visible_response, think_answer_word_counts, update_diagnostics,
)

SAFETY_RE = re.compile(r"Safety:\s*(Safe|Unsafe|Controversial)", re.IGNORECASE)
REFUSAL_RE = re.compile(r"Refusal:\s*(yes|no)", re.IGNORECASE)

RESP_CKPT_PATH = os.path.join(config.OUTPUT_DIR, "03_guard_responses.ckpt.jsonl")
PROMPT_CKPT_PATH = os.path.join(config.OUTPUT_DIR, "03_guard_prompts.ckpt.jsonl")
CKPT_META_PATH = os.path.join(config.OUTPUT_DIR, "03_guard.ckpt.meta.json")


def _yn(m):
    return None if m is None else (m.group(1).lower() == "yes")


def _binary_safety(m):
    """Safe -> 'Safe'; Unsafe or Controversial -> 'Unsafe'; unparsed -> None."""
    if m is None:
        return None
    return "Safe" if m.group(1).lower() == "safe" else "Unsafe"


def parse_qwen3guard_response(text):
    """Return (guard_label, refusal) from a two-turn (prompt+response) Qwen3Guard generation."""
    guard_label = _binary_safety(SAFETY_RE.search(text))
    refusal = _yn(REFUSAL_RE.search(text))
    return guard_label, refusal


def parse_qwen3guard_prompt(text):
    """Return harmful_request (bool | None) from a prompt-only Qwen3Guard generation."""
    label = _binary_safety(SAFETY_RE.search(text))
    return None if label is None else (label == "Unsafe")


def render_ids(tokenizer, messages):
    """Chat-template `messages` to token ids and left-truncate to GUARD_MAX_INPUT_TOKENS.

    Qwen3Guard's chat template appends the right assessment cue based on whether the last turn
    is user-only or user+assistant, so no add_generation_prompt is needed. Left truncation
    (keep the tail) preserves the answer + assessment scaffold if the input is over-long.
    """
    ids = tokenizer.apply_chat_template(messages, tokenize=True)
    if len(ids) > config.GUARD_MAX_INPUT_TOKENS:
        ids = ids[-config.GUARD_MAX_INPUT_TOKENS:]
    return ids


# ── Checkpointing ──────────────────────────────────────────────────────────────
# Each pass writes one JSONL line per completed item (append-only, fsync'd per shard); a
# half-written trailing line on a crash is simply ignored on reload. A sidecar meta file
# fingerprints the inputs so a checkpoint left over from a *different* 02_responses.joblib is
# discarded rather than silently reused.

def _fingerprint(flat, all_responses):
    n = len(flat)
    probe = []
    for i in (0, n // 2, n - 1):
        if 0 <= i < n:
            pi, ri = flat[i]
            rd = all_responses[pi][ri]
            probe.append(rd["prompt"])
            probe.append(rd["answer"])
    h = hashlib.sha256(("\x00".join(probe)).encode("utf-8")).hexdigest()
    return {"n_prompts": len(all_responses), "n_flat": n, "sig": h}


def _reset_checkpoints_if_stale(fp):
    """Keep checkpoints only if they match the current inputs; otherwise archive + start fresh."""
    if os.path.exists(CKPT_META_PATH):
        with open(CKPT_META_PATH) as f:
            prev = json.load(f)
        if prev == fp:
            return  # valid resume
        print("  WARNING: existing 03 checkpoint does not match current 02_responses.joblib "
              "(inputs changed) — archiving stale checkpoints and starting fresh.")
        for p in (RESP_CKPT_PATH, PROMPT_CKPT_PATH, CKPT_META_PATH):
            if os.path.exists(p):
                os.replace(p, p + ".stale")
    with open(CKPT_META_PATH, "w") as f:
        json.dump(fp, f)


def _load_ckpt(path):
    """Return {index: record} for all complete JSONL lines (ignoring a torn trailing line)."""
    done = {}
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                break  # torn last line from a mid-write crash
            done[rec["i"]] = rec
    return done


def _run_pass(llm, sampling_params, indices, make_ids, ckpt_path, on_result, total, label):
    """Generate for the not-yet-done `indices` in checkpointed shards.

    make_ids(i) -> token ids for item i; on_result(i, raw_text) applies the parsed result.
    Already-checkpointed items are replayed through on_result (no recompute) and skipped.
    """
    done = _load_ckpt(ckpt_path)
    for i, rec in done.items():
        on_result(i, rec["raw"])
    remaining = [i for i in indices if i not in done]
    n_done = len(done)
    print(f"  {label}: {n_done}/{total} already checkpointed, {len(remaining)} to classify")

    step = config.GUARD_CHECKPOINT_EVERY
    for s in range(0, len(remaining), step):
        shard = remaining[s:s + step]
        prompts = [{"prompt_token_ids": make_ids(i)} for i in shard]
        outs = llm.generate(prompts, sampling_params, use_tqdm=False)
        with open(ckpt_path, "a") as f:
            for i, out in zip(shard, outs):
                raw = out.outputs[0].text
                on_result(i, raw)
                f.write(json.dumps({"i": i, "raw": raw}) + "\n")
            f.flush()
            os.fsync(f.fileno())
        n_done += len(shard)
        print(f"  {label}: {n_done}/{total}")


def main():
    env_setup.setup_env()
    config.ensure_output_dir()
    set_seed(config.SEED)

    all_responses = joblib.load(config.RESPONSES_PATH)
    n_prompts = len(all_responses)
    print(f"Loaded {n_prompts} prompts x {len(all_responses[0])} responses")

    # 1) Split CoT/answer, mark invalid (truncated mid-think), record think/answer lengths.
    n_invalid = 0
    for per_prompt in all_responses:
        for rd in per_prompt:
            visible = extract_visible_response(rd["full_response"])
            think_w, answer_w, has_think = think_answer_word_counts(rd["full_response"])
            rd["think_words"] = think_w
            rd["answer_words"] = answer_w
            rd["has_think"] = has_think
            if visible is None or visible == "":
                rd["answer"] = ""
                rd["invalid"] = True
                n_invalid += 1
            else:
                rd["answer"] = visible
                rd["invalid"] = False
    n_total = n_prompts * len(all_responses[0])
    print(f"Invalid (no visible answer): {n_invalid} / {n_total} "
          f"({100.0 * n_invalid / n_total:.1f}%)")

    # 2) Flatten all valid (prompt, answer) pairs and classify with Qwen3Guard on vLLM.
    flat = [(pi, ri) for pi, pp in enumerate(all_responses)
            for ri, rd in enumerate(pp) if not rd["invalid"]]
    print(f"Classifying {len(flat)} valid (prompt, answer) pairs with {config.GUARD_MODEL} (vLLM) ...")

    _reset_checkpoints_if_stale(_fingerprint(flat, all_responses))

    sampling_params = SamplingParams(
        temperature=0.0,                              # greedy: labels are deterministic
        max_tokens=config.GUARD_MAX_NEW_TOKENS,
    )
    llm = LLM(
        model=config.GUARD_MODEL,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=config.GUARD_MAX_INPUT_TOKENS + config.GUARD_MAX_NEW_TOKENS,
        gpu_memory_utilization=config.GPU_MEMORY_UTILIZATION,
        max_num_seqs=config.GUARD_MAX_NUM_SEQS,
        enable_prefix_caching=True,                   # the N samples of a prompt share its user-turn prefix
    )
    tokenizer = llm.get_tokenizer()

    def resp_ids(i):
        pi, ri = flat[i]
        rd = all_responses[pi][ri]
        return render_ids(tokenizer, [
            {"role": "user", "content": rd["prompt"]},
            {"role": "assistant", "content": rd["answer"]},
        ])

    def apply_resp(i, raw):
        pi, ri = flat[i]
        label, refusal = parse_qwen3guard_response(raw)
        rd = all_responses[pi][ri]
        rd["guard_raw"] = raw
        rd["guard_label"] = label   # 'Safe' | 'Unsafe' | None (unparsed); Controversial folded into Unsafe
        rd["refusal"] = refusal      # bool | None  (for the future benign axis)

    _run_pass(llm, sampling_params, range(len(flat)), resp_ids, RESP_CKPT_PATH,
              apply_resp, len(flat), "responses")

    # 2b) Request-only harmfulness ("harmful_request"): a separate prompt-only generation,
    # since Qwen3Guard needs a distinct pass for this (no single-call 3-verdict mode like
    # WildGuard had). Run once per UNIQUE prompt — all N responses to a prompt share the same
    # request, so this is n_prompts calls, not n_prompts * N.
    unique_prompts = [per_prompt[0]["prompt"] for per_prompt in all_responses]
    print(f"Classifying {len(unique_prompts)} unique prompts for request harmfulness (vLLM) ...")
    prompt_harmful = [None] * n_prompts

    def prompt_ids(i):
        return render_ids(tokenizer, [{"role": "user", "content": unique_prompts[i]}])

    def apply_prompt(i, raw):
        prompt_harmful[i] = parse_qwen3guard_prompt(raw)

    _run_pass(llm, sampling_params, range(n_prompts), prompt_ids, PROMPT_CKPT_PATH,
              apply_prompt, n_prompts, "prompts")

    for pi, per_prompt in enumerate(all_responses):
        for rd in per_prompt:
            rd["harmful_request"] = prompt_harmful[pi]   # bool | None

    # Label distribution — confirms Qwen3Guard is emitting parseable verdicts, not None.
    from collections import Counter
    dist = Counter(all_responses[pi][ri]["guard_label"] for pi, ri in flat)
    n_safe, n_unsafe, n_none = dist.get("Safe", 0), dist.get("Unsafe", 0), dist.get(None, 0)
    print(f"\nQwen3Guard label distribution: Safe={n_safe} Unsafe={n_unsafe} unparsed(None)={n_none}")
    if flat and n_none == len(flat):
        print("  WARNING: EVERY Qwen3Guard output was unparsed — check the chat-template "
              "rendering / SAFETY_RE parsing, and eyeball a guard_raw string in 03_all_labeled.joblib.")

    joblib.dump(all_responses, config.ALL_LABELED_PATH, compress=True)
    print(f"Saved (debug, all prompts) -> {config.ALL_LABELED_PATH}")

    # 3) filter-1: keep a prompt only if its valid responses show BOTH labels.
    survivors = []
    n_dropped = 0
    for pi, per_prompt in enumerate(all_responses):
        labels = [rd["guard_label"] for rd in per_prompt
                  if not rd["invalid"] and rd["guard_label"] is not None]
        if len(labels) < 2 or len(set(labels)) < 2:
            n_dropped += 1
            continue
        survivors.append(per_prompt)

    pct_dropped = 100.0 * n_dropped / n_prompts
    print(f"\n>> filter-1: dropped {n_dropped} / {n_prompts} prompts "
          f"({pct_dropped:.1f}%) — all valid responses shared one label or <2 valid")
    print(f">> {len(survivors)} prompts survive filter-1")

    joblib.dump(survivors, config.LABELED_PATH, compress=True)
    print(f"Saved -> {config.LABELED_PATH}")

    update_diagnostics(config.DIAGNOSTICS_PATH, "classification", {
        "labeler": config.GUARD_MODEL,
        "n_prompts": n_prompts,
        "responses_per_prompt": len(all_responses[0]),
        "invalid_no_answer": n_invalid,
        "invalid_pct": round(100.0 * n_invalid / max(n_total, 1), 3),
        "label_distribution": {"Safe": n_safe, "Unsafe": n_unsafe, "unparsed": n_none},
        "filter1_dropped": n_dropped,
        "filter1_dropped_pct": round(pct_dropped, 3),
        "filter1_survivors": len(survivors),
    })
    print(f"Diagnostics -> {config.DIAGNOSTICS_PATH} [classification]")

    # Both passes finished and outputs are dumped — drop the resume checkpoints.
    for p in (RESP_CKPT_PATH, PROMPT_CKPT_PATH, CKPT_META_PATH):
        if os.path.exists(p):
            os.remove(p)


if __name__ == "__main__":
    main()
