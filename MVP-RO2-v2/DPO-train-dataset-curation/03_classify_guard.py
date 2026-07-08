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

Runs on GPU. Output: config.LABELED_PATH (survivors) + config.ALL_LABELED_PATH (all, debug).
"""
import re

import joblib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
import env_setup
from utils import (
    set_seed, extract_visible_response, think_answer_word_counts, update_diagnostics,
)

SAFETY_RE = re.compile(r"Safety:\s*(Safe|Unsafe|Controversial)", re.IGNORECASE)
REFUSAL_RE = re.compile(r"Refusal:\s*(yes|no)", re.IGNORECASE)


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


def classify_batch(model, tokenizer, rendered, max_new_tokens):
    """rendered: list of already chat-templated strings (Qwen3Guard's chat template appends
    the right assessment cue based on whether the last turn is user-only or user+assistant,
    so no add_generation_prompt is needed). Returns list of raw Qwen3Guard generations (str)."""
    enc = tokenizer(
        rendered, return_tensors="pt", padding=True, truncation=True, max_length=8192,
    ).to(model.device)
    with torch.no_grad():
        gen = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = gen[:, enc["input_ids"].shape[1]:]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


def pack_length_sorted_batches(rendered, max_batch, token_budget, max_new_tokens):
    """Batch plan for left-padded batched generation: returns index lists into `rendered`,
    length-sorted and packed so batch_size x (longest padded length in the batch) stays
    under token_budget.

    Left padding makes every sequence in a batch pay prefill compute + per-step KV reads for
    the batch's LONGEST input, and generate() only early-stops once the WHOLE batch is done —
    so mixed-length batches spend most of their work on pad tokens. Sorting by length gives
    near-homogeneous batches (many short inputs per call, the rare long ones in small calls),
    and the token budget bounds worst-case KV memory by construction.

    Lengths are estimated conservatively from characters (~3 chars/token, i.e. overestimating
    typical English tokenization), capped at the 8192-token truncation limit, plus the
    generation allowance. Throughput-only: decoding is greedy, so an item's output does not
    depend on which batch it lands in; callers write results back by original index.
    """
    def est_tokens(s):
        return min(len(s) // 3 + 16, 8192) + max_new_tokens

    order = sorted(range(len(rendered)), key=lambda i: est_tokens(rendered[i]))
    batches, cur = [], []
    for i in order:
        # Ascending order -> item i would be the batch's longest; padded cost if added:
        if cur and (len(cur) == max_batch
                    or (len(cur) + 1) * est_tokens(rendered[i]) > token_budget):
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    return batches


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

    # 2) Flatten all valid (prompt, answer) pairs and classify with Qwen3Guard.
    flat = [(pi, ri) for pi, pp in enumerate(all_responses)
            for ri, rd in enumerate(pp) if not rd["invalid"]]
    print(f"Classifying {len(flat)} valid (prompt, answer) pairs with {config.GUARD_MODEL} ...")

    tokenizer = AutoTokenizer.from_pretrained(config.GUARD_MODEL)
    tokenizer.padding_side = "left"      # batched generation
    tokenizer.truncation_side = "left"   # if over-long, keep the answer + assessment scaffold
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.GUARD_MODEL, torch_dtype="auto", device_map="auto"
    ).eval()

    rendered = [
        tokenizer.apply_chat_template(
            [
                {"role": "user", "content": all_responses[pi][ri]["prompt"]},
                {"role": "assistant", "content": all_responses[pi][ri]["answer"]},
            ],
            tokenize=False,
        )
        for pi, ri in flat
    ]
    batches = pack_length_sorted_batches(
        rendered, config.GUARD_MAX_BATCH, config.GUARD_TOKEN_BUDGET, config.GUARD_MAX_NEW_TOKENS)
    print(f"  packed into {len(batches)} length-sorted batches "
          f"(budget={config.GUARD_TOKEN_BUDGET} padded tokens, max batch={config.GUARD_MAX_BATCH})")
    n_done = 0
    for bi, batch_ix in enumerate(batches):
        raw_outs = classify_batch(
            model, tokenizer, [rendered[i] for i in batch_ix], config.GUARD_MAX_NEW_TOKENS)
        for i, raw in zip(batch_ix, raw_outs):
            pi, ri = flat[i]
            label, refusal = parse_qwen3guard_response(raw)
            rd = all_responses[pi][ri]
            rd["guard_raw"] = raw
            rd["guard_label"] = label            # 'Safe' | 'Unsafe' | None (unparsed); Controversial folded into Unsafe
            rd["refusal"] = refusal              # bool | None  (for the future benign axis)
        n_done += len(batch_ix)
        if bi % 20 == 0:
            print(f"  {n_done}/{len(flat)}")
    del rendered

    # 2b) Request-only harmfulness ("harmful_request"): a separate prompt-only generation,
    # since Qwen3Guard needs a distinct pass for this (no single-call 3-verdict mode like
    # WildGuard had). Run once per UNIQUE prompt — all N responses to a prompt share the same
    # request, so this is n_prompts calls, not n_prompts * N.
    unique_prompts = [per_prompt[0]["prompt"] for per_prompt in all_responses]
    print(f"Classifying {len(unique_prompts)} unique prompts for request harmfulness ...")
    prompt_harmful = [None] * n_prompts
    rendered_prompts = [
        tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False)
        for p in unique_prompts
    ]
    for batch_ix in pack_length_sorted_batches(
            rendered_prompts, config.GUARD_MAX_BATCH, config.GUARD_TOKEN_BUDGET,
            config.GUARD_MAX_NEW_TOKENS):
        raw_outs = classify_batch(
            model, tokenizer, [rendered_prompts[i] for i in batch_ix], config.GUARD_MAX_NEW_TOKENS)
        for i, raw in zip(batch_ix, raw_outs):
            prompt_harmful[i] = parse_qwen3guard_prompt(raw)
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


if __name__ == "__main__":
    main()
