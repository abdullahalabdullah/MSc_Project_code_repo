"""Stage 03 — Split CoT/answer, classify with Qwen3Guard-Gen-4B, apply filter-1.

For each response: strip everything up to and including </think>; the remainder is the
`answer`. Responses truncated mid-think (no </think>) have no visible answer and are
marked invalid.

Qwen3Guard-Gen labels each valid (prompt, answer) as Safe / Unsafe / Controversial.
Controversial -> Unsafe (conservative).

filter-1: drop a prompt if all its VALID responses share one safety label (or if it has
fewer than 2 valid responses) — such prompts cannot yield a chosen/rejected pair.

Runs on GPU. Output: config.LABELED_PATH (surviving prompts only).
"""
import re

import joblib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
import env_setup
from utils import set_seed, extract_visible_response

SAFETY_RE = re.compile(r"Safety:\s*(Safe|Unsafe|Controversial)", re.IGNORECASE)


def parse_label(text):
    """Return 'Safe' or 'Unsafe' from a Qwen3Guard generation ('Controversial' -> 'Unsafe')."""
    m = SAFETY_RE.search(text)
    if not m:
        return None
    raw = m.group(1).capitalize()
    return "Safe" if raw == "Safe" else "Unsafe"


def classify_batch(model, tokenizer, pairs):
    """pairs: list of (prompt, answer). Returns list of raw guard generations (str)."""
    rendered = []
    for prompt, answer in pairs:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
        # Per the Qwen3Guard-Gen model card: apply the template to the [user, assistant]
        # conversation; the guard's own template appends the moderation cue, and the model
        # then generates "Safety: ...\nCategories: ...". (If a smoke run shows all labels
        # unparsed/None, flip add_generation_prompt to True.)
        rendered.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        )

    enc = tokenizer(rendered, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=config.GUARD_MAX_NEW_TOKENS, do_sample=False)
    # Strip the prompt tokens from each row (left-padded, so slice by input length).
    new_tokens = gen[:, enc["input_ids"].shape[1]:]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


def main():
    env_setup.setup_env()
    config.ensure_output_dir()
    set_seed(config.SEED)

    all_responses = joblib.load(config.RESPONSES_PATH)
    n_prompts = len(all_responses)
    print(f"Loaded {n_prompts} prompts x {len(all_responses[0])} responses")

    # 1) Split CoT/answer, mark invalid (truncated mid-think).
    n_invalid = 0
    for per_prompt in all_responses:
        for rd in per_prompt:
            visible = extract_visible_response(rd["full_response"])
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

    # 2) Flatten all valid (prompt, answer) pairs and classify in batches.
    flat = [(pi, ri) for pi, pp in enumerate(all_responses)
            for ri, rd in enumerate(pp) if not rd["invalid"]]
    print(f"Classifying {len(flat)} valid (prompt, answer) pairs with {config.GUARD_MODEL} ...")

    tokenizer = AutoTokenizer.from_pretrained(config.GUARD_MODEL)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.GUARD_MODEL, torch_dtype="auto", device_map="auto"
    ).eval()

    for b in range(0, len(flat), config.GUARD_BATCH):
        chunk = flat[b:b + config.GUARD_BATCH]
        pairs = [(all_responses[pi][ri]["prompt"], all_responses[pi][ri]["answer"])
                 for pi, ri in chunk]
        raw_outs = classify_batch(model, tokenizer, pairs)
        for (pi, ri), raw in zip(chunk, raw_outs):
            label = parse_label(raw)
            all_responses[pi][ri]["guard_raw"] = raw
            all_responses[pi][ri]["guard_label"] = label  # 'Safe' | 'Unsafe' | None (unparsed)
        if (b // config.GUARD_BATCH) % 20 == 0:
            print(f"  {min(b + config.GUARD_BATCH, len(flat))}/{len(flat)}")

    # Label distribution across all classified pairs — diagnostic that confirms the guard
    # is emitting parseable "Safety:" verdicts rather than silently returning None. If every
    # output is None, the Qwen3Guard prompt/template is wrong (100% filter-1 drop would then
    # be an artefact, not real label agreement).
    from collections import Counter
    dist = Counter(all_responses[pi][ri]["guard_label"] for pi, ri in flat)
    n_safe, n_unsafe, n_none = dist.get("Safe", 0), dist.get("Unsafe", 0), dist.get(None, 0)
    print(f"\nGuard label distribution: Safe={n_safe} Unsafe={n_unsafe} unparsed(None)={n_none}")
    if flat and n_none == len(flat):
        print("  WARNING: EVERY guard output was unparsed — the Qwen3Guard template is likely "
              "wrong. Try flipping add_generation_prompt=True in classify_batch(), and eyeball "
              "a guard_raw string in 03_all_labeled.joblib.")

    # Persist ALL prompts+labels (not just survivors) so filter-1 decisions and raw guard
    # outputs stay inspectable even when 0 prompts survive.
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


if __name__ == "__main__":
    main()
