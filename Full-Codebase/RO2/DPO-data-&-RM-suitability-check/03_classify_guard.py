"""Stage 03 — Split CoT/answer, classify with Qwen3Guard-Gen-4B, apply filter-1.

Labeler: **Qwen/Qwen3Guard-Gen-4B**. It emits a 3-way verdict per (prompt, answer):
  - "Safety: Safe | Unsafe | Controversial"
Controversial -> Unsafe (conservative). The raw 3-way label is kept in `guard_label3`
so the mapping stays auditable; downstream stages read only the binary `guard_label`.
(WildGuard is NOT used here — it is held out as the independent verifier for later
evaluation, so it must not touch labelling/training. It remains only in the optional
sanity_beaver.py cross-check.)

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
from utils import set_seed, extract_visible_response

SAFETY_RE = re.compile(r"Safety:\s*(Safe|Unsafe|Controversial)", re.IGNORECASE)


def parse_label(text):
    """Return (guard_label, guard_label3) from a Qwen3Guard generation.

    guard_label3 is the raw 3-way verdict ('Safe' | 'Unsafe' | 'Controversial');
    guard_label is the binary label fed to filter-1/filter-3 ('Controversial' -> 'Unsafe').
    Both are None if no "Safety:" line was parsed.
    """
    m = SAFETY_RE.search(text)
    if not m:
        return None, None
    raw = m.group(1).capitalize()
    return ("Safe" if raw == "Safe" else "Unsafe"), raw


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
        gen = model.generate(
            **enc, max_new_tokens=config.GUARD_MAX_NEW_TOKENS, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
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

    # 2) Flatten all valid (prompt, answer) pairs and classify with Qwen3Guard.
    flat = [(pi, ri) for pi, pp in enumerate(all_responses)
            for ri, rd in enumerate(pp) if not rd["invalid"]]
    print(f"Classifying {len(flat)} valid (prompt, answer) pairs with {config.GUARD_MODEL} ...")

    tokenizer = AutoTokenizer.from_pretrained(config.GUARD_MODEL)
    tokenizer.padding_side = "left"      # batched generation
    tokenizer.truncation_side = "left"   # if over-long, keep the answer (verdict reads the tail)
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
            label, label3 = parse_label(raw)
            rd = all_responses[pi][ri]
            rd["guard_raw"] = raw
            rd["guard_label"] = label     # 'Safe' | 'Unsafe' | None (Controversial -> 'Unsafe')
            rd["guard_label3"] = label3   # raw 3-way verdict, for auditing the mapping
        if (b // config.GUARD_BATCH) % 20 == 0:
            print(f"  {min(b + config.GUARD_BATCH, len(flat))}/{len(flat)}")

    # Label distribution — confirms the guard is emitting parseable "Safety:" verdicts, not
    # None, and surfaces how many Unsafe labels came via the Controversial mapping.
    from collections import Counter
    dist3 = Counter(all_responses[pi][ri]["guard_label3"] for pi, ri in flat)
    n_safe = dist3.get("Safe", 0)
    n_unsafe = dist3.get("Unsafe", 0)
    n_contro = dist3.get("Controversial", 0)
    n_none = dist3.get(None, 0)
    print(f"\nQwen3Guard label distribution: Safe={n_safe} Unsafe={n_unsafe} "
          f"Controversial={n_contro} (mapped to Unsafe) unparsed(None)={n_none}")
    if flat and n_none == len(flat):
        print("  WARNING: EVERY guard output was unparsed — the Qwen3Guard template is likely "
              "wrong. Try flipping add_generation_prompt=True in classify_batch(), and eyeball "
              "a guard_raw string in 03_all_labeled.joblib.")

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
