"""Stage 04 — Score surviving (prompt, answer) pairs with beaver-7b-unified-cost.

Runs in the SAME environment as the other stages: the cost model is loaded via the
self-contained loader in beaver_cost_model.py (plain transformers, no safe-rlhf). If
safe_rlhf happens to be installed it is used automatically; otherwise the local
LlamaModelForScore is used (see beaver_cost_model.py for the correctness argument).

Cost sign convention: higher cost = more harmful. Per prompt:
    chosen   = valid response with the MINIMUM cost (safest)
    rejected = valid response with the MAXIMUM cost (most harmful)

Output: config.SCORED_PATH (each prompt annotated with per-response cost + chosen/rejected idx).
"""
import joblib
import torch
from transformers import AutoTokenizer

import config
import env_setup
from utils import set_seed
from beaver_cost_model import load_cost_model, end_scores

COST_PROMPT = "BEGINNING OF CONVERSATION: USER: {prompt} ASSISTANT:{answer}"


def score_batch(model, source_tag, tokenizer, pairs):
    """pairs: list of (prompt, answer). Returns list[float] costs (higher = more harmful)."""
    texts = [COST_PROMPT.format(prompt=p, answer=a) for p, a in pairs]
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        scores = end_scores(model, source_tag, enc)  # (B,)
    return scores.float().cpu().tolist()


def main():
    env_setup.setup_env()
    config.ensure_output_dir()
    set_seed(config.SEED)

    survivors = joblib.load(config.LABELED_PATH)
    print(f"Loaded {len(survivors)} prompts surviving filter-1")

    print(f"Loading cost model {config.COST_MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(config.COST_MODEL)
    tokenizer.padding_side = "right"  # local scorer reads the last real token via sum(mask)-1
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model, source_tag = load_cost_model(config.COST_MODEL, torch_dtype=torch.bfloat16)
    print(f"Cost model loaded via: {source_tag}")

    # Flatten valid responses across all prompts and score in batches.
    flat = [(pi, ri) for pi, pp in enumerate(survivors)
            for ri, rd in enumerate(pp) if not rd["invalid"]]
    print(f"Scoring {len(flat)} valid (prompt, answer) pairs ...")

    for b in range(0, len(flat), config.COST_BATCH):
        chunk = flat[b:b + config.COST_BATCH]
        pairs = [(survivors[pi][ri]["prompt"], survivors[pi][ri]["answer"]) for pi, ri in chunk]
        costs = score_batch(model, source_tag, tokenizer, pairs)
        for (pi, ri), c in zip(chunk, costs):
            survivors[pi][ri]["cost"] = c
        if (b // config.COST_BATCH) % 20 == 0:
            print(f"  {min(b + config.COST_BATCH, len(flat))}/{len(flat)}")

    # Per prompt: chosen = min cost (safest), rejected = max cost (most harmful).
    for per_prompt in survivors:
        valid = [(ri, rd["cost"]) for ri, rd in enumerate(per_prompt) if not rd["invalid"]]
        chosen_ix = min(valid, key=lambda x: x[1])[0]
        rejected_ix = max(valid, key=lambda x: x[1])[0]
        per_prompt_meta = {"chosen_ix": chosen_ix, "rejected_ix": rejected_ix}
        # stash on element 0's dict under a dedicated key to keep the list structure simple
        for rd in per_prompt:
            rd["_pair_meta"] = per_prompt_meta

    joblib.dump(survivors, config.SCORED_PATH, compress=True)
    print(f"Saved -> {config.SCORED_PATH}")


if __name__ == "__main__":
    main()
