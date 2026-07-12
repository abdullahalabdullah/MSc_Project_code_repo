"""Stage 04 — Score surviving (prompt, answer) pairs with beaver-7b-unified-cost.

Runs in the SAME environment as the other stages: the cost model is loaded via the
self-contained loader in beaver_cost_model.py (plain transformers, no safe-rlhf). If
safe_rlhf happens to be installed it is used automatically; otherwise the local
LlamaModelForScore is used (see beaver_cost_model.py for the correctness argument).

Cost sign convention: higher cost = more harmful. Per prompt:
    chosen   = valid response with the MINIMUM cost (safest)
    rejected = valid response with the MAXIMUM cost (most harmful)

Scoring is forward-only (no generation). Pairs are **length-sorted** before batching so each
batch pads to a near-uniform length (right-padding otherwise wastes the 7B forward on pad
tokens); the score reads each sequence's own last real token, so it is independent of batch
composition and results are written back by original index. Scoring is **checkpointed** every
batch to <OUTPUT_DIR>/04_cost.ckpt.jsonl, so an interrupted run resumes instead of re-scoring
from scratch; a fingerprint invalidates a checkpoint left over from a different 03_labeled.joblib.

Output: config.SCORED_PATH (each prompt annotated with per-response cost + chosen/rejected idx).
"""
import hashlib
import json
import os

import joblib
import torch
from transformers import AutoTokenizer

import config
import env_setup
from utils import set_seed, update_diagnostics
from beaver_cost_model import load_cost_model, end_scores

SCORE_CKPT_PATH = os.path.join(config.OUTPUT_DIR, "04_cost.ckpt.jsonl")
CKPT_META_PATH = os.path.join(config.OUTPUT_DIR, "04_cost.ckpt.meta.json")


def score_batch(model, source_tag, tokenizer, pairs):
    """pairs: list of (prompt, answer). Returns list[float] costs (higher = more harmful)."""
    texts = [config.COST_PROMPT.format(prompt=p, answer=a) for p, a in pairs]
    # Left-truncate at Beaver's 2048 ctx so an over-long input drops the PROMPT head, not the
    # ASSISTANT answer (whose final token the score head reads). truncation_side is set at load.
    enc = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=config.BEAVER_MAX_CTX
    ).to(model.device)
    with torch.no_grad():
        scores = end_scores(model, source_tag, enc)  # (B,)
    return scores.float().cpu().tolist()


# ── Checkpointing ──────────────────────────────────────────────────────────────
# Append-only JSONL, one line per scored pair (fsync'd per batch); a torn trailing line from a
# crash is ignored on reload. A sidecar meta file fingerprints the inputs so a checkpoint left
# over from a *different* 03_labeled.joblib is discarded rather than silently reused. Costs are
# doubles and round-trip exactly through JSON.

def _fingerprint(flat, survivors):
    n = len(flat)
    probe = []
    for i in (0, n // 2, n - 1):
        if 0 <= i < n:
            pi, ri = flat[i]
            rd = survivors[pi][ri]
            probe.append(rd["prompt"])
            probe.append(rd["answer"])
    h = hashlib.sha256(("\x00".join(probe)).encode("utf-8")).hexdigest()
    return {"n_prompts": len(survivors), "n_flat": n, "sig": h}


def _reset_ckpt_if_stale(fp):
    """Keep the checkpoint only if it matches the current inputs; else archive + start fresh."""
    if os.path.exists(CKPT_META_PATH):
        with open(CKPT_META_PATH) as f:
            prev = json.load(f)
        if prev == fp:
            return  # valid resume
        print("  WARNING: existing 04 checkpoint does not match current 03_labeled.joblib "
              "(inputs changed) — archiving stale checkpoint and starting fresh.")
        for p in (SCORE_CKPT_PATH, CKPT_META_PATH):
            if os.path.exists(p):
                os.replace(p, p + ".stale")
    with open(CKPT_META_PATH, "w") as f:
        json.dump(fp, f)


def _load_ckpt(path):
    """Return {flat_index: cost} for all complete JSONL lines (ignoring a torn trailing line)."""
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
            done[rec["i"]] = rec["cost"]
    return done


def main():
    env_setup.setup_env()
    config.ensure_output_dir()
    set_seed(config.SEED)

    survivors = joblib.load(config.LABELED_PATH)
    print(f"Loaded {len(survivors)} prompts surviving filter-1")

    print(f"Loading cost model {config.COST_MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(config.COST_MODEL)
    tokenizer.padding_side = "right"    # local scorer reads the last real token via sum(mask)-1
    tokenizer.truncation_side = "left"  # keep the answer, drop the prompt head if over 2048
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model, source_tag = load_cost_model(config.COST_MODEL, torch_dtype=torch.bfloat16)
    print(f"Cost model loaded via: {source_tag}")

    # Flatten valid responses across all prompts.
    flat = [(pi, ri) for pi, pp in enumerate(survivors)
            for ri, rd in enumerate(pp) if not rd["invalid"]]
    print(f"Scoring {len(flat)} valid (prompt, answer) pairs ...")

    _reset_ckpt_if_stale(_fingerprint(flat, survivors))

    # Length-sorted order (throughput-only): each batch pads to a near-uniform length. The
    # char count is a cheap proxy for token length; the score is per-sequence, so ordering
    # never changes an item's cost — we write results back by original flat index.
    def est_len(i):
        pi, ri = flat[i]
        rd = survivors[pi][ri]
        return len(rd["prompt"]) + len(rd["answer"])

    order = sorted(range(len(flat)), key=est_len)

    done = _load_ckpt(SCORE_CKPT_PATH)
    for i, c in done.items():
        pi, ri = flat[i]
        survivors[pi][ri]["cost"] = c
    remaining = [i for i in order if i not in done]
    print(f"  {len(done)}/{len(flat)} already checkpointed, {len(remaining)} to score")

    n_done = len(done)
    for b in range(0, len(remaining), config.COST_BATCH):
        chunk = remaining[b:b + config.COST_BATCH]
        pairs = [(survivors[pi][ri]["prompt"], survivors[pi][ri]["answer"])
                 for pi, ri in (flat[i] for i in chunk)]
        costs = score_batch(model, source_tag, tokenizer, pairs)
        with open(SCORE_CKPT_PATH, "a") as f:
            for i, c in zip(chunk, costs):
                pi, ri = flat[i]
                survivors[pi][ri]["cost"] = c
                f.write(json.dumps({"i": i, "cost": c}) + "\n")
            f.flush()
            os.fsync(f.fileno())
        n_done += len(chunk)
        if (b // config.COST_BATCH) % 20 == 0:
            print(f"  {n_done}/{len(flat)}")

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

    all_costs = [survivors[pi][ri]["cost"] for pi, ri in flat]
    update_diagnostics(config.DIAGNOSTICS_PATH, "scoring", {
        "cost_model": config.COST_MODEL,
        "loader": source_tag,
        "n_prompts": len(survivors),
        "n_responses_scored": len(flat),
        "cost_mean": round(sum(all_costs) / max(len(all_costs), 1), 4),
        "cost_min": round(min(all_costs), 4) if all_costs else None,
        "cost_max": round(max(all_costs), 4) if all_costs else None,
    })
    print(f"Diagnostics -> {config.DIAGNOSTICS_PATH} [scoring]")

    # Scoring finished and SCORED_PATH is dumped — drop the resume checkpoint.
    for p in (SCORE_CKPT_PATH, CKPT_META_PATH):
        if os.path.exists(p):
            os.remove(p)


if __name__ == "__main__":
    main()
