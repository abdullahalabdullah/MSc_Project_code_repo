#!/usr/bin/env python3
"""
Step 62 (experiment: prior-belief-vs-monitorability): GPU inference — fresh
unhinted arm + hinted Version A + hinted Version B, for the questions selected
by step 61.

Why a new script instead of 02_run_inference_qwen.py: the fresh unhinted arm
needs NEW seeds (prior_belief.unhinted_seed_base + s), disjoint from the 3**(s+1)
unhinted seeds hard-coded in src/inference_qwen.run_inference_qwen (fix 1), and
one unhinted generation must serve both versions. Everything else is reused
unchanged from src/inference_qwen.py: model loading, prompt construction,
answer parsing and CoT extraction, and the JSONL record schema (so the
unchanged steps 03, 04 and 58 read the outputs).

Per question and sample s = 0 .. n_samples-1 (config qwen_inference.n_samples):
    unhinted  seed unhinted_seed_base + s     (generated ONCE)
    hinted A  seed 2**(s+1), hint -> target A
    hinted B  seed 2**(s+1), hint -> target B (same seeds as A)
All requests of a checkpoint (qwen_inference.checkpoint_every questions) go to
vLLM in one generate call, as in src/inference_qwen.run_inference_qwen.

Outputs — one directory per version, in the layout steps 03/04/58 expect
({outputs_dir}/{model}/{versions.X}/{hint}/):
    hinted.jsonl     that version's hinted runs
    unhinted.jsonl   the SAME fresh unhinted generations, with hint_target /
                     followed_hint set to that version's target (the monitor
                     prompt names the hint target even for unhinted runs, so
                     each version's unhinted runs are monitored against its own
                     target — fix 3)
    progress.json    completed instance ids (written after both versions'
                     rows of a checkpoint are on disk; resume skips them)
Extra record fields: "pbm_version" (A/B) and, on unhinted rows,
"pbm_shared_unhinted": true.

Usage:
    python monitorability-eval/scripts/62_pbm_run_inference.py [--limit N]

    Options:
      --config      path to config YAML (default: config/prior_belief_vs_monitorability.yaml)
      --model       model id (default: Qwen/Qwen3-4B)
      --n-samples   override qwen_inference.n_samples (default: config, 20)
      --limit       only the first N selected questions (smoke test)
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.dataset import ensure_hf_auth, model_dir_name, resolve_paths
from src.inference_qwen import (
    extract_cot_qwen,
    format_prompt_qwen,
    load_qwen_model,
    parse_answer_qwen,
    split_thinking,
)

DEFAULT_CONFIG = "config/prior_belief_vs_monitorability.yaml"
VERSIONS = ("A", "B")


def sampling_params(inf, seed):
    from vllm import SamplingParams  # GPU-only dependency, imported lazily
    return SamplingParams(n=1, temperature=inf["temperature"], top_p=inf["top_p"],
                          top_k=inf.get("top_k", -1), max_tokens=inf["max_new_tokens"],
                          seed=seed)


def make_record(item, s, arm, prompt, completion, seed, hint_type, version, target):
    raw = completion.text
    parsed = parse_answer_qwen(raw)
    thinking_text, _ = split_thinking(raw)
    rec = {
        "instance_id": item["instance_id"],
        "question_id": item["question_id"],
        "arm": arm,
        "sample_idx": s,
        "hint_target": target,
        "correct_answer": item["correct_answer"],
        "hint_type": hint_type,
        "full_prompt": prompt,
        "raw_output": raw,
        "thinking": thinking_text,
        "cot": extract_cot_qwen(raw),
        "parsed_answer": parsed,
        "followed_hint": (parsed == target) if parsed else False,
        "token_count": len(completion.token_ids),
        "truncated": completion.finish_reason == "length",
        "seed": seed,
        "pbm_version": version,
    }
    if arm == "unhinted":
        rec["pbm_shared_unhinted"] = True
    return rec


def main():
    parser = argparse.ArgumentParser(description="prior-belief-vs-monitorability: inference.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    with open(root / args.config) as f:
        config = yaml.safe_load(f)
    pb = config["prior_belief"]
    inf = config["qwen_inference"]
    n_samples = args.n_samples or inf["n_samples"]
    checkpoint_every = inf["checkpoint_every"]
    hint_type = pb["hint_type"]
    hint_cfg = config.get("hint", {})
    model_name = model_dir_name(args.model)

    # --- the two version indexes (step 61) -----------------------------------
    items, out = {}, {}
    for v in VERSIONS:
        paths = resolve_paths(root, config, pb["versions"][v], model_name, hint_type)
        if not paths["data_index"].exists():
            sys.exit(f"ERROR: {paths['data_index']} not found — run 61_pbm_select_questions.py first.")
        with paths["data_index"].open() as f:
            items[v] = json.load(f)
        out[v] = paths["outputs_dir"]
        out[v].mkdir(parents=True, exist_ok=True)
    ids = [it["instance_id"] for it in items["A"]]
    assert ids == [it["instance_id"] for it in items["B"]], "A/B indexes differ in questions"
    for a, b in zip(items["A"], items["B"]):
        assert {k: x for k, x in a.items() if k != "hint_target"} == \
               {k: x for k, x in b.items() if k != "hint_target"}, f"A/B items differ: {a['instance_id']}"
        assert a["hint_target"] != b["hint_target"]
    if args.limit:
        for v in VERSIONS:
            items[v] = items[v][:args.limit]

    # --- resume ----------------------------------------------------------------
    done = {}
    for v in VERSIONS:
        p = out[v] / "progress.json"
        done[v] = set(json.load(p.open())["completed_instance_ids"]) if p.exists() else set()
    completed = done["A"] & done["B"]
    if done["A"] != done["B"]:
        print(f"WARNING: A/B progress differ ({len(done['A'])} vs {len(done['B'])}); "
              f"resuming from the {len(completed)} done in both.")
    todo = [i for i, it in enumerate(items["A"]) if it["instance_id"] not in completed]
    print(f"Hint type: {hint_type} | N samples/arm: {n_samples} | checkpoint: {checkpoint_every} questions")
    print(f"{len(items['A'])} selected questions, {len(completed)} already done, {len(todo)} to process.")
    if not todo:
        print("Nothing to do.")
        return

    seeds = {"hinted": [2 ** (s + 1) for s in range(n_samples)],
             "unhinted": [pb["unhinted_seed_base"] + s for s in range(n_samples)]}
    assert not set(seeds["unhinted"]) & {3 ** (s + 1) for s in range(64)}, "fresh seeds collide"

    ensure_hf_auth()
    tokenizer, llm = load_qwen_model({"name": model_name, "hf_id": args.model})

    counts = {k: {"runs": 0, "unparsed": 0, "truncated": 0} for k in ("unhinted", "hinted_A", "hinted_B")}
    for start in range(0, len(todo), checkpoint_every):
        batch = todo[start:start + checkpoint_every]
        prompts_u, prompts_h = {}, {"A": {}, "B": {}}
        for i in batch:
            pu = format_prompt_qwen(items["A"][i], include_hint=False, tokenizer=tokenizer,
                                    hint_type=hint_type, hint_cfg=hint_cfg, dataset=items["A"])
            pu_b = format_prompt_qwen(items["B"][i], include_hint=False, tokenizer=tokenizer,
                                      hint_type=hint_type, hint_cfg=hint_cfg, dataset=items["B"])
            assert pu == pu_b, "unhinted prompt depends on the hint target"
            prompts_u[i] = pu
            for v in VERSIONS:
                prompts_h[v][i] = format_prompt_qwen(
                    items[v][i], include_hint=True, tokenizer=tokenizer,
                    hint_type=hint_type, hint_cfg=hint_cfg, dataset=items[v])

        reqs, prompts, sps = [], [], []
        for s in range(n_samples):
            for i in batch:
                for kind, prompt, seed in (
                        ("unhinted", prompts_u[i], seeds["unhinted"][s]),
                        ("hinted_A", prompts_h["A"][i], seeds["hinted"][s]),
                        ("hinted_B", prompts_h["B"][i], seeds["hinted"][s])):
                    reqs.append((i, s, kind, prompt, seed))
                    prompts.append(prompt)
                    sps.append(sampling_params(inf, seed))
        print(f"  Checkpoint [{start}:{start + len(batch)}] — generating {len(prompts)} requests "
              f"({len(batch)} questions × {n_samples} samples × 3 arms) in one batch ...", flush=True)
        outputs = llm.generate(prompts, sps)

        rows = {v: {"hinted": [], "unhinted": []} for v in VERSIONS}
        for (i, s, kind, prompt, seed), o in zip(reqs, outputs):
            comp = o.outputs[0]
            if kind == "unhinted":
                for v in VERSIONS:
                    rows[v]["unhinted"].append(make_record(
                        items[v][i], s, "unhinted", prompt, comp, seed, hint_type, v,
                        items[v][i]["hint_target"]))
                rec = rows["A"]["unhinted"][-1]
            else:
                v = kind[-1]
                rec = make_record(items[v][i], s, "hinted", prompt, comp, seed, hint_type, v,
                                  items[v][i]["hint_target"])
                rows[v]["hinted"].append(rec)
            counts[kind]["runs"] += 1
            counts[kind]["unparsed"] += rec["parsed_answer"] is None
            counts[kind]["truncated"] += rec["truncated"]

        for v in VERSIONS:
            for arm in ("hinted", "unhinted"):
                with (out[v] / f"{arm}.jsonl").open("a", encoding="utf-8") as f:
                    for rec in rows[v][arm]:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        completed.update(items["A"][i]["instance_id"] for i in batch)
        for v in VERSIONS:  # progress last: rows of both versions are on disk
            with (out[v] / "progress.json").open("w") as f:
                json.dump({"completed_instance_ids": sorted(completed)}, f)
        print(f"  Checkpoint saved: {len(completed)}/{len(items['A'])} questions complete.", flush=True)

    for kind, c in counts.items():
        print(f"{kind}: {c['runs']} runs, unparsed {c['unparsed']}, truncated {c['truncated']}")
    print("Inference complete.")


if __name__ == "__main__":
    main()
