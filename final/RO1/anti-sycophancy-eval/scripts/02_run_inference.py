#!/usr/bin/env python3
"""
Step 2: Target-model inference for one model (GPU).

Three generation passes with ONE vLLM engine load, all with the frozen anti_sycophancy
sampling config (spec §2: temp 1.0, thinking ON) and fixed per-sample seeds — identical
across models, which is what makes the base-vs-trained comparison paired:

  1. ID (Eval set 1): every question x {bare, wrong_opinion, right_opinion} arms.
  2. OOD (Eval set 2) turn 1: the bare question.
  3. OOD turn 2: the "Are you sure?" challenge appended to EVERY turn-1 sample
     regardless of correctness — no API round-trip inside the GPU session; the
     regressive/progressive conditioning happens at metric time (both branches of the
     M3 split need challenge responses anyway).

Usage:
    python scripts/02_run_inference.py --model Qwen/Qwen3-4B
    python scripts/02_run_inference.py --model er-abd/anti-syc-dpo-qwen3-4B-step-N --merge
    Options: --config, --limit N (smoke tests), --force

Output (namespaced by model):
    outputs-and-results/anti-sycophancy-results/responses/{model}/id_responses.jsonl
    outputs-and-results/anti-sycophancy-results/responses/{model}/ood_turn1_responses.jsonl
    outputs-and-results/anti-sycophancy-results/responses/{model}/ood_responses.jsonl
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from src.dataset import ensure_hf_auth, load_jsonl, model_dir_name, resolve_paths, write_jsonl
from src.inference import (build_challenge_prompt, build_prompt, generate_batch,
                           load_qwen_model)
from src.prompts import ANSWER_FORMAT_INSTRUCTION

# Fixed seed schemes (identical for every model — the pairing requirement).
ID_TURN_SEED_BASE = 1000    # ID arms + OOD turn 1: seed = 1000 + sample_idx
CHALLENGE_SEED_BASE = 2000  # OOD turn 2: seed = 2000 + sample_idx


def main():
    parser = argparse.ArgumentParser(description="Anti-sycophancy inference for one model (GPU).")
    parser.add_argument("--model", required=True,
                        help="HF model ID or local dir. Namespaces all output dirs.")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--merge", action="store_true",
                        help="Treat --model as a LoRA adapter repo: merge into the config "
                             "merge: base first, run inference on the merged dir.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap questions in BOTH eval sets (overrides config limit).")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate even when response files exist.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    with open(root / args.config) as f:
        config = yaml.safe_load(f)
    a_cfg = config["anti_sycophancy"]
    inf_cfg = a_cfg["inference"]
    n_samples = inf_cfg.get("n_samples", 1)
    limit = args.limit if args.limit is not None else a_cfg.get("limit")

    model_name = model_dir_name(args.model)
    paths = resolve_paths(root, config, model_name)
    out_dir = paths["responses_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    id_path = out_dir / "id_responses.jsonl"
    ood_t1_path = out_dir / "ood_turn1_responses.jsonl"
    ood_path = out_dir / "ood_responses.jsonl"

    if id_path.exists() and ood_path.exists() and not args.force:
        print(f"Responses already exist for {model_name} ({id_path}, {ood_path}) — "
              f"skipping. Use --force to regenerate.")
        return

    for p in (paths["eval_id_index"], paths["eval_ood_index"]):
        if not p.exists():
            sys.exit(f"ERROR: {p} not found. Run scripts/01_prepare_eval_data.py first.")

    id_index = load_jsonl(paths["eval_id_index"])
    ood_index = load_jsonl(paths["eval_ood_index"])
    if limit is not None:
        id_index = id_index[:limit]
        ood_index = ood_index[:limit]
    print(f"Eval set 1 (ID): {len(id_index)} questions x 3 arms x {n_samples} samples")
    print(f"Eval set 2 (OOD): {len(ood_index)} questions x {n_samples} samples x 2 turns")

    ensure_hf_auth()

    # --- Optional adapter merge (same flow as the harmlessness eval) -----------
    model_path = args.model
    merged_dir = None
    if args.merge:
        from src.merge import merge_adapter
        merged_dir = root / config["merge"]["out_root"] / model_name
        model_path = str(merge_adapter(base=config["merge"]["base"], adapter=args.model,
                                       out_dir=merged_dir))

    tokenizer, llm = load_qwen_model(model_path, inf_cfg)
    if merged_dir is not None and Path(merged_dir).exists():
        # Weights are in GPU memory now; free the disk copy (small container disks).
        print(f"Deleting merged model dir {merged_dir} (already loaded).")
        shutil.rmtree(merged_dir, ignore_errors=True)

    enable_thinking = inf_cfg.get("enable_thinking", True)

    # --- Pass 1: ID arms --------------------------------------------------------
    if id_path.exists() and not args.force:
        print(f"Pass 1 output exists ({id_path}) — skipping.")
    else:
        metas, prompt_strs, seeds = [], [], []
        for rec in id_index:
            arms = [("bare", rec["bare_prompt"], None),
                    ("wrong_opinion", rec["wrong_prompt"], rec["wrong_template_id"]),
                    ("right_opinion", rec["right_prompt"], rec["right_template_id"])]
            for arm, prompt, template_id in arms:
                templated = build_prompt(tokenizer, prompt, enable_thinking)
                for s in range(n_samples):
                    metas.append((rec, arm, template_id, s))
                    prompt_strs.append(templated)
                    seeds.append(ID_TURN_SEED_BASE + s)
        print(f"Pass 1 (ID): {len(prompt_strs)} requests in one continuous-batched call")
        results = generate_batch(llm, prompt_strs, inf_cfg, seeds)
        lines = []
        for (rec, arm, template_id, s), r in zip(metas, results):
            lines.append({"question_id": rec["question_id"], "arm": arm,
                          "template_id": template_id, "sample_idx": s, **r})
        write_jsonl(id_path, lines)
        n_trunc = sum(1 for l in lines if l["truncated"])
        print(f"  -> {id_path} ({len(lines)} responses, {n_trunc} truncated)")

    # --- Pass 2: OOD turn 1 -------------------------------------------------------
    if ood_t1_path.exists() and not args.force:
        print(f"Pass 2 output exists ({ood_t1_path}) — reusing.")
        turn1_lines = load_jsonl(ood_t1_path)
    else:
        metas, prompt_strs, seeds = [], [], []
        for rec in ood_index:
            templated = build_prompt(tokenizer, rec["turn1_prompt"], enable_thinking)
            for s in range(n_samples):
                metas.append((rec, s))
                prompt_strs.append(templated)
                seeds.append(ID_TURN_SEED_BASE + s)
        print(f"Pass 2 (OOD turn 1): {len(prompt_strs)} requests")
        results = generate_batch(llm, prompt_strs, inf_cfg, seeds)
        turn1_lines = [{"question_id": rec["question_id"], "sample_idx": s, **r}
                       for (rec, s), r in zip(metas, results)]
        write_jsonl(ood_t1_path, turn1_lines)
        print(f"  -> {ood_t1_path} ({len(turn1_lines)} responses)")

    # --- Pass 3: OOD turn 2 (challenge appended to EVERY turn-1 sample) -----------
    if ood_path.exists() and not args.force:
        print(f"Pass 3 output exists ({ood_path}) — skipping.")
    else:
        turn1_prompt_of = {rec["question_id"]: rec["turn1_prompt"] for rec in ood_index}
        challenge_msg = f"{a_cfg['challenge_text']}\n\n{ANSWER_FORMAT_INSTRUCTION}"
        metas, prompt_strs, seeds = [], [], []
        for t1 in turn1_lines:
            templated = build_challenge_prompt(
                tokenizer, turn1_prompt_of[t1["question_id"]], t1["answer"],
                challenge_msg, enable_thinking)
            metas.append(t1)
            prompt_strs.append(templated)
            seeds.append(CHALLENGE_SEED_BASE + t1["sample_idx"])
        print(f"Pass 3 (OOD turn 2): {len(prompt_strs)} requests")
        results = generate_batch(llm, prompt_strs, inf_cfg, seeds)
        lines = []
        for t1, r in zip(metas, results):
            lines.append({
                "question_id": t1["question_id"],
                "sample_idx": t1["sample_idx"],
                **{f"turn1_{k}": t1[k] for k in
                   ("raw_output", "thinking", "answer", "token_count", "truncated", "seed")},
                **{f"turn2_{k}": r[k] for k in
                   ("raw_output", "thinking", "answer", "token_count", "truncated", "seed")},
            })
        write_jsonl(ood_path, lines)
        print(f"  -> {ood_path} ({len(lines)} two-turn records)")

    print(f"\nInference complete for {model_name}. Next (no GPU needed): "
          f"bash anti-sycophancy-eval/run_antisyc_results.sh {args.model}")


if __name__ == "__main__":
    main()
