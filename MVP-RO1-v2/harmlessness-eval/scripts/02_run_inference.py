#!/usr/bin/env python3
"""
Step 2 (harmlessness): Generate target-model responses to the WildJailbreak prompts.

Qwen3 thinking-mode free-form generation (no MCQ/hints). Reuses the shared
`qwen_inference` sampling block; one response per prompt by default. Stores the
full output, the <think> trace, and the visible answer (post-</think>) that the
WildGuard classifier will judge.

Usage:
    python harmlessness-eval/scripts/02_run_inference.py --model Qwen/Qwen3-4B

Input:
    harmlessness-eval/data/wildjailbreak_eval/prompts.jsonl
Output:
    outputs-and-results/harmlessness/responses/{model}/responses.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from src.dataset import ensure_hf_auth, model_dir_name, load_prompts
from src.inference import load_qwen_model, generate_responses

REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="Generate harmlessness responses for one model.")
    parser.add_argument("--model", required=True,
                        help="HF model ID (e.g. Qwen/Qwen3-4B) or local model dir.")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    with open(config_path) as f:
        config = yaml.safe_load(f)

    h = config["harmlessness"]
    model_name = model_dir_name(args.model)

    index_path = REPO_ROOT / h["paths"]["data_dir"] / "wildjailbreak_eval" / "prompts.jsonl"
    if not index_path.exists():
        print(f"ERROR: prompts index not found at {index_path}. Run 01_prepare_wildjailbreak.py first.")
        sys.exit(1)
    prompts = load_prompts(index_path)
    print(f"Loaded {len(prompts)} prompts from {index_path}")

    out_dir = REPO_ROOT / h["paths"]["responses_dir"] / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "responses.jsonl"

    ensure_hf_auth()
    tokenizer, llm = load_qwen_model(args.model)

    records = generate_responses(
        tokenizer, llm, prompts,
        gen_cfg=h["generation"],
        sampling_cfg=config["qwen_inference"],
    )

    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_trunc = sum(1 for r in records if r["truncated"])
    print(f"  Wrote {len(records)} responses to {out_path} ({n_trunc} truncated).")


if __name__ == "__main__":
    main()
