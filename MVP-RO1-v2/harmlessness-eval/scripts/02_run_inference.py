#!/usr/bin/env python3
"""
Step 2 (harmlessness): Generate target-model responses to the WildJailbreak prompts.

Qwen3 thinking-mode free-form generation (no MCQ/hints). Reuses the shared
`qwen_inference` sampling block; one response per prompt by default. Stores the
full output, the <think> trace, and the visible answer (post-</think>) that the
WildGuard classifier will judge.

Usage:
    python harmlessness-eval/scripts/02_run_inference.py --model Qwen/Qwen3-4B
    python harmlessness-eval/scripts/02_run_inference.py --model er-abd/some-adapter --merge

Options:
    --merge    default False. Treat --model as a LoRA ADAPTER repo (adapter-only
               push, no config.json): merge it into the fixed base from the
               config `merge:` block, save the merged model locally under
               merge.out_root, and run inference on that. Output dirs stay
               namespaced by the ADAPTER id.

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
    parser.add_argument("--merge", action="store_true", default=False,
                        help="Treat --model as a LoRA adapter repo and merge it into the "
                             "fixed base from the config 'merge:' block before inference. "
                             "Outputs stay namespaced by the adapter id.")
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

    # With --merge, args.model is an adapter-only repo: merge it into the fixed
    # base (config merge.base) and point vLLM at the local merged dir. The
    # output namespace stays the ADAPTER id (model_name, above) so downstream
    # steps (03-05) are run with the same --model.
    model_id = args.model
    if args.merge:
        from src.merge import merge_adapter
        merge_cfg = config.get("merge", {})
        if not merge_cfg.get("base"):
            print("ERROR: --merge requires a 'merge:' block with 'base:' in the config.")
            sys.exit(1)
        merged_dir = merge_adapter(
            base=merge_cfg["base"],
            adapter=args.model,
            out_dir=REPO_ROOT / merge_cfg.get("out_root", "merged-models") / model_name,
        )
        model_id = str(merged_dir)

    tokenizer, llm = load_qwen_model(model_id)

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
