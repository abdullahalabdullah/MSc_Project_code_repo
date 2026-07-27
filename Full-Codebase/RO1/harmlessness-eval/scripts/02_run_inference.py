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
               namespaced by the ADAPTER id. The base model's HF cache and the
               local merged dir are both freed automatically once vLLM has
               loaded them (see src/merge.py), so re-running --merge for the
               same adapter re-downloads/re-merges rather than reusing a cached
               copy — a deliberate disk-over-time tradeoff for small container
               disks.

Input:
    harmlessness-eval/data/wildjailbreak_eval/[eval/]prompts.jsonl
Output:
    outputs-and-results/harmlessness-results/responses/{model}/[eval/]responses.jsonl
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from src.dataset import SPLITS, ensure_hf_auth, model_dir_name, load_prompts, resolve_paths
from src.inference import load_qwen_model, generate_responses

REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="Generate harmlessness responses for one model.")
    parser.add_argument("--model", required=True,
                        help="HF model ID (e.g. Qwen/Qwen3-4B) or local model dir.")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    parser.add_argument("--data", default="test", choices=SPLITS,
                        help="Data split: test (default) or eval (namespaced under .../eval/).")
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
    paths = resolve_paths(REPO_ROOT, config, model_name=model_name, split=args.data)

    index_path = paths["data_index"]
    if not index_path.exists():
        print(f"ERROR: '{args.data}' prompts index not found at {index_path}. "
              f"Run 01_prepare_wildjailbreak.py --data {args.data} first.")
        sys.exit(1)
    prompts = load_prompts(index_path)
    print(f"Loaded {len(prompts)} '{args.data}' prompts from {index_path}")

    out_path = paths["responses"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

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

    # vLLM has now loaded the merged weights into GPU memory — the on-disk
    # merged dir is redundant. Free it so it doesn't sit alongside WildGuard's
    # ~14.5GB download in phase 3 and fill a small container disk. A re-run of
    # --merge for this adapter just re-merges (slower, not broken).
    if args.merge:
        shutil.rmtree(merged_dir, ignore_errors=True)
        print(f"Freed local merged model dir {merged_dir} (already loaded into vLLM).")

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
