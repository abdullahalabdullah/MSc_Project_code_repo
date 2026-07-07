#!/usr/bin/env python3
"""
Step 3 (harmlessness): Classify the responses with WildGuard (independent judge).

For each response, WildGuard judges the (prompt, response) pair for safety and
refusal. `classify_target` selects what text is judged: "answer" (the visible
post-</think> response, default) or "full" (the whole generation). Runs locally
on GPU via vLLM.

Usage:
    python harmlessness-eval/scripts/03_run_wildguard.py --model Qwen/Qwen3-4B

Input:
    outputs-and-results/harmlessness/responses/{model}/responses.jsonl
Output:
    outputs-and-results/harmlessness/wildguard/{model}/classifications.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from src.dataset import ensure_hf_auth, model_dir_name
from src.wildguard import load_wildguard, classify

REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="Classify harmlessness responses with WildGuard.")
    parser.add_argument("--model", required=True,
                        help="HF model ID used at inference (e.g. Qwen/Qwen3-4B).")
    parser.add_argument("--config", default="config/mvp_config.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    with open(config_path) as f:
        config = yaml.safe_load(f)

    h = config["harmlessness"]
    model_name = model_dir_name(args.model)
    classify_target = h.get("classify_target", "answer")

    resp_path = REPO_ROOT / h["paths"]["responses_dir"] / model_name / "responses.jsonl"
    if not resp_path.exists():
        print(f"ERROR: responses not found at {resp_path}. Run 02_run_inference.py first.")
        sys.exit(1)

    records = []
    with open(resp_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            # What WildGuard judges: the visible answer or the full generation.
            response_text = r["raw_output"] if classify_target == "full" else r["answer"]
            records.append({
                "prompt_id":  r["prompt_id"],
                "subset":     r["subset"],
                "sample_idx": r["sample_idx"],
                "prompt":     r["prompt"],
                "response":   response_text,
            })
    print(f"Loaded {len(records)} responses (classify_target={classify_target}) from {resp_path}")

    out_dir = REPO_ROOT / h["paths"]["wildguard_dir"] / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "classifications.jsonl"

    ensure_hf_auth()
    tokenizer, llm = load_wildguard(h["classifier"]["hf_id"],
                                    max_model_len=h["classifier"].get("max_model_len", 8192))

    results = classify(llm, tokenizer, records, h["classifier"])

    n_missing = sum(1 for r in results if r["response_harmful"] is None or r["refusal"] is None)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(results)} classifications to {out_path} ({n_missing} unparsed verdicts).")


if __name__ == "__main__":
    main()
