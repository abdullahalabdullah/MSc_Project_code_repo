"""Merge a LoRA adapter into a base model and push the merged model to the Hub.

Usage:
    python monitorability-eval/scripts/merge_and_push.py \
        --base   Qwen/Qwen3-1.7B \
        --adapter er-abd/aqua-dpo-qwen3-1.7B-sk-llama_original-seed-0 \
        --merged-repo er-abd/aqua-dpo-qwen3-1.7B-sk-llama_original-seed-0-merged \
        --out-dir ./merged_aqua_dpo_qwen3_1_7b_s0

    # Local merge only, no push:
    python monitorability-eval/scripts/merge_and_push.py ... --no-push
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.dataset import ensure_hf_auth


def merge_and_push(base: str, adapter: str, out_dir: str, merged_repo: str,
                   push: bool, public: bool, dtype=torch.bfloat16):
    print(f"Loading base model {base} ({dtype}) ...")
    base_model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype)

    print(f"Attaching adapter {adapter} ...")
    model = PeftModel.from_pretrained(base_model, adapter)

    print("Merging adapter into base weights (merge_and_unload) ...")
    merged = model.merge_and_unload()

    os.makedirs(out_dir, exist_ok=True)
    merged.save_pretrained(out_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(base).save_pretrained(out_dir)
    print(f"Merged model + tokenizer saved to {out_dir}")

    if not push:
        print(f"--no-push set; done. Pass the local dir as the model ID:\n  --model {out_dir}")
        return

    api = HfApi()
    api.create_repo(merged_repo, repo_type="model", private=not public, exist_ok=True)
    print(f"Uploading {out_dir} -> {merged_repo} (private={not public}) ...")
    api.upload_folder(folder_path=out_dir, repo_id=merged_repo, repo_type="model")
    info = api.repo_info(repo_id=merged_repo, repo_type="model")
    commit = getattr(info, "sha", "unknown")
    print(f"Push OK -> {merged_repo} (commit={commit})")
    print(f"\nRun the pipeline with the merged model as the model ID:\n"
          f"  bash monitorability-eval/run_inference.sh {merged_repo} ALL ALL")


def main():
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into a base model and push to Hub.")
    parser.add_argument("--base", required=True, help="Base model HF id (e.g. Qwen/Qwen3-1.7B)")
    parser.add_argument("--adapter", required=True, help="Adapter HF repo id or local checkpoint dir")
    parser.add_argument("--merged-repo", required=True, help="Target Hub repo id for the merged model")
    parser.add_argument("--out-dir", required=True, help="Local directory to save the merged model")
    parser.add_argument("--no-push", action="store_true", help="Merge locally, do not push to Hub")
    parser.add_argument("--public", action="store_true", help="Push as a public repo (default: private)")
    args = parser.parse_args()

    ensure_hf_auth()

    merge_and_push(
        base=args.base,
        adapter=args.adapter,
        out_dir=args.out_dir,
        merged_repo=args.merged_repo,
        push=not args.no_push,
        public=args.public,
    )


if __name__ == "__main__":
    main()
