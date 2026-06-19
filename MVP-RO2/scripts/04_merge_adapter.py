"""Merge a pushed LoRA adapter into the base model for evaluation (spec §8.2).

    # merge one epoch's adapter into a local standalone model
    python scripts/04_merge_adapter.py \
        --adapter er-abd/qwen3-4b-sft-harmlessness-pku-adapter-epoch3 \
        --out ./merged/qwen3-4b-sft-harmlessness-epoch3

    # optionally push the merged model to the Hub (private)
    python scripts/04_merge_adapter.py \
        --adapter er-abd/qwen3-4b-sft-harmlessness-pku-adapter-epoch3 \
        --out ./merged/qwen3-4b-sft-harmlessness-epoch3 \
        --push er-abd/qwen3-4b-sft-harmlessness-epoch3

The adapter repos are adapter-only (no base weights). This loads the base model
named in the adapter's config, applies the LoRA, calls merge_and_unload(), and
saves a full standalone model + tokenizer that RO1's vLLM pipeline can load by
path (or by Hub id if --push is given). Run with --base to override the base.
"""

import argparse

import _bootstrap  # noqa: F401
from sft.credentials import check_credentials
from sft.logging_utils import make_logger


def main():
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into its base model.")
    parser.add_argument("--adapter", required=True,
                        help="Adapter repo id or local dir (e.g. er-abd/...-epoch3).")
    parser.add_argument("--out", required=True,
                        help="Local dir to write the merged standalone model + tokenizer.")
    parser.add_argument("--base", default=None,
                        help="Base model id. Default: read from the adapter's config.")
    parser.add_argument("--dtype", default="bfloat16", help="torch dtype for load/save.")
    parser.add_argument("--push", default=None,
                        help="Optional Hub repo id to upload the merged model to (private).")
    args = parser.parse_args()

    logger = make_logger("merge_adapter")

    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.dtype)

    base_id = args.base or PeftConfig.from_pretrained(args.adapter).base_model_name_or_path
    logger.info("Base model: %s", base_id)
    logger.info("Adapter:    %s", args.adapter)

    logger.info("Loading base model ...")
    base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=dtype)

    logger.info("Applying adapter and merging ...")
    model = PeftModel.from_pretrained(base, args.adapter)
    model = model.merge_and_unload()

    # Tokenizer: prefer the one shipped in the adapter repo (pushed at train time);
    # fall back to the base model's tokenizer.
    try:
        tok = AutoTokenizer.from_pretrained(args.adapter)
    except Exception:  # noqa: BLE001
        tok = AutoTokenizer.from_pretrained(base_id)

    logger.info("Saving merged model to %s ...", args.out)
    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    logger.info("Merge complete -> %s", args.out)

    if args.push:
        check_credentials(logger)
        logger.info("Pushing merged model to %s (private) ...", args.push)
        model.push_to_hub(args.push, private=True, safe_serialization=True)
        tok.push_to_hub(args.push, private=True)
        logger.info("PUSH OK -> %s", args.push)


if __name__ == "__main__":
    main()
