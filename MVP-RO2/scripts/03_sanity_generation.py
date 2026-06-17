"""Thinking-mode generation sanity check (spec §9.1) for one adapter checkpoint.

    # local checkpoint dir
    python scripts/03_sanity_generation.py --config configs/sft_harmlessness.yaml \
        --checkpoint ./outputs/qwen3-4b-sft-harmlessness/checkpoint-1065

    # or a pushed Hub repo
    python scripts/03_sanity_generation.py --config configs/sft_harmlessness.yaml \
        --checkpoint er-abd/qwen3-4b-sft-harmlessness-pku-adapter-epoch3

Confirms the model still produces non-trivial <think>...</think> blocks. Run per
epoch; if epoch 3 suppresses thinking, prefer epoch 1/2 for evaluation.
"""

import argparse

import _bootstrap  # noqa: F401
from sft.config import load_config
from sft.credentials import check_credentials
from sft.generate import generation_check, load_tokenizer_for_gen
from sft.logging_utils import make_logger


def main():
    parser = argparse.ArgumentParser(description="RO2 SFT — thinking-mode generation check.")
    parser.add_argument("--config", default="configs/sft_harmlessness.yaml")
    parser.add_argument("--checkpoint", required=True,
                        help="Local checkpoint dir, a Hub adapter repo id, or "
                             "'base' to run the un-adapted base model as a reference.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--min-think-tokens", type=int, default=20)
    parser.add_argument("--sample", action="store_true",
                        help="Sample (temp 0.6, top_p 0.95) like the RO1 eval, "
                             "instead of greedy (avoids repetition loops).")
    args = parser.parse_args()

    logger = make_logger("sanity_generation")
    cfg = load_config(args.config)
    check_credentials(logger)

    tokenizer = load_tokenizer_for_gen(cfg["base_model"])
    generation_check(
        base_model_id=cfg["base_model"],
        adapter_dir=args.checkpoint,
        tokenizer=tokenizer,
        logger=logger,
        max_new_tokens=args.max_new_tokens,
        min_think_tokens=args.min_think_tokens,
        do_sample=args.sample,
    )


if __name__ == "__main__":
    main()
