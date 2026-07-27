"""Average think-token count over the held-out eval set (CoT baseline / monitor).

Run the BASE model first to store a baseline, then re-run on any checkpoint to
quantify how much chain-of-thought survived SFT (see sft/think_stats.py).

    # 1) baseline (run before training): stores outputs/think_stats/base.json
    python scripts/05_think_token_eval.py --config configs/sft_harmlessness.yaml --checkpoint base

    # 2) a trained adapter (auto-compares against the stored base baseline)
    python scripts/05_think_token_eval.py --config configs/sft_harmlessness.yaml \
        --checkpoint outputs/qwen3-4b-sft-harmlessness/checkpoint-1065

    # or a pushed Hub repo
    python scripts/05_think_token_eval.py --config configs/sft_harmlessness.yaml \
        --checkpoint er-abd/qwen3-4b-sft-harmlessness-pku-adapter-epoch3
"""

import argparse

import _bootstrap  # noqa: F401
from sft.config import load_config
from sft.credentials import check_credentials
from sft.generate import load_tokenizer_for_gen
from sft.logging_utils import make_logger
from sft.think_stats import (
    DEFAULT_STATS_DIR,
    build_eval_prompts,
    load_stats,
    log_summary,
    save_stats,
    think_token_stats_on_eval,
)


def main():
    p = argparse.ArgumentParser(
        description="RO2 — average think-token count over the eval set."
    )
    p.add_argument("--config", default="configs/sft_harmlessness.yaml")
    p.add_argument("--checkpoint", default="base",
                   help="'base' for the un-adapted base model (the baseline), or a "
                        "local checkpoint dir / Hub adapter repo id.")
    p.add_argument("--max-eval-samples", type=int, default=200,
                   help="Cap eval prompts for tractable generation (default 200; "
                        "pass 0 for the full eval split).")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--sample", action="store_true",
                   help="Sample (temp 0.6, top_p 0.95) like the RO1 eval. Default is "
                        "greedy, for reproducible and model-to-model comparable counts.")
    p.add_argument("--label", default=None,
                   help="Override the stored file name (default: derived from --checkpoint).")
    p.add_argument("--out-dir", default=DEFAULT_STATS_DIR)
    args = p.parse_args()

    logger = make_logger("think_token_eval")
    cfg = load_config(args.config)
    check_credentials(logger)

    tokenizer = load_tokenizer_for_gen(cfg["base_model"])
    prompts = build_eval_prompts(
        cfg, logger, max_samples=(args.max_eval_samples or None)
    )

    stats = think_token_stats_on_eval(
        base_model_id=cfg["base_model"],
        adapter_dir=args.checkpoint,
        tokenizer=tokenizer,
        prompts=prompts,
        logger=logger,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.sample,
        batch_size=args.batch_size,
    )
    if args.label:
        stats["label"] = args.label

    baseline = None
    if stats["label"] != "base":
        baseline = load_stats("base", args.out_dir)
        if baseline is None:
            logger.warning(
                "No base baseline found in %s. Run with --checkpoint base first "
                "to enable the retained-CoT comparison.", args.out_dir
            )

    log_summary(stats, logger, baseline=baseline)
    path = save_stats(stats, args.out_dir)
    logger.info("Saved think-token stats -> %s", path)


if __name__ == "__main__":
    main()
