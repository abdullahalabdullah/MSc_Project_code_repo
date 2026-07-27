"""Check chain-of-thought preservation across ALL epoch checkpoints at once.

Runs the thinking-mode generation check on every checkpoint-* dir under the
config's output_dir (optionally the base model too), with eval-like sampling by
default, then prints a summary table of think-token counts per checkpoint so you
can pick the best eval target in one go.

    python scripts/check_all_epochs_cot.py
    python scripts/check_all_epochs_cot.py --greedy            # deterministic decoding
    python scripts/check_all_epochs_cot.py --no-base           # skip the base reference
    python scripts/check_all_epochs_cot.py --output-dir ./outputs/qwen3-4b-sft-harmlessness
"""

import argparse

import _bootstrap  # noqa: F401
from sft.config import load_config
from sft.credentials import check_credentials
from sft.generate import generation_check, load_tokenizer_for_gen
from sft.logging_utils import make_logger
from sft.push import discover_checkpoints


def main():
    parser = argparse.ArgumentParser(description="CoT-preservation check across all epoch checkpoints.")
    parser.add_argument("--config", default="configs/sft_harmlessness.yaml")
    parser.add_argument("--output-dir", default=None,
                        help="Override the dir to scan for checkpoint-* (default: config output_dir).")
    parser.add_argument("--greedy", action="store_true",
                        help="Use greedy decoding instead of eval-like sampling.")
    parser.add_argument("--no-base", action="store_true",
                        help="Skip the base-model reference run.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--min-think-tokens", type=int, default=20)
    args = parser.parse_args()

    logger = make_logger("check_all_epochs_cot")
    cfg = load_config(args.config)
    check_credentials(logger)

    output_dir = args.output_dir or cfg["output_dir"]
    checkpoints = discover_checkpoints(output_dir)
    if not checkpoints:
        raise SystemExit(f"No checkpoint-* dirs under {output_dir}. Train first, or pass --output-dir.")

    targets = []
    if not args.no_base:
        targets.append(("base", "base"))
    for i, ck in enumerate(checkpoints, start=1):
        targets.append((f"epoch{i}", ck))

    tokenizer = load_tokenizer_for_gen(cfg["base_model"])
    do_sample = not args.greedy
    logger.info("Decoding: %s | targets: %s",
                "sampling (temp 0.6, top_p 0.95)" if do_sample else "greedy",
                [t[0] for t in targets])

    summary = []
    for label, target in targets:
        logger.info("########## %s : %s ##########", label, target)
        res = generation_check(
            base_model_id=cfg["base_model"],
            adapter_dir=target,
            tokenizer=tokenizer,
            logger=logger,
            max_new_tokens=args.max_new_tokens,
            min_think_tokens=args.min_think_tokens,
            do_sample=do_sample,
        )
        per = res["per_prompt"]
        avg_think = sum(p["n_think_tokens"] for p in per) / max(len(per), 1)
        n_with_think = sum(1 for p in per if p["n_think_tokens"] >= args.min_think_tokens)
        summary.append((label, target, avg_think, n_with_think, len(per), res["all_ok"]))

    logger.info("=" * 78)
    logger.info("CoT SUMMARY  (avg think tokens / prompts-with-CoT / total)")
    logger.info("%-10s %-12s %-10s %s", "target", "avg_think", "with_CoT", "all_ok")
    for label, target, avg_think, n_ok, n_total, all_ok in summary:
        logger.info("%-10s %-12.0f %-10s %s", label, avg_think, f"{n_ok}/{n_total}", all_ok)
    logger.info("=" * 78)
    logger.info("Pick the latest epoch whose think blocks are substantial; if only "
                "earlier epochs reason, evaluate that earlier checkpoint (spec §9.1).")


if __name__ == "__main__":
    main()
