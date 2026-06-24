"""Main SFT training entry point.

    python scripts/01_train_sft.py --config configs/sft_harmlessness.yaml

Produces one LoRA adapter checkpoint per epoch under cfg.output_dir. Run the
preflight (00) and the sanity gate (sanity_check_sft.py) first.
"""

import argparse

import _bootstrap  # noqa: F401
from sft.config import load_config
from sft.logging_utils import make_logger
from sft.training import run_training


def main():
    parser = argparse.ArgumentParser(description="RO2 MVP — Llama-3.2-3B-Instruct harmlessness SFT.")
    parser.add_argument("--config", default="configs/sft_harmlessness.yaml")
    args = parser.parse_args()

    logger = make_logger("train_sft")
    cfg = load_config(args.config)
    logger.info("Loaded config from %s", args.config)

    summary = run_training(cfg, logger)

    logger.info(
        "DONE. train=%d eval=%d checkpoints=%d -> %s",
        summary["train_size"], summary["eval_size"],
        len(summary["checkpoint_dirs"]), summary["checkpoint_dirs"],
    )
    logger.info("Next: push with scripts/02_push_adapters.py --config %s", args.config)


if __name__ == "__main__":
    main()
