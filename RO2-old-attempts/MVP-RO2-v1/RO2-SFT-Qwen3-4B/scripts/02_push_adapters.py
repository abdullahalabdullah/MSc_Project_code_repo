"""Push all per-epoch LoRA adapters to private HF Hub repos (spec §8.2).

    python scripts/02_push_adapters.py --config configs/sft_harmlessness.yaml

Creates er-abd/<repo_prefix>-epoch{1,2,3} (private) from the checkpoint dirs under
cfg.output_dir. Adapter-only upload (no model loading / merging).
"""

import argparse
import os

import _bootstrap  # noqa: F401
from sft.config import load_config
from sft.credentials import check_credentials
from sft.logging_utils import make_logger
from sft.push import push_all_checkpoints


def main():
    parser = argparse.ArgumentParser(description="RO2 SFT — push adapter checkpoints.")
    parser.add_argument("--config", default="configs/sft_harmlessness.yaml")
    args = parser.parse_args()

    logger = make_logger("push_adapters")
    cfg = load_config(args.config)
    logger.info("Loaded config from %s", args.config)

    check_credentials(logger)

    # output_dir holds the tokenizer saved at the end of training; copy it into
    # each repo so the adapter repos are self-contained at eval time.
    tok_dir = cfg["output_dir"] if os.path.isdir(cfg["output_dir"]) else None
    results = push_all_checkpoints(cfg, logger, tokenizer_dir=tok_dir)

    if not all(r["pushed"] for r in results):
        raise SystemExit("One or more pushes FAILED — see the log above.")
    logger.info("All adapter checkpoints pushed. Share the repo IDs with the evaluator.")


if __name__ == "__main__":
    main()
