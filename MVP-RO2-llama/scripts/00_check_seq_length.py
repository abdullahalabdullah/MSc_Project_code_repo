"""Pre-training preflight: report the token-length distribution of the formatted
training examples and recommend `max_length`.

    python scripts/00_check_seq_length.py --config configs/sft_harmlessness.yaml

Set `max_length` in the config to the recommended value before training. No GPU
needed (tokenizer only). hh-rlhf is public; the tokenizer download for the gated
Llama model still needs a token whose account accepted the Llama-3.2 license.
"""

import argparse

import _bootstrap  # noqa: F401  (puts repo root on sys.path)
from sft import data as data_mod
from sft.config import load_config
from sft.credentials import check_credentials
from sft.logging_utils import make_logger
from sft.model import load_tokenizer


def main():
    parser = argparse.ArgumentParser(description="RO2 SFT — sequence-length preflight.")
    parser.add_argument("--config", default="configs/sft_harmlessness.yaml")
    args = parser.parse_args()

    logger = make_logger("seq_length")
    cfg = load_config(args.config)
    logger.info("Loaded config from %s", args.config)

    # Llama-3.2 is a gated MODEL -> we still need HF auth to download the tokenizer.
    check_credentials(logger)

    tokenizer = load_tokenizer(cfg)
    pairs = data_mod.load_hh_harmless(cfg, logger)
    report = data_mod.sequence_length_report(pairs, tokenizer, logger=logger)

    rec = report["recommended"]
    cur = cfg["max_length"]
    if rec is not None and rec != cur:
        logger.info(
            "Config max_length=%d; recommended>=%d. Update %s if you want to change it.",
            cur, rec, args.config,
        )
    else:
        logger.info("Config max_length=%d looks fine.", cur)


if __name__ == "__main__":
    main()
