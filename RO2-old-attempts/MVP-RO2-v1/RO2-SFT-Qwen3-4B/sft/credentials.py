"""Credential loading: read .env, fail fast if HF_TOKEN is missing, hf login."""

import logging
import os

from dotenv import load_dotenv
from huggingface_hub import login


def check_credentials(logger: logging.Logger) -> str:
    """Load .env and authenticate with the Hub. Returns the HF token.

    Fails fast (before any GPU work or downloads) if HF_TOKEN is missing.
    OPENAI_API_KEY is only needed by the downstream RO1 eval/monitor step, so its
    absence is a warning, not an error.
    """
    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN is not set. Create a .env file in the MVP-RO2 root with "
            "HF_TOKEN=hf_... (WRITE access to the target Hub repos, and access to "
            "the gated PKU-SafeRLHF dataset). Aborting before any download/GPU work."
        )
    if not os.environ.get("OPENAI_API_KEY"):
        logger.warning(
            "OPENAI_API_KEY is not set. Not needed for SFT training, but the "
            "downstream RO1 eval/monitor step will require it later."
        )
    login(token=hf_token)
    logger.info("Hugging Face authentication OK (HF_TOKEN loaded from .env).")
    return hf_token
