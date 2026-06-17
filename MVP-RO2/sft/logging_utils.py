"""Simple stdout logger so the RunPod/terminal log captures everything."""

import logging
import sys


def make_logger(name: str = "sft") -> logging.Logger:
    """Return a process-wide logger that streams to stdout with timestamps."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s  %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
