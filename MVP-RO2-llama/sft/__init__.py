"""RO2 MVP — harmlessness SFT (Llama-3.2-3B-Instruct, LoRA) training package.

All logic lives here as importable functions; the entry points in ../scripts
are thin argparse wrappers. The sanity gate reuses the exact same code path as
the real run, just on a tiny slice of data.
"""

__all__ = [
    "config",
    "logging_utils",
    "credentials",
    "data",
    "model",
    "training",
    "push",
]
