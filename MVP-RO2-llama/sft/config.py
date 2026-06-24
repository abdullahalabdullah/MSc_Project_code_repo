"""YAML config loading with optional deep-merge of an override file.

The sanity-check config holds only the keys that differ from the full config,
so `load_config(full, override)` deep-merges them. Helpers derive the per-epoch
and sanity Hub repo ids from the config so repo naming lives in one place.
"""

from typing import Optional

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` onto a copy of `base`."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(config_path: str, override_path: Optional[str] = None) -> dict:
    """Load a YAML config, optionally deep-merging an override file on top."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if override_path is not None:
        with open(override_path, "r", encoding="utf-8") as f:
            override = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, override)
    return cfg


def epoch_repo_id(cfg: dict, epoch_idx: int) -> str:
    """`<hf_username>/<repo_prefix>-epoch<idx>` for the given 1-based epoch."""
    return f"{cfg['hf_username']}/{cfg['repo_prefix']}-epoch{epoch_idx}"


def sanity_repo_id(cfg: dict) -> str:
    """`<hf_username>/<sanity_repo>` (throwaway repo used by the sanity gate)."""
    return f"{cfg['hf_username']}/{cfg['sanity_repo']}"
