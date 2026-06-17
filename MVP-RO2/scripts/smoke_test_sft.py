"""Minimal SFT smoke test: train a few steps, then inspect the model's outputs.

    python scripts/smoke_test_sft.py

Loads configs/sft_harmlessness.yaml + configs/sft_harmlessness_smoke.yaml, trains
on a small slice (~125 steps, a few minutes, NO Hub push), prints the label-masking
regions, and then runs the thinking-mode generation check on the resulting
checkpoint so you can SEE whether chain-of-thought is preserved after the masking
fix. For a base-model reference, run separately:

    python scripts/03_sanity_generation.py --config configs/sft_harmlessness.yaml --checkpoint base
"""

import argparse

import _bootstrap  # noqa: F401
from sft.config import load_config
from sft.generate import generation_check, load_tokenizer_for_gen
from sft.logging_utils import make_logger
from sft.training import run_training

BASE_CONFIG = "configs/sft_harmlessness.yaml"
SMOKE_OVERRIDE = "configs/sft_harmlessness_smoke.yaml"


def main():
    parser = argparse.ArgumentParser(description="RO2 SFT — minimal smoke test + output inspection.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--min-think-tokens", type=int, default=20)
    args = parser.parse_args()

    logger = make_logger("smoke_test_sft")
    cfg = load_config(BASE_CONFIG, override_path=SMOKE_OVERRIDE)
    logger.info("Smoke config: %s + %s | max_train_samples=%s",
                BASE_CONFIG, SMOKE_OVERRIDE, cfg.get("max_train_samples"))

    summary = run_training(cfg, logger)
    if not summary["checkpoint_dirs"]:
        raise SystemExit("No checkpoint was saved — cannot inspect outputs.")

    ckpt = summary["checkpoint_dirs"][-1]
    logger.info("Running thinking-mode generation check on %s ...", ckpt)
    tokenizer = load_tokenizer_for_gen(cfg["base_model"])
    result = generation_check(
        base_model_id=cfg["base_model"],
        adapter_dir=ckpt,
        tokenizer=tokenizer,
        logger=logger,
        max_new_tokens=args.max_new_tokens,
        min_think_tokens=args.min_think_tokens,
    )
    logger.info("Smoke test done. CoT preserved: %s (eyeball the generations above).",
                result["all_ok"])


if __name__ == "__main__":
    main()
