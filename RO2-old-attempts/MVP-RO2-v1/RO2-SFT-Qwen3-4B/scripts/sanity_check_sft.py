"""RO2 SFT sanity check — the environment GATE (run FIRST on the GPU box).

    python scripts/sanity_check_sft.py

Loads configs/sft_harmlessness.yaml overlaid with configs/sft_harmlessness_sanity.yaml,
runs the SAME run_training() code path on a tiny slice (~64 examples, ~20 steps),
pushes the resulting adapter to a THROWAWAY repo, then asserts the programmatic
subset of the spec's checks (label masking, loss finite & non-zero, checkpoint
exists with adapter weights, push works). Prints PASS/FAIL/MANUAL and exits
non-zero if anything fails — so a broken environment fails here cheaply, not
hours into the real run.
"""

import math
import os
import sys

import _bootstrap  # noqa: F401
from sft.config import load_config, sanity_repo_id
from sft.logging_utils import make_logger
from sft.push import push_checkpoint
from sft.training import run_training

BASE_CONFIG = "configs/sft_harmlessness.yaml"
SANITY_OVERRIDE = "configs/sft_harmlessness_sanity.yaml"


def _train_losses(log_history):
    return [(int(e.get("step", -1)), float(e["loss"]))
            for e in log_history if "loss" in e and "eval_loss" not in e]


def _eval_events(log_history):
    return [(int(e.get("step", -1)), float(e["eval_loss"]))
            for e in log_history if "eval_loss" in e]


def run_checks(summary, push_result, logger) -> bool:
    results = []  # (name, passed|None, detail)

    # 1. Label masking — asserted inside run_training (verify_masking). Reaching
    #    here means it passed; flag as MANUAL re-eyeball of the printed regions.
    results.append((
        "1. Label masking (prompt -> -100, response supervised)",
        True,
        "verify_masking() asserted during the run; eyeball the printed "
        "MASKED/SUPERVISED regions above.",
    ))

    # 2. Training loss finite and not flat-zero, and a downward-ish trend.
    losses = _train_losses(summary["log_history"])
    if losses:
        vals = [l for _, l in losses]
        finite = all(math.isfinite(v) for v in vals)
        all_zero = all(abs(v) < 1e-12 for v in vals)
        logger.info("Training loss curve (step: loss):")
        for s, l in losses:
            logger.info("    %6d : %.5f", s, l)
        results.append((
            "2. Training loss finite & not flat-zero",
            finite and not all_zero,
            f"{len(vals)} losses; first={vals[0]:.4f}, last={vals[-1]:.4f}, "
            f"finite={finite}, all_zero={all_zero}.",
        ))
    else:
        results.append(("2. Training loss", False, "No training-loss entries logged."))

    # 3. At least one eval event with a finite eval_loss.
    evals = _eval_events(summary["log_history"])
    eval_ok = len(evals) >= 1 and all(math.isfinite(l) for _, l in evals)
    results.append((
        "3. >=1 eval event with finite eval_loss",
        eval_ok,
        f"{len(evals)} eval events: {[(s, round(l, 3)) for s, l in evals]}",
    ))

    # 4. Checkpoint dir exists and contains adapter weights.
    ckpts = summary["checkpoint_dirs"]
    has_adapter = bool(ckpts) and os.path.isfile(
        os.path.join(ckpts[-1], "adapter_model.safetensors")
    )
    results.append((
        "4. Checkpoint dir has adapter_model.safetensors",
        has_adapter,
        f"checkpoints={ckpts}",
    ))

    # 5. Adapter push to the throwaway repo succeeded.
    results.append((
        "5. Adapter push to throwaway repo",
        bool(push_result and push_result.get("pushed")),
        f"push_result={push_result}",
    ))

    # 6. GPU peak memory — manual.
    results.append((
        "6. GPU peak memory headroom",
        None,
        "Manual: watch `nvidia-smi`; the full run has longer sequences than the "
        "sanity slice, so leave headroom.",
    ))

    logger.info("=" * 78)
    logger.info("SANITY-CHECK RESULTS")
    logger.info("=" * 78)
    all_passed = True
    for name, passed, detail in results:
        status = "MANUAL" if passed is None else ("PASS" if passed else "FAIL")
        if passed is False:
            all_passed = False
        logger.info("[%-6s] %s", status, name)
        logger.info("         %s", detail)
    logger.info("=" * 78)
    if all_passed:
        logger.info("All programmatic checks PASSED. Proceed to the full run "
                    "(python scripts/01_train_sft.py).")
    else:
        logger.error("One or more checks FAILED. Do NOT launch the full run yet.")
    return all_passed


def main():
    logger = make_logger("sanity_check_sft")
    cfg = load_config(BASE_CONFIG, override_path=SANITY_OVERRIDE)
    logger.info("Sanity config: %s + %s | max_train_samples=%s max_steps=%s",
                BASE_CONFIG, SANITY_OVERRIDE,
                cfg.get("max_train_samples"), cfg.get("max_steps"))

    summary = run_training(cfg, logger)

    # Push the last checkpoint to the THROWAWAY repo to validate the push path.
    push_result = None
    if summary["checkpoint_dirs"]:
        push_result = push_checkpoint(
            summary["checkpoint_dirs"][-1], sanity_repo_id(cfg), cfg,
            epoch_idx=1, logger=logger,
        )

    sys.exit(0 if run_checks(summary, push_result, logger) else 1)


if __name__ == "__main__":
    main()
