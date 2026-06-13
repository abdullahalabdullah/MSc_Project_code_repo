"""
RO2 MVP — DPO sanity check (60-example dry run).

A SEPARATE entry point from `train_dpo.py` (not a flag bolted onto it), driven
by `configs/harmlessness_sanity.yaml` overlaid on `configs/harmlessness.yaml`.
It reuses the importable functions in `train_dpo.py` so the sanity code path
exercises the exact same data-loading / formatting / training / selection /
push logic as the real run — just on a tiny slice of data and capped at 50
optimizer steps.

Purpose: catch formatting / OOM / config / environment (Unsloth-compat) errors
cheaply, BEFORE committing the GPU allocation to a 3-6 hour run. Because the
environment (trl==0.8.6, transformers==4.44.0, peft==0.12.0, unsloth) is not yet
proven on the cluster, this script is the gate: if the pinned stack is broken,
it fails here, loudly.

Run:
    python sanity_check_dpo.py

It pushes the selected adapter to the THROWAWAY repo (`sanity_hub_model_id`) to
validate auth + push end-to-end without touching the real `hub_model_id`.

Exit code is non-zero if any programmatic check fails.
"""

import math
import sys

import train_dpo as T  # reuse all shared functions (no duplicated logic)


BASE_CONFIG = "configs/harmlessness.yaml"
SANITY_OVERRIDE = "configs/harmlessness_sanity.yaml"


def _collect_eval_points(log_history: list) -> list:
    """Return [(step, acc, loss), ...] for every eval event, in step order."""
    points = []
    for entry in log_history:
        if "eval_rewards/accuracies" in entry and "eval_loss" in entry:
            points.append(
                (int(entry["step"]),
                 float(entry["eval_rewards/accuracies"]),
                 float(entry["eval_loss"]))
            )
    points.sort(key=lambda x: x[0])
    return points


def _collect_train_losses(log_history: list) -> list:
    """Return [(step, loss), ...] for every training-loss log entry."""
    losses = []
    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            losses.append((int(entry.get("step", -1)), float(entry["loss"])))
    losses.sort(key=lambda x: x[0])
    return losses


def run_checks(summary: dict, logger) -> bool:
    """Assert the programmatic subset of the spec's 8 checks. Returns all_passed."""
    log_history = summary["log_history"]
    eval_points = _collect_eval_points(log_history)
    train_losses = _collect_train_losses(log_history)

    results = []  # (name, passed: bool|None, detail)  -- None == manual

    # Check 1 — formatting triples: printed during run_training (inspect_formatting).
    results.append((
        "1. Chat-template formatting (5-10 triples)",
        None,
        "Printed above by run_training; eyeball that prompt ends at the "
        "assistant boundary and chosen/rejected have no leaked role markers.",
    ))

    # Check 2 — prompt-length distribution: printed; assert it's not catastrophic.
    pls = summary["prompt_length_stats"]
    frac_over = pls["frac_over_max_prompt_length"]
    results.append((
        "2. Prompt-length distribution",
        frac_over <= 0.50,  # sanity-only guard; >5% already warned in the run
        f"{100*frac_over:.1f}% of prompts exceed max_prompt_length="
        f"{pls['max_prompt_length']} (p95={pls['p95']:.0f}, max={pls['max']}).",
    ))

    # Check 3 — training loss curve: print for eyeballing; assert finite, not flat-zero.
    if train_losses:
        loss_vals = [l for _, l in train_losses]
        all_finite = all(math.isfinite(l) for l in loss_vals)
        all_zero = all(abs(l) < 1e-12 for l in loss_vals)
        logger.info("Training loss curve (step: loss):")
        for step, loss in train_losses:
            logger.info("    %6d : %.5f", step, loss)
        results.append((
            "3. Training loss finite and not flat-zero",
            all_finite and not all_zero,
            f"{len(loss_vals)} logged losses; first={loss_vals[0]:.4f}, "
            f"last={loss_vals[-1]:.4f}, finite={all_finite}, all_zero={all_zero}. "
            "(Visually confirm it trends down / is noisy, not exploding.)",
        ))
    else:
        results.append(("3. Training loss curve", False, "No training-loss entries logged."))

    # Check 4 — intermediate eval fired >=2 times, each with loss + accuracy.
    results.append((
        "4. >=2 intermediate eval events (loss + accuracy)",
        len(eval_points) >= 2,
        f"{len(eval_points)} eval events: "
        f"{[(s, round(a,3), round(l,3)) for s, a, l in eval_points]}",
    ))

    # Check 5 — eval_rewards/accuracies in [0,1], not NaN, not identical across evals.
    if eval_points:
        accs = [a for _, a, _ in eval_points]
        losses = [l for _, _, l in eval_points]
        in_range = all(math.isfinite(a) and 0.0 <= a <= 1.0 for a in accs)
        not_identical = len(set(round(a, 6) for a in accs)) > 1 if len(accs) > 1 else True
        losses_finite = all(math.isfinite(l) for l in losses)
        results.append((
            "5a. eval accuracies in [0,1], finite",
            in_range,
            f"accuracies={[round(a,4) for a in accs]}",
        ))
        results.append((
            "5b. eval accuracies not identical across all evals",
            not_identical,
            "Identical accuracies across evals can indicate the model isn't "
            "differentiating chosen/rejected (DPO not backprop'ing into LoRA).",
        ))
        results.append((
            "5c. eval_loss finite",
            losses_finite,
            f"eval_losses={[round(l,4) for l in losses]}",
        ))
    else:
        results.append(("5. eval value sanity", False, "No eval points to check."))

    # Check 6 — checkpoint selection ran and returned a step.
    results.append((
        "6. Checkpoint selection picked a step",
        summary["best_step"] is not None,
        f"best_step={summary['best_step']}, metrics={summary['metrics']}",
    ))

    # Check 7 — GPU peak memory: manual (watch `nvidia-smi` during the run).
    results.append((
        "7. GPU peak memory under ~35-37 GB",
        None,
        "Manual: watch `nvidia-smi` during this run; confirm headroom for the "
        "full run (avg sequence lengths are longer on the full dataset).",
    ))

    # Check 8 — Hub push to the throwaway repo worked end-to-end.
    push = summary["push_result"]
    results.append((
        "8. Adapter push to throwaway repo succeeded",
        bool(push and push.get("pushed")),
        f"push_result={push}",
    ))

    # Report.
    logger.info("=" * 78)
    logger.info("SANITY-CHECK RESULTS")
    logger.info("=" * 78)
    all_passed = True
    for name, passed, detail in results:
        if passed is None:
            status = "MANUAL"
        elif passed:
            status = "PASS"
        else:
            status = "FAIL"
            all_passed = False
        logger.info("[%-6s] %s", status, name)
        logger.info("         %s", detail)
    logger.info("=" * 78)
    if all_passed:
        logger.info(
            "All programmatic checks PASSED. Review the MANUAL items above, then "
            "proceed: `sbatch train_harmlessness.sh`."
        )
    else:
        logger.error(
            "One or more programmatic checks FAILED. Do NOT launch the full run "
            "until these are resolved."
        )
    return all_passed


def main():
    logger = T.make_logger("sanity_check_dpo")
    cfg = T.load_config(BASE_CONFIG, override_path=SANITY_OVERRIDE)
    logger.info("Loaded sanity config: %s overlaid with %s", BASE_CONFIG, SANITY_OVERRIDE)
    logger.info(
        "Sanity run: subsample_size=%d val_size=%d max_steps=%d eval_steps=%d",
        cfg["subsample_size"], cfg["val_size"], cfg["max_steps"], cfg["eval_steps"],
    )

    # Push to the THROWAWAY repo, not the real hub_model_id.
    hub_id = cfg["sanity_hub_model_id"]
    summary = T.run_training(cfg, hub_id, logger)

    all_passed = run_checks(summary, logger)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
