"""Stage I (10) — sham (shuffled-preference) control dataset (CPU).

Copies pairs/anti_syc_dpo.jsonl and, for each pair independently, swaps
chosen/rejected with probability 0.5 (seed 69, file order). Identical prompts,
identical response pool, destroyed preference signal (spec §9).

Input : pairs/anti_syc_dpo.jsonl
Output: pairs/anti_syc_dpo_sham.jsonl (adds "sham_swapped": bool)
"""
import argparse
import random

import config
import env_setup
from utils import (check_stage_cache, read_jsonl, update_diagnostics, upstream_hash,
                   write_jsonl, write_stage_marker)

STAGE = "10_sham"


def main():
    ap = argparse.ArgumentParser(description="Stage I — sham control (CPU).")
    ap.add_argument("--force", action="store_true", help="Re-run even if cached.")
    args = ap.parse_args()

    env_setup.setup_env()
    config.ensure_output_dirs()

    params = {"upstream": upstream_hash("09_pairs"), "seed_sham": config.SEED_SHAM}
    if check_stage_cache(STAGE, params, [config.SHAM_PAIRS_PATH], force=args.force):
        return

    pairs = read_jsonl(config.PAIRS_PATH)
    rng = random.Random(config.SEED_SHAM)
    sham = []
    n_swapped = 0
    for p in pairs:
        swap = rng.random() < 0.5
        record = dict(p)
        if swap:
            record["chosen"], record["rejected"] = p["rejected"], p["chosen"]
            n_swapped += 1
        record["sham_swapped"] = swap
        sham.append(record)

    write_jsonl(config.SHAM_PAIRS_PATH, sham)
    print(f"Sham dataset: {len(sham)} pairs, {n_swapped} swapped "
          f"({100.0 * n_swapped / max(len(sham), 1):.1f}%) -> {config.SHAM_PAIRS_PATH}")

    update_diagnostics(config.DIAGNOSTICS_PATH, "sham", {
        "n_pairs": len(sham),
        "n_swapped": n_swapped,
        "seed_sham": config.SEED_SHAM,
    })
    write_stage_marker(STAGE, params, {"pairs": len(sham), "swapped": n_swapped})


if __name__ == "__main__":
    main()
