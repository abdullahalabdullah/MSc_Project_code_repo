"""Stage 06 — export the final preference pairs as a training-ready DPO dataset.

This is the handoff to the (external) DPO trainer. It converts the module's
`final_pairs.joblib` (a list[dict] with rich metadata) into the standard TRL
preference schema and saves it as a HF DatasetDict:

    DatasetDict({
        "train": Dataset({prompt, chosen, rejected}),
        "valid": Dataset({prompt, chosen, rejected}),
    })
    -> saved via save_to_disk(config.DPO_DATASET_DIR)

- `prompt`   : the harmful WildJailbreak prompt (the trainer applies its own chat template).
- `chosen`   : the safest response; `rejected`: the most harmful response.
- By default `chosen`/`rejected` are the FULL responses (<think>…</think> + answer), matching
  what the model generates (config.EXPORT_FIELD="full"). Set EXPORT_FIELD="answer" to export
  only the visible post-</think> answer.

The split is deterministic (config.SEED) with config.VALID_FRACTION held out as "valid",
drawn from the FINAL pairs (i.e. after all filtering). By default the holdout is a plain
seeded RANDOM sample — representative of the training distribution, the right basis for
checkpoint selection. Set STRATIFY_VALID=true to instead target a 50/50
vanilla/adversarial mix (best-effort — if one subset runs out, the shortfall is filled
from the other). The realised train/valid mix is reported either way.

Runs on CPU. Input: config.FINAL_PAIRS_PATH. Output: config.DPO_DATASET_DIR (+ diagnostics).
"""
import random
from collections import Counter

import joblib
from datasets import Dataset, DatasetDict

import config
from utils import set_seed, update_diagnostics


def _pick_fields(pair):
    """Map one final_pairs dict to the {prompt, chosen, rejected} training row."""
    if config.EXPORT_FIELD == "answer":
        chosen, rejected = pair["chosen_answer"], pair["rejected_answer"]
    else:  # "full" (default)
        chosen, rejected = pair["chosen"], pair["rejected"]
    return {"prompt": pair["prompt"], "chosen": chosen, "rejected": rejected}


def _split_train_valid(pairs, valid_fraction, seed, stratify=False):
    """Deterministic train/valid split over the final (post-filtering) pairs.

    Default (stratify=False): a plain seeded random holdout — its subset mix follows the
    training distribution, so eval metrics estimate in-distribution generalization.

    stratify=True: the holdout targets a 50/50 vanilla/adversarial mix — half its size is
    drawn from each subset (seeded). If one subset has fewer pairs than its target, the
    shortfall is filled from the remaining pool so the holdout size is preserved
    (best-effort mix — the realised mix is printed and recorded in diagnostics by the caller).

    Guarantees a non-empty train split; holds out at least one valid example when there is
    more than one pair (so downstream eval always has something), and never lets valid take
    the whole set.
    """
    pairs = list(pairs)
    n = len(pairs)
    if n <= 1:
        return pairs, []                      # too few to split — everything trains
    n_valid = round(n * valid_fraction)
    n_valid = max(1, min(n_valid, n - 1))     # 1 <= n_valid <= n-1

    rng = random.Random(seed)
    if not stratify:
        rng.shuffle(pairs)
        return pairs[n_valid:], pairs[:n_valid]

    vanilla = [p for p in pairs if p.get("subset") == "vanilla"]
    adversarial = [p for p in pairs if p.get("subset") == "adversarial"]
    other = [p for p in pairs if p.get("subset") not in ("vanilla", "adversarial")]
    rng.shuffle(vanilla)
    rng.shuffle(adversarial)

    # Target: half the holdout from each subset (odd n_valid: adversarial gets the extra).
    target_van = n_valid // 2
    take_van = min(target_van, len(vanilla))
    take_adv = min(n_valid - target_van, len(adversarial))
    valid = vanilla[:take_van] + adversarial[:take_adv]

    # Everything not taken; fills any shortfall (a subset ran out), then becomes train.
    remaining = vanilla[take_van:] + adversarial[take_adv:] + other
    rng.shuffle(remaining)
    shortfall = n_valid - len(valid)
    valid = valid + remaining[:shortfall]
    train = remaining[shortfall:]
    rng.shuffle(train)
    return train, valid


def main():
    config.ensure_output_dir()
    set_seed(config.SEED)

    if config.EXPORT_FIELD not in ("full", "answer"):
        raise ValueError(f"EXPORT_FIELD must be 'full' or 'answer', got {config.EXPORT_FIELD!r}")

    final_pairs = joblib.load(config.FINAL_PAIRS_PATH)
    print(f"Loaded {len(final_pairs)} final pairs from {config.FINAL_PAIRS_PATH}")
    if not final_pairs:
        raise SystemExit("No final pairs to export — run stages 01-05 first (final_pairs is empty).")

    # Split FIRST (on the pairs, which carry `subset` so the mix can be reported and,
    # optionally, stratified on), then map to the 3-column training rows.
    train_pairs, valid_pairs = _split_train_valid(
        final_pairs, config.VALID_FRACTION, config.SEED, stratify=config.STRATIFY_VALID)
    train_rows = [_pick_fields(p) for p in train_pairs]
    valid_rows = [_pick_fields(p) for p in valid_pairs]
    train_mix = Counter(p.get("subset") for p in train_pairs)
    valid_mix = Counter(p.get("subset") for p in valid_pairs)
    if config.STRATIFY_VALID and valid_rows and valid_mix.get("vanilla") != valid_mix.get("adversarial"):
        print(f"WARNING: valid split is not exactly 50/50 "
              f"(vanilla={valid_mix.get('vanilla', 0)}, adversarial={valid_mix.get('adversarial', 0)}) "
              f"— one subset had too few pairs; shortfall filled from the other.")

    # train is always non-empty (split guarantees it); reuse its schema for valid so an
    # empty valid split (tiny datasets / smoke runs) still saves instead of failing schema inference.
    train_ds = Dataset.from_list(train_rows)
    if valid_rows:
        valid_ds = Dataset.from_list(valid_rows)
    else:
        valid_ds = Dataset.from_dict(
            {c: [] for c in train_ds.column_names}, features=train_ds.features
        )
    dataset = DatasetDict({"train": train_ds, "valid": valid_ds})
    dataset.save_to_disk(config.DPO_DATASET_DIR)

    update_diagnostics(config.DIAGNOSTICS_PATH, "export", {
        "n_pairs": len(final_pairs),
        "n_train": len(train_rows),
        "n_valid": len(valid_rows),
        "valid_fraction": config.VALID_FRACTION,
        "valid_sampling": ("stratified 50/50 vanilla-adversarial (best-effort)"
                           if config.STRATIFY_VALID else "random (seeded)"),
        "train_mix": dict(train_mix),
        "valid_mix": dict(valid_mix),
        "export_field": config.EXPORT_FIELD,
        "columns": ["prompt", "chosen", "rejected"],
        "path": config.DPO_DATASET_DIR,
    })

    print(f"Exported DPO dataset -> {config.DPO_DATASET_DIR}")
    print(f"  splits: train={len(train_rows)}  valid={len(valid_rows)}  "
          f"(field={config.EXPORT_FIELD})")
    print(f"  train mix: " + ", ".join(f"{s}={train_mix.get(s, 0)}" for s in ("vanilla", "adversarial")))
    print(f"  valid mix: " + ", ".join(f"{s}={valid_mix.get(s, 0)}" for s in ("vanilla", "adversarial"))
          + ("  (stratified 50/50, best-effort)" if config.STRATIFY_VALID else "  (random, seeded)"))
    print("  columns: prompt, chosen, rejected")
    print(f"  load with: datasets.load_from_disk({config.DPO_DATASET_DIR!r})")


if __name__ == "__main__":
    main()
