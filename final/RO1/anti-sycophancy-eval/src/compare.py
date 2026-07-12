"""
Paired base-vs-trained statistics (spec §4.3 rule 1).

Both models answered IDENTICAL prompts with IDENTICAL seeds (the frozen indexes +
fixed seed scheme guarantee it; the caller asserts the index checksums match), so
every metric delta is a per-question paired comparison:

  - bootstrap CI over questions (percentile, seeded) on each metric's mean delta —
    the primary statistic; valid for rate-valued outcomes at any n_samples.
  - exact McNemar on the discordant pairs of binarized outcomes (question "flipped"
    iff its rate > 0) — the classic paired test; with n_samples=1 the binarization
    is the raw outcome.

Each metric is compared on the questions where it is DEFINED for both models
(e.g. M1: conditioned + a committed wrong-arm answer in both) — the per-model
headline numbers in metrics.json keep their own (per-model) conditioning.
"""

import random


def paired_questions(base_out: dict, trained_out: dict, field: str,
                     require_conditioned: bool = False) -> list:
    """[(qid, base_rate, trained_rate)] where `field` is defined for both models."""
    rows = []
    for qid in sorted(base_out.keys() & trained_out.keys()):
        b, t = base_out[qid], trained_out[qid]
        if require_conditioned and not (b.get("conditioned") and t.get("conditioned")):
            continue
        if b.get(field) is None or t.get(field) is None:
            continue
        rows.append((qid, b[field], t[field]))
    return rows


def bootstrap_delta_ci(rows: list, resamples: int, seed: int) -> dict:
    """Percentile bootstrap CI over questions for mean(trained - base)."""
    deltas = [t - b for _, b, t in rows]
    n = len(deltas)
    if n == 0:
        return {"n": 0, "mean_delta": None, "ci_low": None, "ci_high": None}
    mean_delta = sum(deltas) / n
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return {
        "n": n,
        "mean_delta": mean_delta,
        "ci_low": means[int(0.025 * resamples)],
        "ci_high": means[min(int(0.975 * resamples), resamples - 1)],
        "resamples": resamples,
    }


def mcnemar_exact(rows: list) -> dict:
    """Exact McNemar on binarized outcomes (rate > 0 = the event occurred)."""
    b = sum(1 for _, base, trained in rows if base > 0 and trained == 0)
    c = sum(1 for _, base, trained in rows if base == 0 and trained > 0)
    n_discordant = b + c
    if n_discordant == 0:
        return {"b_base_only": b, "c_trained_only": c, "p_value": 1.0,
                "note": "no discordant pairs"}
    from scipy.stats import binomtest
    p = binomtest(b, n_discordant, 0.5, alternative="two-sided").pvalue
    return {"b_base_only": b, "c_trained_only": c, "n_discordant": n_discordant,
            "p_value": float(p)}


def compare_metric(base_out: dict, trained_out: dict, field: str, metrics_cfg: dict,
                   require_conditioned: bool = False) -> dict:
    rows = paired_questions(base_out, trained_out, field, require_conditioned)
    base_mean = (sum(b for _, b, _ in rows) / len(rows)) if rows else None
    trained_mean = (sum(t for _, _, t in rows) / len(rows)) if rows else None
    return {
        "field": field,
        "n_paired_questions": len(rows),
        "base_mean": base_mean,
        "trained_mean": trained_mean,
        "bootstrap": bootstrap_delta_ci(rows, metrics_cfg.get("bootstrap_resamples", 10000),
                                        metrics_cfg.get("bootstrap_seed", 0)),
        "mcnemar": mcnemar_exact(rows),
    }
