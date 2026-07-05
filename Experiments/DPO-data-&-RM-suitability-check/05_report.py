"""Stage 05 — filter-3, final pairs, and all reports.

filter-3: keep a prompt only if guard_label(chosen) == Safe AND guard_label(rejected) == Unsafe.

chosen/rejected are saved as the FULL response (<think>…</think> + answer) so the CoT is
retained for DPO training; the RM/classifier still only ever saw the visible answer (stages
03/04). The visible answer is stashed alongside as chosen_answer/rejected_answer.

Reports (printed + written to config.REPORT_PATH):
  - unique-prompt count, filter-1 drop %, filter-3 drop %, optional filter-4, survival rate
  - per-prompt cost-spread (max - min) distribution + histogram (config.COST_SPREAD_FIG_PATH)
  - length / verbosity confound: chosen vs rejected length (full-response tokens + answer
    words), % of pairs where rejected is longer, and Pearson(cost margin, answer-length Δ)
  - 10 random surviving pairs -> config.SAMPLE_PAIRS_PATH
  - sign-convention sanity check: mean Beaver cost of Unsafe vs Safe answers (Unsafe should be higher)

Runs on CPU. Output: config.FINAL_PAIRS_PATH + report artifacts.
"""
import csv
import json
import random

import joblib
import numpy as np

import config
from utils import set_seed


def main():
    config.ensure_output_dir()
    set_seed(config.SEED)

    survivors = joblib.load(config.SCORED_PATH)
    n_enter_stage4 = len(survivors)

    # SAMPLE is the end-to-end denominator; filter-1 dropped (SAMPLE - survivors).
    n_dropped_f1 = config.SAMPLE - n_enter_stage4

    # ── filter-3 + build final pairs ───────────────────────────────────────
    final_pairs = []
    n_dropped_f3 = 0
    for per_prompt in survivors:
        meta = per_prompt[0]["_pair_meta"]
        ch = per_prompt[meta["chosen_ix"]]
        rj = per_prompt[meta["rejected_ix"]]
        if ch["guard_label"] == "Safe" and rj["guard_label"] == "Unsafe":
            # The RM (Beaver) and classifier (Qwen3Guard) were applied to the VISIBLE answer
            # only (see stages 03/04, which read rd["answer"]). But the DPO pair itself keeps
            # the FULL response (<think>…</think> + answer) — the CoT is retained for training,
            # not discarded. We also stash the visible answer that was actually scored/labeled.
            ch_answer, rj_answer = ch["answer"], rj["answer"]
            final_pairs.append({
                "prompt": ch["prompt"],
                "chosen": ch["full_response"],       # full response (CoT + answer) — used for DPO
                "rejected": rj["full_response"],
                "chosen_answer": ch_answer,          # visible answer that was scored/labeled
                "rejected_answer": rj_answer,
                "chosen_cost": ch["cost"],
                "rejected_cost": rj["cost"],
                "margin": rj["cost"] - ch["cost"],   # cost margin (rejected - chosen), always >= 0
                "chosen_label": ch["guard_label"],
                "rejected_label": rj["guard_label"],
                # length bookkeeping for the verbosity-confound report
                "chosen_full_tokens": ch["num_tokens"],      # full-response tokens (CoT + answer)
                "rejected_full_tokens": rj["num_tokens"],
                "chosen_answer_words": len(ch_answer.split()),   # visible-answer words
                "rejected_answer_words": len(rj_answer.split()),
            })
        else:
            n_dropped_f3 += 1

    pct_f3 = 100.0 * n_dropped_f3 / max(n_enter_stage4, 1)
    n_after_f3 = len(final_pairs)

    # ── filter-4 (OPTIONAL): minimum cost margin ───────────────────────────
    # Only applied when config.MIN_MARGIN is set (MIN_MARGIN env var non-empty). Off by
    # default (None) — for harmlessness filter-3 already gates quality; this is here mainly
    # for the helpfulness pipeline where the reward margin is the only quality signal.
    n_dropped_f4 = 0
    if config.MIN_MARGIN is not None:
        kept = [p for p in final_pairs if p["margin"] >= config.MIN_MARGIN]
        n_dropped_f4 = len(final_pairs) - len(kept)
        final_pairs = kept

    n_final = len(final_pairs)
    survival = 100.0 * n_final / config.SAMPLE

    joblib.dump(final_pairs, config.FINAL_PAIRS_PATH, compress=True)

    # Per-pair margin table (CSV) — one row per pair, easy to analyse / plot.
    with open(config.PAIRS_MARGINS_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_index", "chosen_cost", "rejected_cost", "margin",
                    "chosen_full_tokens", "rejected_full_tokens",
                    "chosen_answer_words", "rejected_answer_words",
                    "chosen_label", "rejected_label", "prompt"])
        for i, p in enumerate(final_pairs):
            w.writerow([i, p["chosen_cost"], p["rejected_cost"], p["margin"],
                        p["chosen_full_tokens"], p["rejected_full_tokens"],
                        p["chosen_answer_words"], p["rejected_answer_words"],
                        p["chosen_label"], p["rejected_label"], p["prompt"]])

    # ── cost-spread distribution (max - min per surviving pair) ─────────────
    spreads = np.array([p["margin"] for p in final_pairs], dtype=float)

    def q(a, x):
        return float(np.percentile(a, x)) if len(a) else float("nan")

    # ── length / verbosity confound (chosen vs rejected) ────────────────────
    # Refusals (chosen) tend to be short and harmful completions (rejected) long, so DPO can
    # learn "shorter = safer" instead of harmlessness. We measure it on BOTH the full-response
    # tokens (what the DPO pair actually contains) and the visible-answer words (where the
    # refusal-is-short effect lives), plus whether the cost margin just tracks answer length.
    if n_final:
        cho_tok = np.array([p["chosen_full_tokens"] for p in final_pairs], dtype=float)
        rej_tok = np.array([p["rejected_full_tokens"] for p in final_pairs], dtype=float)
        cho_w = np.array([p["chosen_answer_words"] for p in final_pairs], dtype=float)
        rej_w = np.array([p["rejected_answer_words"] for p in final_pairs], dtype=float)
        tok_delta = rej_tok - cho_tok
        w_delta = rej_w - cho_w
        pct_rej_longer_tok = 100.0 * float(np.mean(tok_delta > 0))
        pct_rej_longer_w = 100.0 * float(np.mean(w_delta > 0))
        if n_final >= 2 and np.std(w_delta) > 0 and np.std(spreads) > 0:
            r_margin_len = float(np.corrcoef(spreads, w_delta)[0, 1])
        else:
            r_margin_len = float("nan")

    # ── sign-convention sanity check over ALL scored responses ─────────────
    safe_costs, unsafe_costs = [], []
    for per_prompt in survivors:
        for rd in per_prompt:
            if rd.get("invalid") or "cost" not in rd:
                continue
            if rd["guard_label"] == "Safe":
                safe_costs.append(rd["cost"])
            elif rd["guard_label"] == "Unsafe":
                unsafe_costs.append(rd["cost"])
    mean_safe = float(np.mean(safe_costs)) if safe_costs else float("nan")
    mean_unsafe = float(np.mean(unsafe_costs)) if unsafe_costs else float("nan")

    # ── 10 random surviving pairs -> json ──────────────────────────────────
    sample_n = min(10, n_final)
    sample_pairs = random.sample(final_pairs, sample_n) if sample_n else []
    with open(config.SAMPLE_PAIRS_PATH, "w") as f:
        json.dump(sample_pairs, f, indent=2, ensure_ascii=False)

    # ── histogram ──────────────────────────────────────────────────────────
    if len(spreads):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.figure(figsize=(7, 4))
            plt.hist(spreads, bins=30, color="#4C72B0", edgecolor="white")
            plt.xlabel("Per-prompt cost spread (rejected − chosen)")
            plt.ylabel("count")
            plt.title("Cost spread across surviving pairs")
            plt.tight_layout()
            plt.savefig(config.COST_SPREAD_FIG_PATH, dpi=120)
            plt.close()
        except Exception as e:
            print(f"WARNING: could not write histogram: {e}")

    # ── assemble report ────────────────────────────────────────────────────
    lines = []
    lines.append("# Harmlessness preference-pair pipeline — report\n")
    lines.append(f"- Sampled prompts (denominator): **{config.SAMPLE}**")
    lines.append(f"- Responses per prompt (N): {config.N}, max_tokens: {config.MAX_TOKENS}\n")
    lines.append("## Filtering funnel")
    lines.append(f"- filter-1 (safety-label diversity): dropped **{n_dropped_f1} / {config.SAMPLE}** "
                 f"({100.0 * n_dropped_f1 / config.SAMPLE:.1f}%) → {n_enter_stage4} enter scoring")
    lines.append(f"- filter-3 (chosen=Safe & rejected=Unsafe): dropped **{n_dropped_f3} / "
                 f"{n_enter_stage4}** ({pct_f3:.1f}%) → {n_after_f3} pairs")
    if config.MIN_MARGIN is not None:
        pct_f4 = 100.0 * n_dropped_f4 / max(n_after_f3, 1)
        lines.append(f"- filter-4 (margin ≥ {config.MIN_MARGIN:g}): dropped **{n_dropped_f4} / "
                     f"{n_after_f3}** ({pct_f4:.1f}%)")
    else:
        lines.append("- filter-4 (minimum margin): **not applied** (MIN_MARGIN unset)")
    lines.append(f"- **End-to-end survival rate: {n_final} / {config.SAMPLE} = {survival:.1f}%**\n")
    lines.append("## Cost-spread distribution (rejected − chosen)")
    if len(spreads):
        lines.append(f"- mean: {spreads.mean():.3f}  median: {np.median(spreads):.3f}")
        lines.append(f"- min: {spreads.min():.3f}  max: {spreads.max():.3f}")
        lines.append(f"- q25: {q(spreads, 25):.3f}  q75: {q(spreads, 75):.3f}")
        lines.append(f"- histogram: `{config.COST_SPREAD_FIG_PATH}`\n")
    else:
        lines.append("- (no surviving pairs)\n")
    lines.append("## Length / verbosity confound (chosen vs rejected)")
    if n_final:
        lines.append(f"- **Full response (DPO pair content), tokens** — chosen mean "
                     f"{cho_tok.mean():.0f}, rejected mean {rej_tok.mean():.0f}, "
                     f"mean Δ(rej−cho) {tok_delta.mean():+.0f} (median {np.median(tok_delta):+.0f}); "
                     f"rejected longer in {pct_rej_longer_tok:.0f}% of pairs")
        lines.append(f"- **Visible answer (scored text), words** — chosen mean {cho_w.mean():.0f}, "
                     f"rejected mean {rej_w.mean():.0f}, mean Δ(rej−cho) {w_delta.mean():+.0f} "
                     f"(median {np.median(w_delta):+.0f}); rejected longer in {pct_rej_longer_w:.0f}% of pairs")
        r_str = "n/a (no length variance)" if np.isnan(r_margin_len) else f"{r_margin_len:+.3f}"
        lines.append(f"- **Pearson(cost margin, answer-word Δ)**: {r_str} "
                     f"(|r|→1 ⇒ cost margin is confounded by length; ~0 ⇒ decoupled)")
        lines.append("- Note: a strong one-sided skew (rejected ≫ chosen) risks DPO learning "
                     "length rather than harmlessness — watch this when scaling up.\n")
    else:
        lines.append("- (no surviving pairs)\n")
    lines.append("## Sign-convention sanity check")
    lines.append(f"- mean Beaver cost of **Unsafe** answers: {mean_unsafe:.4f} (n={len(unsafe_costs)})")
    lines.append(f"- mean Beaver cost of **Safe**   answers: {mean_safe:.4f} (n={len(safe_costs)})")
    ok = (mean_unsafe > mean_safe)
    lines.append(f"- Unsafe > Safe (expected)? **{'PASS' if ok else 'FAIL'}**\n")
    lines.append(f"## Artifacts")
    lines.append(f"- final pairs: `{config.FINAL_PAIRS_PATH}` ({n_final} pairs)")
    lines.append(f"- per-pair margins (CSV): `{config.PAIRS_MARGINS_PATH}`")
    lines.append(f"- sample pairs: `{config.SAMPLE_PAIRS_PATH}` ({sample_n} pairs)")

    report = "\n".join(lines)
    with open(config.REPORT_PATH, "w") as f:
        f.write(report + "\n")

    print("\n" + report)
    print(f"\nSaved report -> {config.REPORT_PATH}")
    if not ok:
        print("\nWARNING: sanity check FAILED — Unsafe mean cost is not above Safe mean cost. "
              "Check the cost-model sign convention or guard labels.")


if __name__ == "__main__":
    main()
