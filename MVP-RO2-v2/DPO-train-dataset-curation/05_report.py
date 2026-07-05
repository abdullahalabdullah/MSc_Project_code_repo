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
from collections import Counter

import joblib
import numpy as np

import config
from utils import set_seed, update_diagnostics


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
        # Margin calibration + labeler decisiveness, computed from ALL valid responses of
        # this prompt (available here, discarded after we keep only chosen/rejected):
        #   runner-up gaps  = how isolated the extreme picks are from the 2nd-best/2nd-worst;
        #                     a tiny gap ⇒ the chosen/rejected choice was nearly a tie (fragile).
        #   unsafe_frac     = Qwen3Guard decisiveness on the prompt (fraction Unsafe; ~0.5 = split).
        valid_rds = [rd for rd in per_prompt if not rd["invalid"] and "cost" in rd]
        sorted_costs = sorted(rd["cost"] for rd in valid_rds)
        chosen_gap = sorted_costs[1] - sorted_costs[0] if len(sorted_costs) >= 2 else 0.0
        rejected_gap = sorted_costs[-1] - sorted_costs[-2] if len(sorted_costs) >= 2 else 0.0
        prompt_labels = [rd["guard_label"] for rd in valid_rds if rd["guard_label"] is not None]
        unsafe_frac = (sum(l == "Unsafe" for l in prompt_labels) / len(prompt_labels)
                       if prompt_labels else float("nan"))
        if ch["guard_label"] == "Safe" and rj["guard_label"] == "Unsafe":
            # The RM (Beaver) and classifier (Qwen3Guard) were applied to the VISIBLE answer
            # only (see stages 03/04, which read rd["answer"]). But the DPO pair itself keeps
            # the FULL response (<think>…</think> + answer) — the CoT is retained for training,
            # not discarded. We also stash the visible answer that was actually scored/labeled.
            ch_answer, rj_answer = ch["answer"], rj["answer"]
            final_pairs.append({
                "prompt": ch["prompt"],
                "subset": ch.get("subset"),          # vanilla | adversarial (per-subset funnel)
                "chosen": ch["full_response"],       # full response (CoT + answer) — used for DPO
                "rejected": rj["full_response"],
                "chosen_answer": ch_answer,          # visible answer that was scored/labeled
                "rejected_answer": rj_answer,
                "chosen_cost": ch["cost"],
                "rejected_cost": rj["cost"],
                "margin": rj["cost"] - ch["cost"],   # cost margin (rejected - chosen), always >= 0
                # margin calibration: isolation of the chosen/rejected picks + labeler decisiveness
                "chosen_runner_up_gap": chosen_gap,
                "rejected_runner_up_gap": rejected_gap,
                "unsafe_frac": unsafe_frac,
                "chosen_label": ch["guard_label"],
                "rejected_label": rj["guard_label"],
                # length bookkeeping for the verbosity-confound report
                "chosen_full_tokens": ch["num_tokens"],      # full-response tokens (CoT + answer)
                "rejected_full_tokens": rj["num_tokens"],
                "chosen_answer_words": len(ch_answer.split()),   # visible-answer words
                "rejected_answer_words": len(rj_answer.split()),
                "chosen_think_words": ch.get("think_words", 0),  # CoT length (words)
                "rejected_think_words": rj.get("think_words", 0),
            })
        else:
            n_dropped_f3 += 1

    pct_f3 = 100.0 * n_dropped_f3 / max(n_enter_stage4, 1)
    n_after_f3 = len(final_pairs)

    # ── per-subset funnel: sample mix carried through each filter ───────────
    # Subset travels on every response dict (attached in stage 02), so each stage's mix is a
    # plain Counter. filter-1 is measured against the sampled set via the meta file (stage 05
    # only receives filter-1 survivors, so the dropped prompts are not otherwise visible here).
    prompt_meta = joblib.load(config.PROMPTS_META_PATH)
    subsets = sorted({m["subset"] for m in prompt_meta})
    sampled_mix = Counter(m["subset"] for m in prompt_meta)
    f1_mix = Counter(pp[0].get("subset") for pp in survivors)         # after filter-1
    f3_mix = Counter(p["subset"] for p in final_pairs)                # after filter-3 (pre f4)

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
    final_mix = Counter(p["subset"] for p in final_pairs)             # after filter-4 (== f3 if off)

    joblib.dump(final_pairs, config.FINAL_PAIRS_PATH, compress=True)

    # Per-pair margin table (CSV) — one row per pair, easy to analyse / plot.
    with open(config.PAIRS_MARGINS_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_index", "subset", "chosen_cost", "rejected_cost", "margin",
                    "chosen_runner_up_gap", "rejected_runner_up_gap", "unsafe_frac",
                    "chosen_full_tokens", "rejected_full_tokens",
                    "chosen_answer_words", "rejected_answer_words",
                    "chosen_think_words", "rejected_think_words",
                    "chosen_label", "rejected_label", "prompt"])
        for i, p in enumerate(final_pairs):
            w.writerow([i, p["subset"], p["chosen_cost"], p["rejected_cost"], p["margin"],
                        p["chosen_runner_up_gap"], p["rejected_runner_up_gap"], p["unsafe_frac"],
                        p["chosen_full_tokens"], p["rejected_full_tokens"],
                        p["chosen_answer_words"], p["rejected_answer_words"],
                        p["chosen_think_words"], p["rejected_think_words"],
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
        # think-block (CoT) length vs visible-answer length, chosen vs rejected
        cho_think = np.array([p["chosen_think_words"] for p in final_pairs], dtype=float)
        rej_think = np.array([p["rejected_think_words"] for p in final_pairs], dtype=float)
        # margin-calibration aggregates: how isolated the extreme picks are + labeler split
        cho_gap = np.array([p["chosen_runner_up_gap"] for p in final_pairs], dtype=float)
        rej_gap = np.array([p["rejected_runner_up_gap"] for p in final_pairs], dtype=float)
        unsafe_fracs = np.array([p["unsafe_frac"] for p in final_pairs], dtype=float)

    # ── labeler agreement: Beaver argmin/argmax vs Qwen3Guard, over filter-1 survivors ──
    # chosen-side agree = Beaver's safest (min-cost) response is also Qwen3Guard-Safe;
    # rejected-side agree = Beaver's harshest (max-cost) response is also Qwen3Guard-Unsafe;
    # both = the pair the two labelers agree on end-to-end (== filter-3 survivors).
    agree_chosen = agree_rejected = agree_both = 0
    for per_prompt in survivors:
        meta = per_prompt[0]["_pair_meta"]
        c_ok = per_prompt[meta["chosen_ix"]]["guard_label"] == "Safe"
        r_ok = per_prompt[meta["rejected_ix"]]["guard_label"] == "Unsafe"
        agree_chosen += c_ok
        agree_rejected += r_ok
        agree_both += (c_ok and r_ok)
    n_surv = max(n_enter_stage4, 1)

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

    # point-biserial r(cost, label=Unsafe) over all scored responses: one number for how well
    # Beaver's continuous cost separates Qwen3Guard's Safe/Unsafe classes (+1 = perfect, 0 = none).
    all_c = np.array(safe_costs + unsafe_costs, dtype=float)
    all_y = np.array([0] * len(safe_costs) + [1] * len(unsafe_costs), dtype=float)
    if len(all_c) >= 2 and np.std(all_c) > 0 and np.std(all_y) > 0:
        point_biserial = float(np.corrcoef(all_c, all_y)[0, 1])
    else:
        point_biserial = float("nan")

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
    lines.append("## Per-subset funnel (sample mix after each filter)")
    lines.append("| subset | sampled | after filter-1 | after filter-3 | final | survival % |")
    lines.append("|---|---|---|---|---|---|")
    for s in subsets:
        s_samp = sampled_mix.get(s, 0)
        surv = 100.0 * final_mix.get(s, 0) / s_samp if s_samp else float("nan")
        lines.append(f"| {s} | {s_samp} | {f1_mix.get(s, 0)} | {f3_mix.get(s, 0)} | "
                     f"{final_mix.get(s, 0)} | {surv:.1f} |")
    lines.append(f"| **all** | {config.SAMPLE} | {n_enter_stage4} | {n_after_f3} | "
                 f"{n_final} | {survival:.1f} |\n")
    lines.append("## Labeler agreement (Beaver argmin/argmax vs Qwen3Guard)")
    lines.append(f"- chosen-side (Beaver-safest is Qwen3Guard-Safe): "
                 f"**{100.0 * agree_chosen / n_surv:.1f}%** ({agree_chosen}/{n_enter_stage4})")
    lines.append(f"- rejected-side (Beaver-harshest is Qwen3Guard-Unsafe): "
                 f"**{100.0 * agree_rejected / n_surv:.1f}%** ({agree_rejected}/{n_enter_stage4})")
    lines.append(f"- both agree (→ filter-3 survivors): "
                 f"**{100.0 * agree_both / n_surv:.1f}%** ({agree_both}/{n_enter_stage4})")
    pb_str = "n/a" if np.isnan(point_biserial) else f"{point_biserial:+.3f}"
    lines.append(f"- point-biserial r(cost, Unsafe) over all scored responses: **{pb_str}** "
                 f"(+1 = cost perfectly separates Qwen3Guard classes; 0 = no separation)\n")
    lines.append("## Margin calibration (isolation of the chosen/rejected picks)")
    if n_final:
        lines.append(f"- **chosen runner-up gap** (2nd-safest − safest cost): mean "
                     f"{cho_gap.mean():.3f}, median {np.median(cho_gap):.3f} — small ⇒ the safe pick was near-tied")
        lines.append(f"- **rejected runner-up gap** (harshest − 2nd-harshest cost): mean "
                     f"{rej_gap.mean():.3f}, median {np.median(rej_gap):.3f}")
        for s in subsets:
            s_margins = [p["margin"] for p in final_pairs if p["subset"] == s]
            if s_margins:
                lines.append(f"- mean margin ({s}): {float(np.mean(s_margins)):.3f} (n={len(s_margins)})")
        lines.append(f"- **Qwen3Guard decisiveness** (mean Unsafe fraction per surviving prompt): "
                     f"{np.nanmean(unsafe_fracs):.2f} (0.5 = maximally split)\n")
    else:
        lines.append("- (no surviving pairs)\n")
    lines.append("## Think-block vs answer length (final pairs, words)")
    if n_final:
        lines.append(f"- **chosen** — think mean {cho_think.mean():.0f}, answer mean {cho_w.mean():.0f} "
                     f"(think/answer ratio {cho_think.mean() / max(cho_w.mean(), 1):.1f})")
        lines.append(f"- **rejected** — think mean {rej_think.mean():.0f}, answer mean {rej_w.mean():.0f} "
                     f"(think/answer ratio {rej_think.mean() / max(rej_w.mean(), 1):.1f})")
        lines.append("- (CoT is retained in the DPO pair but never scored/labeled — this tracks how "
                     "much of each response is reasoning vs visible answer.)\n")
    else:
        lines.append("- (no surviving pairs)\n")
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
    lines.append(f"- per-pair margins + gaps + think/answer lengths (CSV): `{config.PAIRS_MARGINS_PATH}`")
    lines.append(f"- sample pairs: `{config.SAMPLE_PAIRS_PATH}` ({sample_n} pairs)")
    lines.append(f"- consolidated per-stage diagnostics (JSON): `{config.DIAGNOSTICS_PATH}`")

    report = "\n".join(lines)
    with open(config.REPORT_PATH, "w") as f:
        f.write(report + "\n")

    # ── consolidated diagnostics: stage-05 sections (join the 02/03/04 sections) ──
    def _f(x):  # nan -> None so the JSON stays valid/portable
        return None if (isinstance(x, float) and np.isnan(x)) else x

    update_diagnostics(config.DIAGNOSTICS_PATH, "funnel", {
        "sampled": config.SAMPLE,
        "after_filter1": n_enter_stage4,
        "after_filter3": n_after_f3,
        "min_margin_filter4": config.MIN_MARGIN,
        "dropped_filter4": n_dropped_f4,
        "final_pairs": n_final,
        "survival_pct": round(survival, 3),
    })
    update_diagnostics(config.DIAGNOSTICS_PATH, "per_subset", {
        s: {
            "sampled": sampled_mix.get(s, 0),
            "after_filter1": f1_mix.get(s, 0),
            "after_filter3": f3_mix.get(s, 0),
            "final": final_mix.get(s, 0),
            "survival_pct": round(100.0 * final_mix.get(s, 0) / sampled_mix[s], 3)
                            if sampled_mix.get(s) else None,
        } for s in subsets
    })
    update_diagnostics(config.DIAGNOSTICS_PATH, "labeler_agreement", {
        "n_prompts": n_enter_stage4,
        "chosen_side_pct": round(100.0 * agree_chosen / n_surv, 3),
        "rejected_side_pct": round(100.0 * agree_rejected / n_surv, 3),
        "both_pct": round(100.0 * agree_both / n_surv, 3),
        "point_biserial_cost_unsafe": _f(round(point_biserial, 4) if not np.isnan(point_biserial) else point_biserial),
        "mean_cost_unsafe": _f(round(mean_unsafe, 4)),
        "mean_cost_safe": _f(round(mean_safe, 4)),
        "sign_convention_pass": bool(mean_unsafe > mean_safe),
    })
    update_diagnostics(config.DIAGNOSTICS_PATH, "margin_calibration", {
        "margin_mean": _f(round(float(spreads.mean()), 4)) if len(spreads) else None,
        "margin_median": _f(round(float(np.median(spreads)), 4)) if len(spreads) else None,
        "chosen_runner_up_gap_mean": _f(round(float(cho_gap.mean()), 4)) if n_final else None,
        "rejected_runner_up_gap_mean": _f(round(float(rej_gap.mean()), 4)) if n_final else None,
        "guard_decisiveness_mean_unsafe_frac":
            _f(round(float(np.nanmean(unsafe_fracs)), 4)) if n_final else None,
        "per_subset_mean_margin": {
            s: round(float(np.mean([p["margin"] for p in final_pairs if p["subset"] == s])), 4)
            for s in subsets if any(p["subset"] == s for p in final_pairs)
        },
    })
    update_diagnostics(config.DIAGNOSTICS_PATH, "length", {
        "chosen_full_tokens_mean": round(float(cho_tok.mean()), 1) if n_final else None,
        "rejected_full_tokens_mean": round(float(rej_tok.mean()), 1) if n_final else None,
        "chosen_answer_words_mean": round(float(cho_w.mean()), 1) if n_final else None,
        "rejected_answer_words_mean": round(float(rej_w.mean()), 1) if n_final else None,
        "chosen_think_words_mean": round(float(cho_think.mean()), 1) if n_final else None,
        "rejected_think_words_mean": round(float(rej_think.mean()), 1) if n_final else None,
        "pct_rejected_longer_answer": round(pct_rej_longer_w, 1) if n_final else None,
        "pearson_margin_vs_answer_len": _f(round(r_margin_len, 4)) if n_final else None,
    })

    print("\n" + report)
    print(f"\nSaved report -> {config.REPORT_PATH}")
    print(f"Saved diagnostics -> {config.DIAGNOSTICS_PATH}")
    if not ok:
        print("\nWARNING: sanity check FAILED — Unsafe mean cost is not above Safe mean cost. "
              "Check the cost-model sign convention or guard labels.")


if __name__ == "__main__":
    main()
