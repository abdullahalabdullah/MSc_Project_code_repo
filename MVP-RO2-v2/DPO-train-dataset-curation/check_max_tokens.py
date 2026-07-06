"""ONE-OFF DIAGNOSTIC — was MAX_TOKENS set right? Safe to delete this file at any time.

Standalone on purpose: imports nothing from the pipeline (no config/utils), only reads a
finished run's artifacts. Run it after a smoke/pilot curation run to decide whether the
generation token budget (MAX_TOKENS) and the trainer's MAX_LENGTH are too high, too low,
or about right — before committing to the full-scale run.

What it reads (from the run output dir):
  02_responses.joblib   response token counts + truncation flags (required)
  final_pairs.joblib    chosen/rejected full-response token counts (optional)

What it reports:
  - truncation rate: responses cut off at the budget (each one is a wasted generation —
    truncated-mid-think responses are invalid downstream). High rate => budget too LOW.
  - response-length percentiles vs the budget: if even p99.9 sits far below MAX_TOKENS,
    the budget is too HIGH — a smaller one raises vLLM KV-cache concurrency (max_model_len
    = MAX_TOKENS + prompt budget) without losing responses.
  - final-pair lengths vs the trainer's MAX_LENGTH: the longest completion that actually
    survives into the DPO dataset bounds what training needs to fit.

Usage:
  python check_max_tokens.py outputs/run_<timestamp> \
      [--max-tokens 8192] [--prompt-budget 4096] [--train-max-length 12288]
"""
import argparse
import os

import joblib
import numpy as np


def pctl(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def suggest_budget(need_tokens):
    """Smallest 'round' budget (multiple of 1024) covering `need_tokens` + 10% margin."""
    need = int(need_tokens * 1.10)
    return max(1024, int(np.ceil(need / 1024)) * 1024)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", help="curation run output dir (contains 02_responses.joblib)")
    p.add_argument("--max-tokens", type=int, default=8192,
                   help="the MAX_TOKENS the run generated with (default 8192)")
    p.add_argument("--prompt-budget", type=int, default=4096,
                   help="prompt token budget used for max_model_len / MAX_PROMPT_LENGTH (default 4096)")
    p.add_argument("--train-max-length", type=int, default=12288,
                   help="the trainer's MAX_LENGTH (default 12288)")
    args = p.parse_args()

    resp_path = os.path.join(args.run_dir, "02_responses.joblib")
    if not os.path.exists(resp_path):
        raise SystemExit(f"ERROR: {resp_path} not found — point me at a finished run dir.")

    flat = [rd for per_prompt in joblib.load(resp_path) for rd in per_prompt]
    toks = np.array([rd["num_tokens"] for rd in flat], dtype=float)
    trunc = np.array([bool(rd.get("truncated")) for rd in flat])
    n = len(flat)
    n_trunc = int(trunc.sum())
    pct_trunc = 100.0 * n_trunc / max(n, 1)

    print(f"run: {args.run_dir}")
    print(f"responses: {n}   budget (MAX_TOKENS): {args.max_tokens}\n")

    print("── Truncation (budget too LOW if high) ──")
    print(f"  truncated at budget: {n_trunc} / {n} ({pct_trunc:.2f}%)")

    print("\n── Response length distribution (completion tokens) ──")
    for q in (50, 90, 95, 99, 99.9):
        print(f"  p{q:<5}: {pctl(toks, q):8.0f}")
    print(f"  max   : {toks.max():8.0f}" if n else "  (no responses)")
    p999 = pctl(toks, 99.9)
    print(f"  p99.9 uses {100.0 * p999 / args.max_tokens:.0f}% of the budget")

    # ── verdict on MAX_TOKENS ──
    print("\n── Verdict: MAX_TOKENS ──")
    if pct_trunc > 1.0:
        print(f"  TOO LOW: {pct_trunc:.1f}% of generations hit the budget (>1% threshold).")
        print(f"  Each truncated response is a wasted generation (invalid downstream).")
        print(f"  Consider raising MAX_TOKENS — but note this changes generation settings, ")
        print(f"  so earlier runs would no longer be comparable/mergeable.")
    elif p999 < 0.6 * args.max_tokens:
        sugg = suggest_budget(p999)
        if sugg < args.max_tokens:
            print(f"  TOO HIGH (wasteful, not harmful): p99.9 = {p999:.0f} tokens — even the")
            print(f"  longest 0.1% of responses use <60% of the {args.max_tokens} budget.")
            print(f"  A budget of ~{sugg} would cover p99.9 with 10% margin; smaller")
            print(f"  max_model_len ({sugg}+{args.prompt_budget}) = higher vLLM KV concurrency,")
            print(f"  and the trainer MAX_LENGTH could drop to {sugg + args.prompt_budget}.")
            print(f"  Only worth changing BEFORE the full run — runs must share one setting.")
        else:
            print("  ABOUT RIGHT.")
    else:
        print(f"  ABOUT RIGHT: truncation {pct_trunc:.2f}% (<=1%) and long responses "
              f"actually use the budget (p99.9 = {p999:.0f}).")

    # ── what the trainer actually needs (final pairs only) ──
    pairs_path = os.path.join(args.run_dir, "final_pairs.joblib")
    if os.path.exists(pairs_path):
        pairs = joblib.load(pairs_path)
        if pairs:
            comp = np.array([t for pr in pairs
                             for t in (pr["chosen_full_tokens"], pr["rejected_full_tokens"])],
                            dtype=float)
            need = comp.max() + args.prompt_budget
            print("\n── Trainer MAX_LENGTH check (completions that survived into the DPO dataset) ──")
            print(f"  final pairs: {len(pairs)}   longest completion: {comp.max():.0f} tokens "
                  f"(p99: {pctl(comp, 99):.0f})")
            print(f"  worst case prompt+completion: {comp.max():.0f} + {args.prompt_budget} "
                  f"(prompt budget) = {need:.0f}")
            if need <= args.train_max_length:
                print(f"  OK: fits the trainer's MAX_LENGTH={args.train_max_length} "
                      f"({args.train_max_length - need:.0f} tokens spare).")
            else:
                print(f"  WARNING: exceeds MAX_LENGTH={args.train_max_length} — those pairs "
                      f"would be truncated in training. Raise MAX_LENGTH.")
        else:
            print("\n(final_pairs.joblib is empty — no trainer-side check)")
    else:
        print("\n(final_pairs.joblib not found — skipping the trainer MAX_LENGTH check)")


if __name__ == "__main__":
    main()
