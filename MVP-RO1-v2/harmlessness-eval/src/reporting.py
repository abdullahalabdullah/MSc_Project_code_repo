"""
Markdown report generation for the harmlessness eval (scripts/05_generate_reports.py).

ALL .md generation lives here (plus the 05 script and one driver line) so it can
be removed cleanly for a public release: delete src/reporting.py,
scripts/05_generate_reports.py, and the reports phase in run_harmlessness.sh —
the metric computation (step 04) is untouched.

Reads ONLY the machine-readable metrics.json from step 04, so the .md can be
deleted and regenerated with a single 05 run (no inference/classifier re-run).
No emojis.
"""

import json
from pathlib import Path


def _pct(v) -> str:
    return f"{100 * v:.1f}%" if v is not None else "N/A"


def _metric_row(label: str, m: dict) -> str:
    return (f"| {label} | {m['n']} | {_pct(m['safety_rate'])} | {_pct(m['refusal_rate'])} "
            f"| {_pct(m['prompt_harmful_rate'])} | {m['n_missing_verdict']} "
            f"| {m.get('n_input_truncated', 0)} |")


def write_summary_md(metrics: dict, meta: dict, path: Path) -> None:
    """Render harmlessness_summary.md from a metrics dict + run metadata."""
    lines = []
    w = lines.append

    w(f"# Harmlessness Results: {meta.get('model_name', '?')}\n")
    w(f"- **Model:** `{meta.get('hf_id', '?')}`")
    w(f"- **Benchmark:** `{meta.get('benchmark', 'allenai/wildjailbreak')}` "
      f"(eval: adversarial harmful + benign) · {metrics.get('n_total', 0)} responses")
    w(f"- **Judge:** `{meta.get('classifier', 'allenai/wildguard')}` (independent safety classifier)")
    w(f"- **Generation:** thinking {'ON' if meta.get('enable_thinking') else 'OFF'}, "
      f"n_samples {meta.get('n_samples', '?')}, classify target: {meta.get('classify_target', 'answer')}")
    w(f"- **Sampling:** temperature {meta.get('temperature', '?')}, top_p {meta.get('top_p', '?')}, "
      f"top_k {meta.get('top_k', '?')}, max_new_tokens {meta.get('max_new_tokens', '?')}")
    w("")
    w("**Safety Rate** = responses judged not harmful. **Refusal Rate** = responses "
      "judged refusals. On the harmful subset a high safety rate is the headline; "
      "on the benign subset a low refusal rate indicates little over-refusal. "
      "Prompt-harmful is WildGuard's view of the request (a sanity check on the "
      "subset labels). Missing = responses WildGuard left unparsed (excluded from "
      "the rate denominators). Truncated = responses too long for WildGuard's "
      "context window and cut before judging (runaway generations; verdict is on "
      "a partial response).\n")

    w("## Results\n")
    w("| Scope | N | Safety Rate | Refusal Rate | Prompt-harmful | Missing | Truncated |")
    w("|-------|---|-------------|--------------|----------------|---------|-----------|")
    w(_metric_row("**Overall**", metrics["overall"]))
    for subset, m in metrics.get("by_subset", {}).items():
        w(_metric_row(subset, m))
    w("")

    path.write_text("\n".join(lines), encoding="utf-8")


def load_metrics(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
