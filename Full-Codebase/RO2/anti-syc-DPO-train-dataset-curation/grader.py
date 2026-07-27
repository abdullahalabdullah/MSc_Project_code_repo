"""gpt-5-mini wrappers: answer classifier (spec §6) and wrong-answer generator (spec §5).

Retry/backoff/429-tagging follows RO1 monitorability-eval/src/monitor.py::classify —
`max_retries` attempts with backoff_factor**attempt sleeps; after exhausting retries a
RuntimeError is raised, prefixed "[429]" if a rate limit was seen, which is what
utils.run_concurrent keys its concurrency back-off on.

NOTE on temperature: the spec says temperature=0, but gpt-5-mini rejects non-default
temperature, so it is OMITTED from every call (API default) — same as the existing
monitorability monitor. Recorded in manifest.json.
"""
import json
import os
import re
import sys
import time

import config
import prompts


def get_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set (see .env / .env.example).")
        sys.exit(1)
    from openai import OpenAI
    return OpenAI(api_key=api_key)


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json_response(raw_text: str):
    """Parse a JSON-only reply, tolerating fenced/prefixed output via a {...} fallback."""
    try:
        return json.loads(raw_text)
    except (json.JSONDecodeError, TypeError):
        pass
    m = _JSON_BLOCK_RE.search(raw_text or "")
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _call_with_retries(client, user_prompt: str, what: str) -> str:
    """One chat.completions call with the monitor-style retry loop. Returns raw text."""
    last_exc = None
    hit_rate_limit = False
    for attempt in range(config.GRADER_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=config.GRADER_MODEL,
                messages=[{"role": "user", "content": user_prompt}],
                timeout=config.GRADER_TIMEOUT,
            )
            return response.choices[0].message.content
        except Exception as exc:
            last_exc = exc
            if getattr(exc, "status_code", None) == 429:
                hit_rate_limit = True
            if attempt < config.GRADER_MAX_RETRIES - 1:
                wait = config.GRADER_BACKOFF_FACTOR ** attempt
                print(f"{what} API error (attempt {attempt + 1}/{config.GRADER_MAX_RETRIES}): "
                      f"{exc}. Retrying in {wait:.1f}s ...")
                time.sleep(wait)
    tag = "[429]" if hit_rate_limit else ""
    raise RuntimeError(f"{tag}{what} API failed after {config.GRADER_MAX_RETRIES} attempts: "
                       f"{last_exc}") from last_exc


def classify_answer(client, question: str, gold: str, aliases: list, extracted: str,
                    asserted: str | None = None) -> dict:
    """Classify one extracted final answer (spec §6).

    Returns {"label": str|None, "raw_response": str}. label is None when the reply
    wasn't valid JSON with an allowed label (the caller treats that as UNPARSEABLE,
    matching the spec's "a sample whose classification ultimately fails" rule).
    """
    user_prompt = prompts.grader_prompt(question, gold, aliases, extracted, asserted)
    raw_text = _call_with_retries(client, user_prompt, "Grader")
    parsed = _parse_json_response(raw_text)
    allowed = prompts.BARE_LABELS if asserted is None else prompts.ASSERTED_LABELS
    label = None
    if isinstance(parsed, dict):
        candidate = str(parsed.get("label", "")).strip().upper()
        if candidate in allowed:
            label = candidate
    return {"label": label, "raw_response": raw_text}


def generate_wrong_answer(client, question: str, gold: str, aliases: list) -> dict:
    """Generate the plausible-but-certainly-wrong answer `w` (spec §5 Stage D).

    Returns {"wrong_answer": str|None, "raw_response": str}. wrong_answer is None when
    the reply wasn't valid JSON with a non-empty string.
    """
    user_prompt = prompts.wrong_answer_prompt(question, gold, aliases)
    raw_text = _call_with_retries(client, user_prompt, "Wrong-answer generator")
    parsed = _parse_json_response(raw_text)
    w = None
    if isinstance(parsed, dict):
        candidate = parsed.get("wrong_answer")
        if isinstance(candidate, str) and candidate.strip():
            w = candidate.strip()
    return {"wrong_answer": w, "raw_response": raw_text}
