# harmlessness-exp

Mine **harmlessness preference pairs** from `PKU-Alignment/PKU-SafeRLHF-30K`.

For each sampled prompt we generate 10 Qwen3-4B responses (thinking mode on), keep only
prompts whose responses *disagree* on safety, then use a cost model to pick the safest
response as `chosen` and the most harmful as `rejected`, validated by a safety classifier.

## Pipeline

| Stage | Script | What it does |
|------|--------|--------------|
| 01 | `01_prepare_prompts.py` | Load + dedup on `prompt`, report unique count, sample `SAMPLE=300` (seed 0) |
| 02 | `02_generate_responses.py` (GPU) | Qwen3-4B, N=10, temp 0.6 / top_p 0.95 / top_k 20, `max_tokens=8192`; tuned vLLM engine (prefix caching, capped `max_model_len`, 0.90 VRAM); **live truncation reporting** |
| 03 | `03_classify_guard.py` (GPU) | Split on `</think>`, Qwen3Guard-Gen-4B safe/unsafe, **filter-1** (drop prompts whose valid responses share one label) |
| 04 | `04_score_cost.py` (GPU) | `beaver-7b-unified-cost`; per prompt `chosen=argmin(cost)`, `rejected=argmax(cost)` |
| 05 | `05_report.py` | **filter-3** (chosen=Safe & rejected=Unsafe), survival rate, cost-spread dist, 10 sample pairs, sign-convention sanity check |

Each stage reads/writes a compressed `.joblib` in `outputs/`, so stages are independent
and resumable.

## Environment (single venv)

Everything runs in one environment — the Beaver cost model is loaded via the self-contained
`beaver_cost_model.py` (plain `transformers`, no `safe-rlhf`), so there is no dependency
conflict with the Qwen3 / vLLM stack.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Set `HF_TOKEN` (and optionally `HF_MODELS_CACHE_DIR`) in your shell or a `.env` file.

## Run

```bash
./run_all.sh            # runs all five stages in order
# or a single stage, e.g.:  python 02_generate_responses.py
```

## Smoke test

Edit `config.py`: `SAMPLE=8`, `N=3`, `MAX_TOKENS=1024`, then run the pipeline end-to-end.
Confirm `outputs/report.md`, `outputs/sample_pairs.json`, and `outputs/cost_spread.png`
are produced and the sign-convention sanity check prints `PASS`.

## Key outputs (`outputs/`)

- `final_pairs.joblib` — `{prompt, chosen, rejected, chosen_cost, rejected_cost, chosen_label, rejected_label}`
- `report.md` — funnel percentages, survival rate, cost-spread stats, sanity check
- `sample_pairs.json` — 10 random surviving pairs for manual eyeballing
- `cost_spread.png` — histogram of per-prompt cost spread

## Conventions / notes

- **Cost sign**: higher Beaver cost = more harmful ⇒ `chosen=min`, `rejected=max`. The local
  loader outputs the raw (pre-normalizer) score; this preserves the argmin/argmax and the
  Unsafe>Safe sanity-check direction (see `beaver_cost_model.py`).
- **Controversial** Qwen3Guard label → treated as **Unsafe** (conservative).
- Responses truncated mid-`<think>` have no visible answer → marked invalid, excluded from candidates.
- `PKU-SafeRLHF-30K` has no harm-category field, so survival is reported **overall only**.
- Survival denominator is the **300** sampled prompts.
