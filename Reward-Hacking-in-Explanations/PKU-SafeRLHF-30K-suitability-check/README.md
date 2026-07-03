# harmlessness-exp

Mine **harmlessness preference pairs** from the harmful prompts of `allenai/wildjailbreak`
(the `vanilla_harmful` and `adversarial_harmful` subsets).

For each sampled prompt we generate N Qwen3-4B responses (thinking mode on), keep only
prompts whose responses *disagree* on safety, then use a cost model to pick the safest
response as `chosen` and the most harmful as `rejected`, validated by a safety classifier.

## Pipeline

| Stage | Script | What it does |
|------|--------|--------------|
| 01 | `01_prepare_prompts.py` | Load WildJailbreak `vanilla_harmful` + `adversarial_harmful`, dedup on prompt, report counts + sample mix, sample `SAMPLE=300` (seed 0) |
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

**WildJailbreak is gated** — accept the terms on its HF page, then make `HF_TOKEN` available
(shell `export`, a `.env` file, or `huggingface-cli login`). `.env` is gitignored, so it does
NOT sync via git; create it on the pod or export the token there. Also optionally set
`HF_MODELS_CACHE_DIR`.

## Run

```bash
./run_all.sh            # runs all five stages in order
# or a single stage, e.g.:  python 02_generate_responses.py
```

## Beaver ranker sanity check (`sanity_beaver.py`)

An add-on that tests whether `beaver-7b-unified-cost` is a usable *ranker* on WildJailbreak,
scored (a) with the raw prompt (`cost_full`) and (b) with the vanilla seed query
(`cost_vanilla`). It reuses stages 01+03 (needs `03_all_labeled.joblib` + `01_prompts_meta.joblib`)
and loads Beaver then WildGuard (`allenai/wildguard`) as an independent reference ranker.

It reports (appended to `outputs/report.md`, disagreements to
`outputs/beaver_sanity_disagreements.jsonl`):
- **Truncation** exposure over Beaver's 2048 ctx (inputs are left-truncated, keeping the answer).
- **ROC-AUC** of cost vs Qwen3Guard labels in 3 conditions: (vanilla, `cost_full`),
  (adversarial, `cost_full`), (adversarial, `cost_vanilla`); with mean unsafe/safe cost.
- **Pair agreement**: Beaver vs WildGuard argmin/argmax (chosen/rejected) per subset.
- **Length confound**: Pearson(cost_full, prompt token length) within safe/unsafe answers (adversarial).
- A 10-row **disagreement dump** (Beaver sign vs Qwen3Guard, adversarial).

Run it after stage 03 (stages 04/05 not required):
```bash
python 01_prepare_prompts.py      # writes 01_prompts_meta.joblib (subset + vanilla seed)
python 02_generate_responses.py   # N=10 per prompt
python 03_classify_guard.py       # writes 03_all_labeled.joblib
python sanity_beaver.py           # Beaver + WildGuard scoring + metrics
```

**Reading it:** adversarial `cost_full` AUC ≈ vanilla AUC → Beaver usable as-is; adversarial
`cost_full` ≪ vanilla but adversarial `cost_vanilla` ≈ vanilla → use Beaver only with
vanilla-query scoring; all weak → drop Beaver, use WildGuard as the sole ranker.

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
- Prompts come from WildJailbreak `vanilla_harmful` (the plain request) and
  `adversarial_harmful` (the jailbreak-wrapped request); stage 01 prints the sample's mix.
- Survival is reported **overall**; survival denominator is the **300** sampled prompts.
