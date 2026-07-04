# WildJailbreak harmlessness-pair pilot + Beaver ranker sanity check

## What this experiment is

We are building **harmlessness preference pairs** (`chosen` = safer response, `rejected` =
more harmful response to the *same* prompt) that could later train / evaluate a model on
refusing harmful requests. The prompts come from **`allenai/wildjailbreak`**, using only its
two harmful subsets:

- **`vanilla_harmful`** — plain, direct harmful requests (prompt = the `vanilla` column).
- **`adversarial_harmful`** — the same intents wrapped in jailbreak framing (roleplay,
  obfuscation, persona injection; prompt = the `adversarial` column).

For each prompt we sample **N=10** Qwen3-4B responses (thinking mode on). Because a
well-aligned model refuses some of the time and complies some of the time, a single prompt
can yield both safe and unsafe answers — the raw material for a preference pair.

This run has **two objectives**:

1. **Pilot** — measure how many usable preference pairs WildJailbreak yields via Qwen3-4B
   self-generation (the survival funnel), and produce the pairs themselves.
2. **Beaver ranker sanity check** — decide whether `PKU-Alignment/beaver-7b-unified-cost` is
   a *valid instrument* for **ranking** those responses by harmfulness on this data, or
   whether it must be replaced.

## Why the sanity check matters

`beaver-7b-unified-cost` is a 2023, **LLaMA-1-based** cost model with a **2048-token context**,
trained on PKU-SafeRLHF (relatively plain harmful prompts). WildJailbreak's **adversarial**
prompts are long and out-of-distribution for it, so before trusting Beaver's cost to pick
`chosen`/`rejected`, we test whether it actually separates safe from unsafe answers here —
and whether scoring against the **vanilla seed query** (instead of the long adversarial prompt)
rescues it.

## Models and their roles

| Model | Role |
|---|---|
| `Qwen/Qwen3-4B` | Generator — produces N responses per prompt (thinking mode). |
| `allenai/wildguard` | **Labeler** — binary Harmful-response (yes→Unsafe / no→Safe) per answer; also emits `refusal` + `harmful_request`. Purpose-built for WildJailbreak and independent of the generator. Gates filter-1/filter-3. |
| `PKU-Alignment/beaver-7b-unified-cost` | **Ranker** — scalar cost (higher = more harmful) to pick chosen/rejected. |
| `Qwen/Qwen3Guard-Gen-4B` | Available as an independent reference/validator (different model from the labeler). |

The labeler (WildGuard) is a *different* model from the generator (Qwen3-4B), so labels are
not self-graded; and a *different* model again (Beaver) does the ranking, so filter-3 is a
real agreement test, not circular.

## Pipeline (each stage reads/writes a joblib in `outputs/`)

| Stage | Script | What it does |
|---|---|---|
| 01 | `01_prepare_prompts.py` | Load WildJailbreak `vanilla_harmful` + `adversarial_harmful`, dedup, sample `SAMPLE=300` (seed 0). Also saves per-prompt `{subset, vanilla seed}`. |
| 02 | `02_generate_responses.py` | Qwen3-4B, N=10, temp 0.6 / top_p 0.95 / top_k 20, `max_tokens=8192`; tuned vLLM (prefix caching, capped `max_model_len`, 0.90 VRAM); live truncation reporting. |
| 03 | `03_classify_guard.py` | Strip `</think>`, **WildGuard** label per answer (Harmful-response yes/no → Unsafe/Safe; also stores refusal + harmful_request), **filter-1** (drop prompts whose answers all share one label). Also dumps `03_all_labeled.joblib`. |
| 04 | `04_score_cost.py` | Beaver cost (left-truncated at 2048); per prompt `chosen=argmin`, `rejected=argmax`. |
| 05 | `05_report.py` | **filter-3** (chosen=Safe & rejected=Unsafe), survival rate, cost spread, 10 sample pairs, sign sanity check → `outputs/report.md`. |
| + | `sanity_beaver.py` | Beaver ranker sanity check (below) → `outputs/beaver_sanity_report.md`. |

## Beaver ranker sanity check — what it computes

For a WildJailbreak run's answers (reusing stage 01 + 03), per (prompt, answer):
`cost_full = Beaver(raw prompt, answer)` and `cost_vanilla = Beaver(vanilla seed, answer)`
(one call for the vanilla subset, two for adversarial). Then:

1. **Truncation exposure** over Beaver's 2048 ctx (inputs left-truncated: keep the answer).
2. **ROC-AUC** of cost vs Qwen3Guard labels in 3 conditions — (vanilla, `cost_full`),
   (adversarial, `cost_full`), (adversarial, `cost_vanilla`) — plus mean unsafe/safe cost.
3. **Pair agreement**: Beaver vs WildGuard argmin/argmax (chosen/rejected), per subset.
4. **Length confound** (adversarial): Pearson(cost_full, prompt token length) within
   safe- and within unsafe-labeled answers — detects cost being driven by prompt length.
5. **Disagreement dump**: 10 random adversarial answers where Beaver's sign (cost>0 ⇒ unsafe)
   contradicts Qwen3Guard.

### How to read the verdict
- adversarial `cost_full` AUC **≈** vanilla AUC → **Beaver usable as-is**.
- adversarial `cost_full` **≪** vanilla, but adversarial `cost_vanilla` **≈** vanilla →
  **Beaver usable only with vanilla-query scoring**.
- all conditions weak (AUC ≲ 0.65) or high truncation on adversarial → **drop Beaver, use
  WildGuard as the sole ranker**.

## Results (saved to `.md`)

- **`outputs/report.md`** — pilot funnel + survival + cost spread + the appended sanity section.
- **`outputs/beaver_sanity_report.md`** — standalone sanity-check results with the AUC table,
  pair-agreement, length correlations, and truncation summary.
- **`outputs/beaver_sanity_disagreements.jsonl`** — the 10 disagreement cases for eyeballing.
- **`outputs/final_pairs.joblib`**, **`outputs/sample_pairs.json`** — the mined preference pairs.

## Key settings / conventions

- `SAMPLE=300`, `N=10`, `MAX_TOKENS=8192`, seed `0`.
- Cost sign: higher Beaver cost = more harmful ⇒ `chosen=min`, `rejected=max`.
- Qwen3Guard **Controversial** → treated as **Unsafe**.
- Answers truncated mid-`<think>` (no visible answer) are marked invalid and excluded.
- WildJailbreak and WildGuard are **gated** — accept terms on HF and set `HF_TOKEN`.
