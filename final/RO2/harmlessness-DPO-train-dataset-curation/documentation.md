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
| `Qwen/Qwen3Guard-Gen-4B` | **Labeler** — 3-way Safety verdict (Safe/Unsafe/Controversial) per answer, Controversial folded into Unsafe; also emits `refusal` (same pass) + `harmful_request` (separate prompt-only pass, deduped per unique prompt). Gates filter-1/filter-3. |
| `PKU-Alignment/beaver-7b-unified-cost` | **Ranker** — scalar cost (higher = more harmful) to pick chosen/rejected. |
| `allenai/wildguard` | Independent reference labeler used only by `sanity_beaver.py`, not stage 03. |

Note on independence: Qwen3Guard-Gen-4B is a *different checkpoint* from the generator
(Qwen3-4B), so labels are not literally self-graded, but both are Qwen3-family models from the
same lab with overlapping alignment data — see `labeler_comparison.md` for the earlier
WildGuard-vs-Qwen3Guard comparison and the non-circularity tradeoff it raises (kept as a
historical record; not updated when this switch was made). A *different* model again (Beaver)
does the ranking, so filter-3 is still a real cost-vs-label agreement test, not circular on
that axis.

## Pipeline (each stage reads/writes a joblib in `outputs/`)

| Stage | Script | What it does |
|---|---|---|
| 01 | `01_prepare_prompts.py` | Load WildJailbreak `vanilla_harmful` + `adversarial_harmful` (train config), dedup. Default `SAMPLE=ALL` keeps every unique prompt; `SAMPLE=<int>` (env var) draws a seeded sample. Also saves per-prompt `{subset, vanilla seed}`. |
| 02 | `02_generate_responses.py` | Qwen3-4B, N=10, temp 0.6 / top_p 0.95 / top_k 20, `max_tokens=8192`; tuned vLLM (prefix caching, capped `max_model_len`, 0.95 VRAM); live truncation reporting. |
| 03 | `03_classify_guard.py` | Strip `</think>`, **Qwen3Guard** label per answer (3-way Safe/Unsafe/Controversial → Controversial folded into Unsafe; also stores refusal + harmful_request), **filter-1** (drop prompts whose answers all share one label). Also dumps `03_all_labeled.joblib`. Inference on **vLLM** (greedy, like stage 02); **resumable** — both passes checkpoint every `GUARD_CHECKPOINT_EVERY` pairs to `03_guard_*.ckpt.jsonl` (a fingerprint discards a checkpoint left over from a different `02_responses.joblib`); checkpoints are deleted once the stage completes. |
| 04 | `04_score_cost.py` | Beaver cost (forward-only, left-truncated at 2048; **length-sorted** batches to cut padding waste); per prompt `chosen=argmin`, `rejected=argmax`. **Resumable** — checkpoints every batch to `04_cost.ckpt.jsonl` (fingerprint discards a stale checkpoint), deleted once the stage completes. |
| 05 | `05_report.py` | **filter-3** (chosen=Safe & rejected=Unsafe), survival rate, cost spread, 10 sample pairs, sign sanity check → `outputs/report.md`. |
| 06 | `06_export_dpo_dataset.py` | **Handoff**: `final_pairs.joblib` → HF `DatasetDict({train,valid})` with `prompt`/`chosen`/`rejected` → `outputs/dpo_dataset/`, the training-ready artifact. Valid holdout (`VALID_FRACTION`, default 0.1) is drawn from the final pairs — seeded random by default; `STRATIFY_VALID=true` for a 50/50 vanilla/adversarial mix (best-effort). |
| + | `sanity_beaver.py` | Beaver ranker sanity check (below, optional, not part of `run_all.sh`) → `outputs/beaver_sanity_report.md`. |
| + | `merge_runs.py` | Combine multiple runs' `final_pairs.joblib` into one dataset (dedup on prompt, fresh seeded split via stage 06). Optional `TARGET_PAIRS` env caps the merged pool at an exact size (seeded downsample before export). Pairs with `EXCLUDE_RUNS` (stage 01 env var) for a disjoint pilot + top-up workflow. |

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
- **`outputs/dpo_dataset/`** — the training-ready handoff: HF `DatasetDict({train,valid})` with
  `prompt`/`chosen`/`rejected`, produced by stage 06.

## Key settings / conventions

- `SAMPLE=ALL` (default — every unique harmful prompt; env-overridable to an integer for pilots), `N=10`, `MAX_TOKENS=8192`.
- **One seed everywhere**: the `SEED` env var (default `0`) drives prompt sampling, vLLM generation, every stage's `set_seed`, and the stratified train/valid split — and is the same variable `train.sh` reads, so the full curation+training chain reproduces from a single value.
- Cost sign: higher Beaver cost = more harmful ⇒ `chosen=min`, `rejected=max`.
- Qwen3Guard **Controversial** → treated as **Unsafe**.
- Answers truncated mid-`<think>` (no visible answer) are marked invalid and excluded.
- WildJailbreak and WildGuard are **gated** — accept terms on HF and set `HF_TOKEN`.
