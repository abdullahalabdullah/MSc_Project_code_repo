# harmlessness-exp

Mine **harmlessness preference pairs** from the harmful prompts of `allenai/wildjailbreak`
(the `vanilla_harmful` and `adversarial_harmful` subsets).

For each sampled prompt we generate N Qwen3-4B responses (thinking mode on), keep only
prompts whose responses *disagree* on safety, then use a cost model to pick the safest
response as `chosen` and the most harmful as `rejected`, validated by a safety classifier.

## Pipeline

| Stage | Script | What it does |
|------|--------|--------------|
| 01 | `01_prepare_prompts.py` | Load WildJailbreak `vanilla_harmful` + `adversarial_harmful` (train config), dedup on prompt, report counts + mix. Default `SAMPLE=ALL` keeps **every** unique prompt; `SAMPLE=<int>` (env var) draws a seeded pilot sample |
| 02 | `02_generate_responses.py` (GPU) | Qwen3-4B, N=10, temp 0.6 / top_p 0.95 / top_k 20, `max_tokens=8192`; tuned vLLM engine (prefix caching, capped `max_model_len`, 0.95 VRAM); **live truncation reporting** |
| 03 | `03_classify_guard.py` (GPU) | Split on `</think>`, **Qwen3Guard** label (3-way Safe/Unsafe/Controversial per answer, Controversial folded into Unsafe), **filter-1** (drop prompts whose valid responses share one label). Runs on **vLLM** (greedy); **resumable** — checkpoints every `GUARD_CHECKPOINT_EVERY` pairs to `03_guard_*.ckpt.jsonl`, so an interrupted run continues instead of restarting |
| 04 | `04_score_cost.py` (GPU) | `beaver-7b-unified-cost`; per prompt `chosen=argmin(cost)`, `rejected=argmax(cost)` |
| 05 | `05_report.py` | **filter-3** (chosen=Safe & rejected=Unsafe), survival rate, cost-spread dist, 10 sample pairs, sign-convention sanity check |
| 06 | `06_export_dpo_dataset.py` | **Handoff**: convert `final_pairs.joblib` → a HF `DatasetDict({train,valid})` with `prompt`/`chosen`/`rejected`, saved via `save_to_disk` |

Each stage reads/writes a compressed `.joblib` in `outputs/`, so stages are independent
and resumable.

## Handoff / integration (stage 06)

This module is end-to-end: `run_all.sh` finishes by writing a **training-ready DPO dataset**
that the rest of the project consumes. Stage 06 converts the mined pairs into the standard TRL
preference schema and saves it as a HuggingFace `DatasetDict`:

```
outputs/<run>/dpo_dataset/        # DatasetDict, save_to_disk format
  train/   valid/
```

Schema (both splits): **`prompt`**, **`chosen`**, **`rejected`** (strings).
- `prompt` — the harmful WildJailbreak prompt (the trainer applies its own chat template).
- `chosen` / `rejected` — by default the **full** responses (`<think>…</think>` + answer),
  matching what the model generates. Set `EXPORT_FIELD=answer` to export only the visible
  post-`</think>` answer instead.

Load it in the trainer with:

```python
from datasets import load_from_disk
ds = load_from_disk("DPO-train-dataset-curation/outputs/<run>/dpo_dataset")
train, valid = ds["train"], ds["valid"]   # columns: prompt, chosen, rejected
```

Split is deterministic (`config.SEED`); `config.VALID_FRACTION` (default `0.1`, env-overridable)
sets the valid hold-out, drawn from the final (post-filtering) pairs. By default it is a plain
**seeded random sample** (representative of the training distribution — the right basis for
checkpoint selection); set `STRATIFY_VALID=true` to instead target a **50/50
vanilla/adversarial mix** (best-effort — if one subset runs out the shortfall is filled from
the other). Either way the realised mix is printed and stored in `diagnostics.json` under
`export`. DPO training itself lives in an external trainer (add `trl` to that environment) —
this module stops at the dataset.

## Environment (single venv)

Everything runs in one environment — the Beaver cost model is loaded via the self-contained
`beaver_cost_model.py` (plain `transformers`, no `safe-rlhf`), so there is no dependency
conflict with the Qwen3 / vLLM stack. Dependencies are declared in the repo-root
[`requirements.txt`](../requirements.txt) (single unified environment for the whole project,
curation + training):

```bash
cd ..   # repo root
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd DPO-train-dataset-curation
```

**WildJailbreak is gated** — the `er-abd` account has accepted its terms, and the HF token
ships in the repo-root **committed `.env`** (private repo), so cloning is enough — no token
setup on the pod. On a pod, uncomment `HF_MODELS_CACHE_DIR=/workspace/hf_cache` in `.env`.

## Run

```bash
./run_all.sh            # stages 01-06 (prompts -> generation -> labels -> cost -> pairs -> export)
```

Each `./run_all.sh` writes to a **new** `outputs/run_<timestamp>/` folder (all stages share it
via the exported `RUN_OUTPUT_DIR`), so a new run never overrides an old one. Pin a folder with
`RUN_OUTPUT_DIR=outputs/my_run ./run_all.sh`. The full console output (all stages,
stdout+stderr) is also appended to `<run folder>/run_all.log`, so everything is reviewable
after the terminal is gone.

The Beaver ranker sanity check (`sanity_beaver.py`) is a separate, optional add-on — not run by
`run_all.sh`. Use `run_pilot_and_sanity.sh` to run both together (see below).

`check_max_tokens.py` is a **standalone one-off diagnostic** (imports nothing from the
pipeline — safe to delete): run it on a finished run dir
(`python check_max_tokens.py outputs/<run>`) to judge from the smoke/pilot data whether
`MAX_TOKENS` was too high/low and whether the trainer's `MAX_LENGTH` fits the surviving pairs.

### Two-phase curation: pilot + top-up, then merge

Every run records the exact prompts it consumed (`01_prompts_meta.joblib`). A later run can
**exclude** those prompts (`EXCLUDE_RUNS`, comma-separated run dirs), so the two runs curate
disjoint prompt sets; `merge_runs.py` then combines their final pairs into one
training-ready dataset (dedup on prompt, fresh seeded train/valid split over the merged pool):

```bash
SAMPLE=4500 RUN_OUTPUT_DIR=outputs/pilot ./run_all.sh          # phase 1: ~1k pairs to smoke-test training
SAMPLE=24000 EXCLUDE_RUNS=outputs/pilot RUN_OUTPUT_DIR=outputs/main ./run_all.sh   # phase 2: new prompts only
TARGET_PAIRS=5000 python merge_runs.py outputs/pilot outputs/main --out outputs/merged  # combine + cap -> merged/dpo_dataset
```

`TARGET_PAIRS` (optional) caps the merged pool at an exact size (seeded random downsample
before the stage-06 export): oversample prompts in phase 2 for safety, trim the surplus at
merge time, land on exactly the target dataset size. Unset = keep every merged pair.

Note: the merged split is a *new* split over all pairs — it does not preserve the source
runs' train/valid boundaries (fine when the merged dataset trains from scratch, which is
the standard flow).

Both files this workflow depends on are **git-tracked** (narrow re-includes in
`../.gitignore` override the blanket `*.joblib` ignore): `01_prompts_meta.joblib` (the
prompts a run consumed — what `EXCLUDE_RUNS` reads) and `final_pairs.joblib` (what
`merge_runs.py` reads). So the phase-2 + merge commands above work from a fresh clone on a
new pod. The heavy intermediates (`02_responses.joblib` etc.) stay ignored. The pilot's
copies were recovered after its pod was destroyed: `01_prompts_meta.joblib` regenerated
deterministically (stage 01 is fully seeded; verified against the pilot's subset mix and
final prompts), `final_pairs.joblib` reconstructed from the tracked `dpo_dataset` +
`final_pairs_margins.csv` (pairs carry a `reconstructed: True` flag; full-text fields only,
so re-exporting it with `EXPORT_FIELD=answer` uses answers re-derived by `</think>` split).

To run stages by hand, set the same folder yourself so they share it (otherwise they default
to `./outputs` and would overwrite it):
```bash
export RUN_OUTPUT_DIR=outputs/run_$(date +%Y%m%d_%H%M%S)
python 01_prepare_prompts.py && python 02_generate_responses.py \
  && python 03_classify_guard.py && python 04_score_cost.py \
  && python 05_report.py && python 06_export_dpo_dataset.py
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

Run `SAMPLE=8 ./run_all.sh` to run the pipeline end-to-end on a tiny prompt sample.
Do NOT lower `N` or `MAX_TOKENS` — generation settings are fixed across all runs (smoke,
pilot, full) so that runs stay mergeable/comparable; `SAMPLE` is the only smoke-test knob.
Confirm `outputs/report.md`, `outputs/sample_pairs.json`, and `outputs/cost_spread.png`
are produced and the sign-convention sanity check prints `PASS`.

## Key outputs (`outputs/`)

- `dpo_dataset/` — **the handoff**: HF `DatasetDict({train,valid})` with `prompt`/`chosen`/`rejected`
  (see "Handoff / integration" above). This is what the trainer loads.
- `final_pairs.joblib` — `{prompt, subset, chosen, rejected, chosen_cost, rejected_cost, margin,
  chosen_runner_up_gap, rejected_runner_up_gap, unsafe_frac, chosen_label, rejected_label,
  chosen/rejected_full_tokens, chosen/rejected_answer_words, chosen/rejected_think_words}`
- `report.md` — funnel percentages, survival rate, cost-spread stats, sanity check, plus the
  per-subset funnel, labeler-agreement, margin-calibration, and think-vs-answer sections
- `diagnostics.json` — **consolidated machine-readable metrics**, one section per stage
  (`generation`, `classification`, `scoring`, `funnel`, `per_subset`, `labeler_agreement`,
  `margin_calibration`, `length`). Each stage writes only its own section, so re-running a
  stage refreshes just that slice. This is the single place every stdout stat is also persisted.
- `final_pairs_margins.csv` — per-pair row: subset, costs, margin, runner-up gaps, `unsafe_frac`,
  full-response/answer/think word counts, labels, prompt
- `sample_pairs.json` — 10 random surviving pairs for manual eyeballing
- `cost_spread.png` — histogram of per-prompt cost spread

### Diagnostics recorded (beyond the funnel)

- **Labeler agreement & confidence** (`labeler_agreement`): how often Beaver's safest/harshest
  pick matches WildGuard's Safe/Unsafe verdict (chosen-side, rejected-side, both), and the
  point-biserial r between Beaver cost and the WildGuard label over all scored responses.
- **Margin calibration** (`margin_calibration`): margin mean/median, chosen/rejected
  **runner-up gaps** (how isolated the extreme picks are — a tiny gap means a near-tie),
  per-subset mean margin, and WildGuard decisiveness (mean Unsafe fraction per prompt).
- **Per-subset funnel** (`per_subset`): the vanilla/adversarial sample mix after each filter.
- **Think-block vs answer length** (`length`): chosen/rejected CoT vs visible-answer word counts.

## Conventions / notes

- **Cost sign**: higher Beaver cost = more harmful ⇒ `chosen=min`, `rejected=max`. The local
  loader outputs the raw (pre-normalizer) score; this preserves the argmin/argmax and the
  Unsafe>Safe sanity-check direction (see `beaver_cost_model.py`).
- **Controversial** Qwen3Guard label → treated as **Unsafe** (conservative).
- Responses truncated mid-`<think>` have no visible answer → marked invalid, excluded from candidates.
- Prompts come from WildJailbreak `vanilla_harmful` (the plain request) and
  `adversarial_harmful` (the jailbreak-wrapped request); stage 01 prints the sample's mix.
- Survival is reported **overall**; the survival denominator is the number of prompts that
  entered the pipeline (every unique harmful prompt under the default `SAMPLE=ALL`, or the
  sampled count when `SAMPLE=<int>` is set).
