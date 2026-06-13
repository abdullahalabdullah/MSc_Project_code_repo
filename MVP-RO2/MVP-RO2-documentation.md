# MVP-RO2 — Code Documentation (Training Side)

This document explains the code implemented for the **RO2 MVP** (the effect of
post-training on CoT monitorability). It covers **the training side only**: one
DPO LoRA model trained on the harmlessness objective (PKU-SafeRLHF conflict
cases), starting from `allenai/Llama-3.1-Tulu-3-8B-SFT-no-safety-data`, with the
selected adapter pushed to the Hugging Face Hub.

The spec is `MVP_RO2.md` in this folder; this file documents what was actually
built and how to run it.

---

## 1. What was implemented

| File | Purpose |
|---|---|
| `configs/harmlessness.yaml` | Full training config — the whole run is described by this file. |
| `configs/harmlessness_sanity.yaml` | Overrides for the 100-example dry run (merged on top of the full config). |
| `train_dpo.py` | Main training script. All logic lives in importable functions; `__main__` is a thin wrapper. |
| `sanity_check_dpo.py` | Separate entry point that reuses `train_dpo.py`'s functions on a tiny slice and runs the spec's inspection checks. |
| `train_harmlessness.sh` | SLURM batch script for the full run. |
| `requirements.txt` | Pinned dependencies for the `ro2_env` environment (separate from MVP-RO1). |
| `.env.example` | Template for credentials (`HF_TOKEN`, `OPENAI_API_KEY`). Copy to `.env`. |
| `.gitignore` | Ignores `.env`, `checkpoints/`, SLURM logs, caches. |

### Scope note — what is deliberately NOT here
- **Capability check** (pre/post MMLU accuracy → `capability_check.json`) — deferred.
- **Before/after analysis notebook** (`analysis/ro2_mvp_comparison.ipynb`) — deferred.
- **The g-mean² eval pipeline** (`01–05`) — out of scope; the spec marks it "no
  coding required" because it reuses the existing MVP-RO1 scripts unchanged.

These are noted as follow-ups in §6.

---

## 2. Code structure and data flow

`train_dpo.py` is organised as a library of functions, orchestrated by
`run_training()`. `sanity_check_dpo.py` imports those same functions so the
sanity path is identical to the real path (just smaller and step-capped).

```
run_training(cfg, hub_id, logger)
  ├─ check_credentials(logger)              # load .env; fail fast if HF_TOKEN missing
  ├─ load_model_and_tokenizer(cfg)          # Unsloth: base in bf16 + LoRA adapters
  ├─ load_and_prepare_dataset(cfg, tok)     # PKU-SafeRLHF -> conflict -> subsample -> (prompt,chosen,rejected) -> split
  ├─ inspect_formatting(train, n=8)         # print templated triples (manual format check)
  ├─ check_prompt_lengths(train, tok, 512)  # length distribution vs max_prompt_length
  ├─ build_trainer(...)                      # PatchDPOTrainer() + DPOConfig + DPOTrainer(ref_model=None)
  ├─ trainer.train()
  ├─ select_best_checkpoint(log_history)    # max acc -> min loss -> max step
  └─ push_selected_adapter(...)             # reload selected adapter, push (adapter-only) + model card
```

### Dataset preparation (`load_and_prepare_dataset`)
- Loads `PKU-Alignment/PKU-SafeRLHF` (`train` split) and verifies the expected
  columns (`prompt`, `response_0/1`, `is_response_0/1_safe`,
  `better_response_id`, `safer_response_id`).
- **Conflict filtering.** "Conflict case" = the safer response is not the more
  helpful one. There are two reasonable operationalisations and the spec mentions
  both, so the code **logs the count under both** and uses the one set by
  `conflict_filter_mode`:
  - `better_ne_safer` (default): `better_response_id != safer_response_id` —
    matches the spec's prose and cleanly isolates harmlessness from helpfulness.
  - `safe_xor`: `is_response_0_safe != is_response_1_safe` — the spec's literal
    field suggestion.
  The expected pool size is ~12,616; the code logs the actual count and does
  **not** hard-fail if it differs (per spec). Look at the two logged counts after
  the sanity run to confirm which definition lands near 12,616 and adjust
  `conflict_filter_mode` if needed.
- **Subsample** the filtered pool to `subsample_size` (1,500) with
  `subsample_seed` (7).
- **Format** each row to TRL DPO `(prompt, chosen, rejected)`:
  - `prompt` = user turn via `tokenizer.apply_chat_template(..., add_generation_prompt=True)`.
  - `chosen` = the **safer** response (`response_{safer_response_id}`), raw text only.
  - `rejected` = the other response, raw text only.
  - No EOS is appended manually — TRL 0.9.6's `DPOTrainer.tokenize_row` adds it,
    and keeping the strings role-marker-free is exactly what the spec's format
    check requires.
- **Split** into train / `val_size` (300) validation with `random_seed` (42).

### Model (`load_model_and_tokenizer`)
- `unsloth` is imported **first** in `train_dpo.py` (before transformers/trl/peft)
  so its kernels and the DPO reference-model patch apply.
- Base model loaded in **bf16**, **no quantisation** (`load_in_4bit=False`) — the
  A6000's 48 GB is enough, and quantisation would be a methodological confound.
- LoRA via `FastLanguageModel.get_peft_model` (r=16, alpha=32, dropout=0.05, the
  7 target modules, `use_gradient_checkpointing="unsloth"`).

### Trainer (`build_trainer`)
- `PatchDPOTrainer()` is called before constructing the trainer (enables the
  Unsloth reference-model trick → `ref_model=None`, saving ~16 GB).
- All DPO hyperparameters live in `DPOConfig` (TRL 0.9.6 API). `DPOTrainer`
  receives `tokenizer=` (not the newer `processing_class=`), the model (already
  PEFT, so no `peft_config`), and `ref_model=None`.
- `save_total_limit=None` keeps every checkpoint; `load_best_model_at_end=False`
  because selection is custom (below). `optim="paged_adamw_32bit"` gives paged
  AdamW with fp32 optimizer states.

### Checkpoint selection (`select_best_checkpoint`)
Implements the Ferreira et al. (2025) Appendix B.2 rule, automatically, from
`trainer.state.log_history`:
1. max `eval_rewards/accuracies`;
2. among ties (|Δ| < 1e-9), min `eval_loss`;
3. still tied → highest step (latest / most-trained).
The full selection table (step, accuracy, loss, selected y/n) is logged.

### Pushing the adapter (`push_selected_adapter`)
- **Adapter-only** (your decision): the selected `checkpoint-<best_step>/` is
  reloaded and `model.push_to_hub(hub_id, private=True)` +
  `tokenizer.push_to_hub(...)` upload the LoRA adapter + tokenizer (not a merged
  model). An auto-generated `README.md` model card records base model, dataset +
  filtered count, selected step, `eval_rewards/accuracies`, `eval_loss`, and date.
- The pushed repo id and commit hash are logged.
- **Fallback:** the push is wrapped in `try/except`. On any failure it saves the
  adapter + tokenizer to `checkpoints/harmlessness_dpo/best_checkpoint_<step>_PUSH_FAILED/`
  and logs a loud `ERROR` with full traceback, so the trained adapter is never
  left only as a numbered checkpoint folder.

---

## 3. The sanity check (`sanity_check_dpo.py`)

This is the **gate** before the multi-hour run. The pinned stack (trl 0.9.6,
transformers 4.44.0, peft 0.12.0, unsloth) is **not yet proven** on the cluster;
if it is broken (e.g. an Unsloth/TRL incompatibility), this script fails here,
cheaply and loudly, instead of hours into the real job.

It loads `harmlessness.yaml` overlaid with `harmlessness_sanity.yaml`
(subsample 120, val 20, `max_steps=50`, `eval_steps=20`), runs the **same**
`run_training()` path, pushes to the **throwaway** repo (`sanity_hub_model_id`),
then asserts the programmatic subset of the spec's 8 checks and prints
`PASS / FAIL / MANUAL` for each:

1. Chat-template formatting triples — printed (MANUAL eyeball).
2. Prompt-length distribution — printed + guarded.
3. Training loss finite and not flat-zero — asserted (+ curve printed to eyeball).
4. ≥2 intermediate eval events, each with loss + accuracy — asserted.
5. eval accuracies ∈ [0,1], finite, not identical across evals; eval_loss finite — asserted.
6. Checkpoint selection picked a step — asserted.
7. GPU peak memory under ~35–37 GB — MANUAL (`watch nvidia-smi`).
8. Adapter push to the throwaway repo succeeded — asserted.

The process exits non-zero if any asserted check fails.

---

## 4. Before you run — checklist

1. **Create the environment** (on the cluster):
   ```bash
   conda create -n ro2_env python=3.11 -y
   conda activate ro2_env
   pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt
   ```
   > The pinned versions may not resolve cleanly together (TRL/PEFT/Unsloth move
   > fast). This is expected to be the fiddly part — the sanity check is what
   > confirms the env actually works. There is **no** non-Unsloth fallback.

2. **Credentials:** copy `.env.example` to `.env` and fill in real values.
   `.env` is git-ignored — never commit it.
   ```bash
   cp .env.example .env   # then edit HF_TOKEN (write access) and OPENAI_API_KEY
   ```

3. **Set your real HF username** in `configs/harmlessness.yaml` — replace
   `YOUR_HF_USERNAME` in **both** `hub_model_id` and `sanity_hub_model_id`.

4. **Accept the PKU-SafeRLHF terms** on its Hugging Face dataset page while
   logged in as the account behind `HF_TOKEN` (the dataset is gated).

5. **Hub repos:** `push_to_hub`/`create_repo` will create the private repos
   automatically; just make sure `HF_TOKEN` has **write** scope.

6. **(Optional) confirm the conflict count.** After the sanity run, read the
   logged `better_ne_safer` vs `safe_xor` counts and confirm the one near
   ~12,616 matches `conflict_filter_mode`. Adjust the config if needed before the
   full run.

---

## 5. Commands to run

### Sanity check (do this first — ~minutes)
```bash
# Allocate a GPU interactively
srun -p Teaching --gres=gpu:nvidia_rtx_a6000:1 --pty bash
conda activate ro2_env
cd MVP-RO2

python sanity_check_dpo.py
```
Watch `nvidia-smi` in another shell for the peak-VRAM check. Only proceed once
all programmatic checks report `PASS` and the MANUAL items look right.

### Full training run (~3–6 hours)
```bash
cd MVP-RO2
sbatch train_harmlessness.sh
```
Monitor:
```bash
squeue -u $USER
tail -f slurm-<jobid>.out
```
On success, the selected adapter is at `https://huggingface.co/<username>/tulu3-8b-harmlessness-dpo-mvp`
(private), and the slurm log records the selected step, its metrics, and the
pushed commit hash.

---

## 6. Follow-ups (deferred, not in this deliverable)

- **Merge the adapter for eval.** The MVP-RO1 eval pipeline loads models via
  **vLLM** (`MVP-RO1/src/inference.py`, `model=<hf_id>`), which does not load an
  adapter-only repo directly. Per your decision, merging is done **manually
  later** (e.g. load base + adapter with PEFT, `merge_and_unload()`, push a
  merged 16-bit repo, then point a new `models:` entry in the RO1 config at it).
- **Capability check** (`capability_check.py`): pre/post MMLU 0-shot accuracy on
  the 200 `dataset_index.json` questions (reuse `build_user_content` and
  `parse_answer` from `MVP-RO1/src/inference.py`), written to
  `capability_check.json`.
- **Analysis notebook** (`analysis/ro2_mvp_comparison.ipynb`): the before/after
  table, shared contributing-set comparison, per-instance ΔTE, confusion
  matrices, and qualitative Y=1/Z=0 samples.

---

## 7. Key implementation decisions (and why)

- **Strict Unsloth, no fallback** — per your instruction. The sanity check is the
  environment gate.
- **Adapter-only push** — per your instruction; you merge manually later.
- **TRL 0.9.6 API targeted** (not the newer API in the Ferreira reference repo):
  DPO params in `DPOConfig`, `DPOTrainer(tokenizer=...)`, `eval_strategy` (valid
  in transformers 4.44). Only *patterns* were reused from the reference repo
  (tie-break, chosen/rejected selection), not its code verbatim.
- **No manual EOS** on `chosen`/`rejected` — TRL appends it; this also satisfies
  the spec's requirement that those fields carry no role markers.
- **Conflict filter logs both definitions** — the ~12,616 target is the
  disambiguator; the code never hard-fails on the count.
- **`unsloth` imported first** — required for its patches to take effect.
