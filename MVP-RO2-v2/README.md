# MVP-RO2-v2 — Harmlessness DPO data pipeline

This repo trains a **harmlessness DPO** model end to end. It (1) curates harmlessness preference
pairs from `allenai/wildjailbreak` (Qwen3-4B generation → WildGuard safety labels → Beaver cost
ranking → filtered pairs) in the self-contained module
[`DPO-train-dataset-curation/`](DPO-train-dataset-curation/), then (2) LoRA-DPO-trains Qwen3-4B on
the exported pairs.

> Earlier versions of this repo implemented the aqua math-book reward-hacking setting (adapted
> from Reward-Hacking-in-Explanations). That pipeline has been retired; the project now targets
> the harmlessness setting only.

## TL;DR — two commands

```bash
bash DPO-train-dataset-curation/run_all.sh    # 1. curate the DPO dataset (stages 01–06)
bash train.sh                                 # 2. LoRA DPO train Qwen3-4B on the exported pairs
```

Full runbook with all options: **[DOCUMENTATION.md](DOCUMENTATION.md)**.

## Layout

```
DPO-train-dataset-curation/   # end-to-end DPO dataset curation (self-contained module)
train_dpo.py                  # LoRA DPO trainer (TRL DPOTrainer) — self-contained, no sibling imports
train.sh                      # one-command training wrapper (auto-finds the exported dataset)
requirements.txt              # unified environment for the whole repo
DOCUMENTATION.md              # end-to-end runbook (commands + options)
SMOKE_RUN_LOG-template.md     # fill-in log for the small-scale HP-validation run
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The HF token (write access, `er-abd` account) ships in the **committed `.env`** (this repo is
private — see the note inside `.env` before ever making it public), so no token setup is
needed after cloning. On a pod, uncomment `HF_MODELS_CACHE_DIR=/workspace/hf_cache` in `.env`
so model caches land on the volume disk.

Requires a CUDA GPU (vLLM generation, Beaver/WildGuard scoring, DPO training).

## 1. Curate the DPO dataset

```bash
bash DPO-train-dataset-curation/run_all.sh    # stages 01–06; writes outputs/run_<timestamp>/
```

By default this curates **ALL** unique WildJailbreak harmful prompts (`vanilla_harmful` +
`adversarial_harmful`, train config); set `SAMPLE=<int>` for a seeded pilot. The handoff is
`DPO-train-dataset-curation/outputs/<run>/dpo_dataset/` — a HuggingFace
`DatasetDict({train,valid})` with columns `prompt` / `chosen` / `rejected` (valid holdout: a
seeded random 10% of the final pairs; `STRATIFY_VALID=true` for a 50/50
vanilla/adversarial mix instead) — saved locally so future
training runs reuse it without re-running curation. Stats land in the run folder's
`report.md` + `diagnostics.json`. See the module's
[README](DPO-train-dataset-curation/README.md) for the full pipeline, config knobs, and schema.

## 2. Train

```bash
bash train.sh                 # trains Qwen3-4B on the newest exported dataset
# overrides: bash train.sh path/to/dpo_dataset  |  EPOCHS=5 MODEL=Qwen/Qwen3-4B bash train.sh
```

LoRA adapters are written under `checkpoints/qwen3-4B/harmless_dpo/` (best kept as
`checkpoint-best`). Every checkpoint (step 0, each ~20% interval, final step) is **always pushed**
to its own private HF repo `<HUB_MODEL_ID>-step-<N>` (`HUB_MODEL_ID` defaults to
`er-abd/harmless-dpo-qwen3-4B`). Each eval point also appends to
`eval_metrics.jsonl` (DPO loss / reward-accuracy curves for checkpoint selection; behavioural
safety + monitorability eval is run separately, offline, on the pushed checkpoints). For
inference, load the adapter on top of the base Qwen3-4B (transformers `PeftModel`, or vLLM
`--enable-lora`). See [DOCUMENTATION.md](DOCUMENTATION.md) for details.

---

## Attribution & Provenance

Parts of this repo are **adapted from** the official code release accompanying:

> *Truthful or Fabricated? Using Causal Attribution to Mitigate Reward Hacking in Explanations* — Ferreira et al.
> Paper: https://openreview.net/pdf?id=nkdPLuKoL5
> Source code: https://github.com/PedroMLF/Reward-Hacking-in-Explanations
> Licensed under the **MIT License** (© 2025 Pedro F.) — see [`LICENSE`](LICENSE).

The upstream code is reused under its MIT License; the original copyright and permission notice
are retained in `LICENSE`. The files still derived from the upstream repo — `train_dpo.py` and the
`utils.py` / `env_setup.py` inside `DPO-train-dataset-curation/` (clean local copies) — each carry
a per-file attribution header.
`train_dpo.py` was adapted to the harmlessness dataset schema (`prompt`/`chosen`/`rejected`),
Qwen3, and optional Hub pushing, and is self-contained (its seeding/env helpers are inlined, no
sibling imports). The harmlessness curation module and its stages are our own work for this MSc
project.
