# MVP-RO2-v2 — DPO training pipelines (harmlessness + anti-sycophancy)

This repo trains DPO models end to end for two separate objectives, sharing ONE trainer:

1. **Harmlessness** — curates preference pairs from `allenai/wildjailbreak` (Qwen3-4B
   generation → Qwen3Guard safety labels → Beaver cost ranking → filtered pairs) in
   [`harmlessness-DPO-train-dataset-curation/`](harmlessness-DPO-train-dataset-curation/).
2. **Anti-sycophancy** — curates preference pairs from TriviaQA (knowledge filter →
   LLM-generated wrong answers → opinionated templates → contrast sampling → balanced
   pairs + sham control) in
   [`anti-syc-DPO-train-dataset-curation/`](anti-syc-DPO-train-dataset-curation/), per the
   frozen spec `../anti_sycophancy_dpo_dataset_curation_spec.md`.

Both modules export the identical training contract — a HuggingFace
`DatasetDict({train,valid})` with `prompt`/`chosen`/`rejected` string columns — so
`train_dpo.py` is objective-agnostic and unchanged across objectives.

> Earlier versions of this repo implemented the aqua math-book reward-hacking setting
> (adapted from Reward-Hacking-in-Explanations). That pipeline has been retired.

## TL;DR

```bash
# Harmlessness:
bash harmlessness-DPO-train-dataset-curation/run_all.sh   # curate (stages 01–06)
bash train.sh                                             # LoRA DPO train on the export
METHOD=dpo_norm bash train.sh <dpo_dataset>               # or length-normalized DPO (open-instruct dpo_norm)

# Anti-sycophancy (see the module README for the split GPU/API phase drivers):
X=5 RUN_OUTPUT_DIR=outputs/smoke bash anti-syc-DPO-train-dataset-curation/run_all.sh
bash train_antisyc.sh      anti-syc-DPO-train-dataset-curation/outputs/smoke/dpo_dataset
bash train_antisyc_sham.sh anti-syc-DPO-train-dataset-curation/outputs/smoke/dpo_dataset_sham
```

Full runbook with all options: **[DOCUMENTATION.md](DOCUMENTATION.md)**.

## Layout

```
harmlessness-DPO-train-dataset-curation/   # harmlessness dataset curation (self-contained)
anti-syc-DPO-train-dataset-curation/       # anti-sycophancy dataset curation (self-contained;
                                           #   also produces the RO1 eval's frozen reserved/ files)
train_dpo.py                  # LoRA DPO trainer (TRL DPOTrainer) — objective-agnostic
train_dpo_norm.py             # LoRA length-normalized DPO (DPOTrainer subclass, avg log-probs; open-instruct dpo_norm)
train.sh                      # one-command training wrapper (harmlessness defaults; METHOD=dpo|dpo_norm)
train_antisyc.sh              # anti-syc wrapper: pins output dir / HF repo / W&B names
train_antisyc_sham.sh         # sham-control wrapper (destroyed-preference control run)
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
private — see the note inside `.env` before ever making it public). The anti-sycophancy
module's API stages additionally need `OPENAI_API_KEY` in `.env` (see the module's
`.env.example`). On a pod, uncomment `HF_MODELS_CACHE_DIR=/workspace/hf_cache` in `.env`
so model caches land on the volume disk.

GPU needs: harmlessness curation and training require CUDA. The anti-sycophancy module
needs a GPU **only** for its two generation stages (03/07) — grading/analysis run on any
machine with the OpenAI key (see the module README's phase drivers).

## 1. Curate a DPO dataset

**Harmlessness** — `bash harmlessness-DPO-train-dataset-curation/run_all.sh` (stages
01–06; `SAMPLE=<int>` for a seeded pilot). Handoff:
`harmlessness-DPO-train-dataset-curation/outputs/<run>/dpo_dataset/`. See the module
[README](harmlessness-DPO-train-dataset-curation/README.md).

**Anti-sycophancy** — smoke: `X=5 RUN_OUTPUT_DIR=outputs/smoke bash
anti-syc-DPO-train-dataset-curation/run_all.sh`; pilot (X=500, blocking gates incl. the
100-grade hand audit) and full run (X=20000) via the four `run_phase{1..4}_{gpu,api}.sh`
drivers. Handoff: `outputs/<run>/dpo_dataset/` + `outputs/<run>/dpo_dataset_sham/`, plus
the frozen RO1 eval files in `outputs/<run>/reserved/`. See the module
[README](anti-syc-DPO-train-dataset-curation/README.md).

## 2. Train

```bash
bash train.sh                                   # harmlessness DPO (newest harmlessness export)
METHOD=dpo_norm bash train.sh <path/to/dpo_dataset>     # harmlessness length-normalized DPO
bash train_antisyc.sh      <path/to/dpo_dataset>        # anti-sycophancy
bash train_antisyc_sham.sh <path/to/dpo_dataset_sham>   # sham control
```

**DPO vs length-normalized DPO** (`METHOD`, harmlessness): `METHOD=dpo` (default) runs standard
DPO (`train_dpo.py`, summed log-probs); `METHOD=dpo_norm` runs length-normalized DPO
(`train_dpo_norm.py`, averaged log-probs — open-instruct's `dpo_norm`, still reference-based
sigmoid DPO), the fix for DPO's response-length inflation (see
`RO1/notes/harmless_dpo_response_length_inflation.md`). It uses separate defaults
(`OUTPUT_DIR=checkpoints/qwen3-4B/harmless_dpo_norm`,
`HUB_MODEL_ID=er-abd/harmless-dpo-norm-qwen3-4B`) and knob `DPO_NORM_BETA` (default 5).
`train_dpo.py` is unchanged; `dpo_norm` reuses its machinery via a `DPOTrainer` subclass.

LoRA adapters land under `checkpoints/qwen3-4B/{harmless_dpo,harmless_dpo_norm,anti_syc_dpo,anti_syc_dpo_sham}/`
(best kept as `checkpoint-best`). Every checkpoint (step 0, each ~20% interval, final step)
is **always pushed** to its own private HF repo `<HUB_MODEL_ID>-step-<N>` — HUB_MODEL_ID
defaults: `er-abd/harmless-dpo-qwen3-4B` (train.sh; `er-abd/harmless-dpo-norm-qwen3-4B` for
`METHOD=dpo_norm`), `er-abd/anti-syc-dpo-qwen3-4B` /
`er-abd/anti-syc-dpo-sham-qwen3-4B` (the wrappers — never train anti-syc through bare
train.sh, or checkpoints land in the harmlessness repos). Each eval point appends to
`eval_metrics.jsonl`; behavioural evals (harmlessness / anti-sycophancy / monitorability)
run separately in RO1 on the pushed checkpoints. For inference, load the adapter on top of
base Qwen3-4B (transformers `PeftModel`, or vLLM `--enable-lora`), or use the RO1 evals'
`--merge` flag. See [DOCUMENTATION.md](DOCUMENTATION.md).

---

## Attribution & Provenance

Parts of this repo are **adapted from** the official code release accompanying:

> *Truthful or Fabricated? Using Causal Attribution to Mitigate Reward Hacking in Explanations* — Ferreira et al.
> Paper: https://openreview.net/pdf?id=nkdPLuKoL5
> Source code: https://github.com/PedroMLF/Reward-Hacking-in-Explanations
> Licensed under the **MIT License** (© 2025 Pedro F.) — see [`LICENSE`](LICENSE).

The upstream code is reused under its MIT License; the original copyright and permission notice
are retained in `LICENSE`. The files still derived from the upstream repo — `train_dpo.py` and the
`utils.py` / `env_setup.py` inside the curation modules (clean local copies) — each carry
a per-file attribution header.
`train_dpo.py` was adapted to the preference dataset schema (`prompt`/`chosen`/`rejected`),
Qwen3, and optional Hub pushing, and is self-contained (its seeding/env helpers are inlined, no
sibling imports). `train_dpo_norm.py` (length-normalized DPO) is our own addition — it reuses
`train_dpo.py`'s helpers/callbacks by import and subclasses TRL's `DPOTrainer` to length-average
the completion log-probs, following AllenAI **open-instruct**'s `dpo_norm`
([`open_instruct/dpo_utils.py`](https://github.com/allenai/open-instruct/blob/main/open_instruct/dpo_utils.py),
MIT-licensed), acknowledged in that file's header. The curation modules and their stages are our own work for this MSc project.
The anti-sycophancy eval data derives from TriviaQA (Joshi et al., 2017) and SycophancyEval
(Sharma et al., 2023, github.com/meg-tong/sycophancy-eval).
