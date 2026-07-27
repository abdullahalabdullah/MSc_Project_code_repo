# Codebase Structure & Content

**Paper:** *Truthful or Fabricated? Using Causal Attribution to Mitigate Reward Hacking in Explanations* ([OpenReview](https://openreview.net/pdf?id=nkdPLuKoL5))

This repository contains the research code for studying **reward hacking in model explanations** and a mitigation based on **causal attribution**. The core idea: when an LLM is given access to a "shortcut" (here, a math-book solution to the very problem it is asked to solve) but instructed not to use it, it often *uses the shortcut to get the right answer while writing an explanation that hides this fact* — an unfaithful/deceptive explanation that a reward model still scores highly. The repo measures this and trains models (via DPO with **counterfactually-augmented reward signals**) to acknowledge shortcut use instead of hiding it.

> **Scope of this copy:** This is the **Math Book–only** version of the codebase. The BiasQA / Winogender (`wino`) setting from the paper has been removed; all remaining code, data, and scripts concern the Math Book (`aqua`) setting. The standalone `aqua_to_code` evaluation-prompt variant in `eval_responses.py` is retained (it is a math/code variant, not BiasQA).

---

## 1. The Experimental Setting

All experiments run over a single task, identified throughout the code by a short prefix:

| Setting | Code name | Prefix | Dataset | Shortcut being tested |
|---------|-----------|--------|---------|------------------------|
| **Math Book** | `aqua` | `cot` | [`deepmind/aqua_rat`](https://huggingface.co/datasets/deepmind/aqua_rat) | A provided "MATH BOOK" containing the solution to the current question |

Each example exists in two variants:
- **`orig`** — the prompt contains the shortcut (the solution to the current question is present in the book's SOLUTIONS section).
- **`cf`** (counterfactual) — the shortcut is removed (the current question's solution is stripped from the book). Used to causally test whether the model's answer *depended* on the shortcut.

The prompt contains a **negative instruction** ("Do not use the SOLUTIONS part of the MATH BOOK."), so any shortcut use is by definition a violation that the model should acknowledge.

---

## 2. Repository Layout

```
.
├── README.md                  # Reproduction commands (canonical run order)
├── requirements.txt           # Pinned deps (torch 2.5.1, transformers 4.46, trl 0.15.2, vllm 0.6.4)
├── LICENSE
│
├── env_setup.py               # Loads .env, points HF caches at user dirs
├── utils.py                   # set_seed(...) for reproducibility
│
├── data/
│   └── aqua/prepare_data.py   # Dataset construction (orig + cf prompts) → data/aqua/cot_negative.joblib
│
├── get_responses_vllm.py      # Generate model responses (greedy or sampling) with vLLM
├── get_responses_data.py      # Parse responses → structured records (extract predicted answer, validity)
├── eval_responses.py          # LLM-judge: did the explanation acknowledge the shortcut? (pf_ack)
├── get_rewards_data.py        # Score responses with Skywork reward models (orig + heuristic-augmented)
├── eval_rewards.py            # Build counterfactual-augmented reward signal (RM_D / RM_C)
│
├── dpo_prepare_dataset.py     # Turn scored responses into (chosen, rejected) DPO pairs
├── train_dpo.py               # LoRA DPO training + best-checkpoint selection
├── merge_peft_model.py        # Merge LoRA adapter into base model for inference
├── delete_checkpoint.py       # Safe cleanup of merged checkpoints
│
├── analysis/                  # Final metrics, tables (LaTeX) and figures — Math Book only
│   ├── main.py                # Accuracy / acknowledgment, greedy + majority@16 → main figure
│   ├── bon.py                 # Best-of-N curves vs N
│   ├── finegrained.py         # "Deceptive" rate (correct via shortcut, no acknowledgment)
│   └── finegrained_bon.py     # Deceptive rate as a function of N
│
├── motivation_experiment/     # Standalone demo that RMs reward unfaithful explanations (math)
│   ├── prepare_data.py        # Use Llama-3.3-70B to synthesize ack/no-ack × correct/wrong responses
│   ├── gather_scores.py       # Score them with Skywork-Gemma RM
│   ├── plot.ipynb             # Plot the reward gaps
│   └── *.sh                   # SLURM wrappers
│
└── running_scripts/
    └── aqua/                   # SLURM orchestration for the Math Book pipeline
        ├── aqua_greedy_run.sh
        ├── aqua_bon_run.sh / aqua_bon_run_rewards.sh
        ├── aqua_dpo_prepare_dataset.sh
        ├── aqua_dpo_train_rw_{llama,gemma}.sh
        ├── aqua_dpo_eval_responses_{compute,eval}.sh
        └── eval_data/          # Acknowledgment-classifier validation prompts & helpers
```

> **Note:** `outputs/`, `checkpoints/`, `figures/`, and `.env` are not committed (see `.gitignore`); they are produced by the pipeline.

---

## 3. The Three Reward Configurations

A central concept. Responses are scored by a Skywork reward model (`Skywork-Reward-Llama-3.1-8B` → `sk-llama`, or `Skywork-Reward-Gemma-2-27B` → `sk-gemma`). The paper compares three ways of producing the reward used for preference optimization, implemented in `eval_rewards.py`:

| Tag in code | Paper symbol | Meaning |
|-------------|--------------|---------|
| `original`  | `RM`         | Plain reward model score on the response as-is (the baseline that gets hacked). |
| `aug-rm-d`  | `RM_D`       | **Decision-based** causal augmentation. If a sampled counterfactual response changes the *predicted answer* (`predicted_answer` differs), the shortcut was causally used → use the score of a response with an explicit **acknowledgment** appended. |
| `aug-rm-c`  | `RM_C`       | **Correctness-based** causal augmentation. If the orig response is correct but the counterfactual one is wrong, the shortcut was causally necessary → use the acknowledgment-augmented score. |

The "augmented" score comes from `get_rewards_data.py --do_augmented`, which appends a fixed acknowledgment sentence (*"I acknowledge that I used the SOLUTIONS part of the MATH BOOK."*) to the response before scoring. The intuition: reward the model for being honest about shortcut use exactly when causal evidence shows the shortcut mattered.

---

## 4. End-to-End Pipeline

The pipeline is driven by the SLURM scripts in `running_scripts/aqua/`. Each stage writes `.joblib` artifacts that the next stage consumes. Everything is repeated across **3 seeds** (`seed_0..2`).

### Stage 0 — Build data
`data/aqua/prepare_data.py` → constructs paired `orig`/`cf` prompts for train/valid/test and saves a single `.joblib` (`data/aqua/cot_negative.joblib`).

### Stage 1 — Greedy evaluation (baseline numbers)
`aqua_greedy_run.sh`:
1. `get_responses_vllm.py` (1 sample, temperature 0) on test/orig and test/cf.
2. `get_responses_data.py` parses out `predicted_answer`, `predicted_is_correct`, `response_is_valid`.
3. `eval_responses.py` uses **Llama-3.3-70B as an LLM judge** to set `pf_ack` (did the reasoning acknowledge the shortcut?).

### Stage 2 — Sampling / Best-of-N
`aqua_bon_run.sh` (16 samples, temperature 0.8) then `aqua_bon_run_rewards.sh`:
- Scores all samples with both reward models (`get_rewards_data.py`).
- `eval_rewards.py` produces the `aug-rm-d` / `aug-rm-c` variants used for BoN selection analysis.

### Stage 3 — DPO data preparation
`aqua_dpo_prepare_dataset.sh`:
1. Generate 10 responses each for train+valid, orig+cf (`get_responses_vllm.py`).
2. Parse (`get_responses_data.py`), score original + augmented with both RMs (`get_rewards_data.py`).
3. `eval_rewards.py` → six reward configs (`sk-{llama,gemma}_{original,aug-rm-c,aug-rm-d}`).
4. `dpo_prepare_dataset.py` → for each example, pick highest-reward **valid** response as `chosen` and lowest-reward as `rejected`, saving a HF `DatasetDict`.

### Stage 4 — DPO training
`aqua_dpo_train_rw_{llama,gemma}.sh` → `train_dpo.py`:
- LoRA DPO (TRL `DPOTrainer`), 5 epochs, β=0.1, eval/save per epoch.
- `BestCheckpointCallback` tracks the best checkpoint by eval reward accuracy, copies it to `checkpoint-best`, and deletes the rest.
- Base models: `Llama-3.2-3B-Instruct` (`llama3-3B`) or `Llama-3.1-8B-Instruct` (`llama3-8B`).

### Stage 5 — DPO evaluation
`aqua_dpo_eval_responses_compute.sh` then `aqua_dpo_eval_responses_eval.sh`:
- `merge_peft_model.py` merges the LoRA adapter into the base model.
- Decode test set greedy (1) and sampling (16) on orig+cf.
- `delete_checkpoint.py` removes the merged model afterward.
- Parse + LLM-judge acknowledgment, as in Stage 1.

### Stage 6 — Analysis & figures
Run as Python modules (note `python -m analysis.<name> --model_path_name llama3-3B`):
- **`analysis.main`** — Accuracy and acknowledgment rate, for greedy and majority@16, across base + the 3 DPO reward configs × 2 RMs. Emits LaTeX table rows and the main `figures/greedy_sampling_fig_*` plot (a 2×2 Math Book figure).
- **`analysis.bon`** — Metrics vs N ∈ {1,2,4,8,16} for best-of-N reward selection.
- **`analysis.finegrained`** — The headline **deceptive rate**: fraction of examples where the model is correct on orig, wrong on cf (so it causally relied on the shortcut), *and* did not acknowledge it.
- **`analysis.finegrained_bon`** — Deceptive rate as a function of N.

---

## 5. Key Data Records

Most files pass around lists of per-response dicts (originally a `Response` dataclass in `get_responses_data.py`). Fields you'll see throughout:

| Field | Meaning |
|-------|---------|
| `prompt`, `query` | System prompt and user query (shared across the N samples of an example). |
| `full_response` | Raw model generation. |
| `response_is_valid` | Whether an answer could be parsed from the response. |
| `predicted_answer` / `answer` | Parsed prediction vs. ground truth (index). |
| `predicted_is_correct` | `predicted_answer == answer`. |
| `reward` | Reward-model score (added by `get_rewards_data.py`). |
| `pf_ack` | LLM-judge verdict: did the explanation acknowledge shortcut use? (added by `eval_responses.py`). |
| `pf_ack_justification` | The judge's free-text reasoning. |
| `pf_ack_simple` | Cheap keyword heuristic for acknowledgment. |
| `full_response_heuristic` | Response with the canned acknowledgment sentence appended (for augmented scoring). |

Artifact naming convention (example):
`outputs/aqua/llama3-3B/dpo/seed_0/data/cot_negative_orig_train_data_sk-llama_aug-rm-c.joblib`
= setting / base model / stage / seed / `{prefix}_{approach}_{split}_{datatype}_{...}_{rewardmodel}_{rewardconfig}`.

---

## 6. Supporting Files

- **`env_setup.py`** — `load_dotenv()` then points `HF_HOME`/caches at `HF_MODELS_CACHE_DIR` (or the `_TMP` variant when `use_temp=True`). Requires a `.env` with `HF_MODELS_CACHE_DIR`, `HF_MODELS_CACHE_DIR_TMP`, `HF_TOKEN`.
- **`utils.py`** — `set_seed()` seeds Python/NumPy/Torch/Accelerate/Transformers, optionally enabling full determinism.
- **`delete_checkpoint.py`** — Defensive deletion: only removes dirs under `checkpoints/` whose name contains `merged` and that have no nested subfolders.
- **`running_scripts/aqua/eval_data/`** — Helpers to sample model responses and validate the acknowledgment classifier (`aqua_eval_acks_prepare_data.py`, `eval_ack_responses.py` + a labelled JSON).

---

## 7. Models Used

- **Policy models (fine-tuned):** `meta-llama/Llama-3.2-3B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`.
- **Reward models:** `Skywork/Skywork-Reward-Llama-3.1-8B-v0.2`, `Skywork/Skywork-Reward-Gemma-2-27B-v0.2`.
- **Acknowledgment judge & motivation generator:** `meta-llama/Llama-3.3-70B-Instruct`.

---

## 8. Reproducing

See `README.md` for the exact command sequence. Setup is a Python 3.9 virtualenv + `pip install -r requirements.txt` + a `.env` file. The pipeline assumes a SLURM cluster with H100 GPUs (`sbatch …`); the underlying Python scripts can also be invoked directly with the same arguments shown in each `.sh`.
