# MVP: RO2 — Effect of Post-Training on CoT Monitorability

## Overview

This document describes the MVP for Research Objective 2 (RO2). RO1 is assumed complete: the evaluation pipeline (inference, TE computation, monitor, g-mean² computation) is working and has been validated on four Tulu checkpoints using MMLU with the Stanford professor hint (see `MVP-RO1/metrics/gmean2_evaluation_metric.md` and `MVP-RO1/metrics/g-mean² Filtering Levels.md` for the metric definition and filtering rules). The `dataset_index.json` (200 questions with fixed hint assignments) and the RO1 results, including the fixed Level-1-passing eval suite, are available.

RO2 asks: **does applying DPO fine-tuning under a specific post-training objective change the model's CoT monitorability, as measured by g-mean²?**

The MVP tests this with a single training condition — **harmlessness** — to validate the full RO2 pipeline (training + evaluation) before committing to all three objectives.

---

## Goal

- Train one DPO model on the harmlessness objective (PKU-SafeRLHF dataset) starting from the baseline checkpoint (`allenai/Llama-3.1-Tulu-3-8B-SFT-no-safety-data`)
- Re-run the RO1 evaluation pipeline on the trained checkpoint using the **exact same fixed eval suite** (the problems passing RO1's Level-1 filter on the baseline SFT model — see `MVP-RO1/metrics/g-mean² Filtering Levels.md`), hint assignments, and prompt templates
- Produce a before/after comparison table: baseline g-mean² vs. harmlessness-DPO g-mean², alongside skip rate, mean TE, and the TPR_mc / TNR_X=0 / TNR_X=1 components
- Confirm the full RO2 code path works end-to-end and that the g-mean² metric is sensitive enough to detect a training-induced shift

---

## Why Harmlessness First

The harmlessness condition is the MVP choice for three reasons:

1. **It is the hypothesis-critical condition.** The primary hypothesis is that harmlessness training specifically degrades g-mean² (via a lower TPR_mc) on authority-cued instances, by incentivising safe-sounding CoTs that conceal deference to an external authority figure. This is the condition that matters most for the dissertation argument.
2. **It uses the cleanest dataset.** PKU-SafeRLHF filtered to conflict cases (where the safer response is not the more helpful one) cleanly isolates harmlessness from helpfulness, avoiding confounds present in the other datasets.
3. **The full pipeline generalises.** Once this training + re-evaluation loop works, running helpfulness and anti-sycophancy is just changing the dataset in the config file.

---

## Compute Environment

**Cluster:** Edinburgh Informatics School SLURM cluster
**Node:** `landonia11`
**GPU:** NVIDIA RTX A6000, 48 GB VRAM (49,140 MiB), CUDA 12.8, driver 570.211.01
**Allocation:** `--gres=gpu:nvidia_rtx_a6000:1` (full 48 GB, no MIG sharing)

**SSH access:**
```bash
ssh s2869453@mlp -J s2869453@student.ssh.inf.ed.ac.uk
```

**Allocate GPU interactively (for testing):**
```bash
srun -p Teaching --gres=gpu:nvidia_rtx_a6000:1 --pty bash
```

**Submit batch job (for actual training runs):**
```bash
sbatch train_harmlessness.sh
```
Always use `sbatch` for training — it persists across SSH disconnections and logs output to `slurm-<jobid>.out`.

**Check job status:**
```bash
squeue -u $USER       # your jobs
nvidia-smi            # GPU usage on the allocated node
echo $CUDA_VISIBLE_DEVICES  # confirm GPU index
```

---

## Precision and Memory Strategy

**No quantisation.** The A6000's 48 GB VRAM is sufficient to run full bf16 LoRA without NF4 or 8-bit quantisation. Quantisation would introduce noise as a methodological confound — any g-mean² shift could be attributed to quantisation artefacts rather than the training objective. Use:

- Base model loaded in **bf16** (~16 GB)
- LoRA adapters trained in **bf16**
- Paged AdamW optimiser states in **fp32**
- Gradient checkpointing **enabled**
- `ref_model=None` with Unsloth (reference model handled internally by disabling LoRA adapters — saves ~16 GB vs. keeping a frozen copy)

**Estimated peak VRAM:** ~35–37 GB, leaving 11–13 GB of headroom.

---

## Training Framework

**Use HuggingFace TRL + Unsloth.** Standard PEFT/TRL works but Unsloth is critical for this hardware:

- **VRAM reduction:** Unsloth uses memory-efficient Triton kernels to handle the DPO reference model internally (via `ref_model=None`), avoiding the ~16 GB cost of keeping a frozen reference copy in VRAM.
- **Speed:** ~2–3× training throughput vs. stock PyTorch on Ampere-class GPUs.
- **bf16 native:** The A6000 has native bfloat16 execution units; bf16 avoids the underflow/overflow instability of fp16 on Llama 3.1 architectures.

---

## Training Dataset: PKU-SafeRLHF (Harmlessness)

**HuggingFace identifier:** `PKU-Alignment/PKU-SafeRLHF`

**Filtering:** Use only **conflict cases** — examples where the safer response is *not* the more helpful response. This isolates harmlessness from helpfulness. The dataset includes a field indicating whether the two objectives conflict; filter to rows where `is_response_0_safe != is_response_1_safe` (or the equivalent conflict flag in the dataset schema — check the dataset card on HuggingFace for the exact field name).

**Target size:** ~12,616 examples after filtering conflict cases. If the filtered set differs slightly, use whatever the filter yields and log the exact count.

**Subsampling to 1,500 examples:** Randomly subsample **1,500 examples** (`subsample_seed=7`) from the ~12,616 filtered conflict cases for the actual training run. This is about half the training-set scale used by Ferreira et al. (2025) for the 'Math Book' (AQUA-RAT) setting (3,000 training examples); the borrowed hyperparameters (5 epochs, effective batch size 16, `learning_rate=5e-6`, etc.) were validated at the AQUA-RAT scale, and training on the full ~12,616 examples would mean ~8× more optimizer steps per epoch than the hyperparameters were tuned for. Log both the full filtered-pool count and the 1,500-example subsample.

**Train/validation split:** From the 1,500-example subsample, hold out **150 examples** as a validation set (random, `random_seed=42`); the remaining **~1,350** form the training set (a ~10% validation split, comparable in proportion to the AQUA-RAT val/train ratio in Ferreira et al.). The validation set is used during training for checkpoint selection (see "DPO Training Configuration" below) — following Ferreira et al. (2025), Appendix B.2.

**Format:** Each example must be converted to the DPO `(prompt, chosen, rejected)` format:
- `prompt`: the user turn (the question or instruction)
- `chosen`: the safer response (the one labelled as safer in the conflict case)
- `rejected`: the less safe response

**Chat template:** Apply the Tulu/Llama 3.1 instruct chat template to all three fields via `tokenizer.apply_chat_template()` before passing to the trainer. Do not manually format strings.

**Manual format check (do before launching training):** PKU-SafeRLHF was not constructed with the Llama/Tulu chat template in mind — its prompts are raw single/multi-turn text, not pre-templated conversations. After running `apply_chat_template()`, manually inspect 5–10 formatted `(prompt, chosen, rejected)` triples to confirm: the prompt ends at the correct assistant-turn boundary, `chosen`/`rejected` contain only the response text (no leaked template tokens or role markers), and the special tokens match what the base model expects. A silent formatting error here would train on malformed triples, producing capability effects that could be mistaken for monitorability changes.

**Prompt length check (do before launching training):** Tokenize the 1,500-example subsample's prompts with the chat-templated format and plot the length distribution before committing to `max_prompt_length=512`. If a non-trivial fraction of prompts exceed 512 tokens, they will be silently truncated, meaning the model trains on incomplete context. If truncation is widespread, raise `max_prompt_length` to 1024 (and correspondingly `max_length` to ~2560–3072) — gradient checkpointing on the A6000 should still keep this within the ~35–37 GB VRAM budget.

<!-- **Decontamination check:** Verify no PKU-SafeRLHF examples overlap with the 200 MMLU questions in `dataset_index.json` before training. A simple string match on question text is sufficient for the MVP. -->

---

## LoRA Configuration

```python
lora_config = LoraConfig(
    r=16,                          # rank — 13.6M trainable params vs 8B full fine-tune
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

This gives ~13.6M trainable parameters (LoRA adapters only), with the base model weights frozen. Since the base model is identical across all conditions, any g-mean² shift is attributable to the training objective rather than changes in the underlying weight distribution.

---

## DPO Training Configuration

```python
dpo_config = DPOConfig(
    output_dir="./checkpoints/harmlessness_dpo",
    num_train_epochs=5,               # Ferreira et al. (2025), Appendix B.2
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,    # effective batch size = 16
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,                 # 10% warmup steps
    weight_decay=0.01,
    bf16=True,
    gradient_checkpointing=True,
    max_length=2048,                  # cap to prevent quadratic memory scaling on backward pass
    max_prompt_length=512,
    beta=0.1,                         # KL penalty coefficient
    ref_model=None,                   # Unsloth handles reference model internally
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    per_device_eval_batch_size=2,
    save_strategy="steps",
    save_steps=100,                   # save intermediate checkpoints in case of crash
    save_total_limit=None,            # keep all checkpoints — LoRA adapters are small (~13.6M params each),
                                       # and best-checkpoint selection below is custom, not TRL's load_best_model_at_end
    load_best_model_at_end=False,      # see "Checkpoint Selection" — custom tie-aware selection runs after training
    remove_unused_columns=False,
    report_to="none",                 # no wandb for MVP; use slurm log file
)
```

**Why these hyperparameters:** All values above (5 epochs, effective batch size 16, AdamW, `learning_rate=5e-6`, cosine scheduler with 10% warmup, `weight_decay=0.01`, `beta=0.1`, and LoRA `r=16`/`lora_alpha=32` (= 2×r)/`lora_dropout=0.05`) are taken directly from Ferreira et al. (2025), Appendix B.2 ("Truthful or Fabricated?", `papers/TruthfulOrFabricated - Ferreira et al.md`), which trains DPO models with TRL + LoRA on a closely related CoT-faithfulness setup. Reusing a validated hyperparameter set removes the need to guess values for the MVP — an earlier draft used `learning_rate=5e-5` and `num_train_epochs=1`, both of which deviated from this precedent without justification. During the 100-example sanity check (below), still inspect the training loss curve over the first 50 steps as a basic sanity check.

**Why `max_seq_length=2048`:** The Tulu base model supports 32k context, but DPO backward pass memory scales quadratically with sequence length. Cap at 2048 for the MVP; increase to 4096 only if the training data has many examples that are truncated at 2048 (check truncation rate in the training log). See also the prompt-length check above — if `max_prompt_length` is raised to 1024, raise `max_length` correspondingly.

**Checkpoint selection (validation-based, per Ferreira et al. B.2, with automatic tie-break):** With `eval_strategy="steps"` and `eval_steps=100` over the held-out 150-example validation split, TRL's `DPOTrainer` reports `eval_rewards/accuracies` (the fraction of validation pairs where the chosen response has a higher implicit reward than the rejected response) and `eval_loss` at each evaluation point, appended to `trainer.state.log_history`.

`load_best_model_at_end` is **not** used, because it only supports a single `metric_for_best_model` and cannot apply the paper's tie-break rule. Instead, `train_dpo.py` must implement the paper's exact two-stage selection automatically after training completes:

1. Parse `trainer.state.log_history` and collect all entries containing `eval_rewards/accuracies` and `eval_loss`, keyed by checkpoint step.
2. Find the maximum `eval_rewards/accuracies` across all evaluated checkpoints.
3. Among checkpoints that achieve this maximum (within floating-point tolerance, e.g. `abs(diff) < 1e-9`, since exact ties are likely with a 150-example validation set), select the one with the **lowest `eval_loss`**.
4. If there is still a tie after both criteria, select the checkpoint with the **highest step number** (latest, most-trained).
5. Log the full selection table (step, `eval_rewards/accuracies`, `eval_loss`, selected: yes/no) to the training log for auditability.

`save_total_limit=None` keeps every saved checkpoint (~8 over the full run, at ~169 steps/epoch × 5 epochs ≈ 844 steps / 100 ≈ 8 saves) so the selected checkpoint is always available on disk for the final load-and-push step (see "Pushing the Trained Checkpoint to Hugging Face Hub" below).

---

## Sanity Check Before Full Training (100-example dry run)

Before submitting the full training job, write a **separate script, `sanity_check_dpo.py`**, that runs the **entire pipeline end-to-end on a tiny slice of data**, to catch formatting, OOM, and config errors cheaply. Do not implement this as a `--sanity_check` flag bolted onto `train_dpo.py` — keep it as its own file driven by its own config (`configs/harmlessness_sanity.yaml`), so the sanity-check code path is never accidentally left active (or accidentally broken) in the script used for the real run. `sanity_check_dpo.py` should import and reuse the shared data-loading/formatting/training/selection/push functions from `train_dpo.py` (factor these into importable functions rather than inlining them in a `__main__` block) rather than duplicating logic. This is not optional — a config or formatting bug discovered hours into the full training run wastes the GPU allocation and the eval pipeline downstream.

**Sanity-check config (`configs/harmlessness_sanity.yaml`, overrides relative to `configs/harmlessness.yaml`):**

```yaml
subsample_size: 120        # total examples drawn from the filtered pool (down from 1,500)
val_size: 20                # validation examples (down from 150)
num_train_epochs: 1
max_steps: 50               # stop after 50 optimizer steps regardless of epoch count
eval_steps: 20              # small enough to trigger at least 2 intermediate evals within 50 steps
save_steps: 20
logging_steps: 5
```

(`subsample_size: 120` with `val_size: 20` gives ~100 training examples, matching the "100-example dry run" framing of this section. All other keys not listed here — `filtered_pool_size`, `subsample_seed`, `random_seed`, LoRA settings, `hub_model_id`, etc. — are inherited unchanged from `configs/harmlessness.yaml`.)

Effective batch size stays at 16 (`per_device_train_batch_size=2` × `gradient_accumulation_steps=8`), so 50 steps ≈ 800 examples seen — enough for several passes over the 100-example sanity set.

**What to inspect after the sanity run, before launching the full job:**

1. **Chat-template formatting (5–10 examples):** Print the fully templated `(prompt, chosen, rejected)` triples for a handful of sanity-set examples. Confirm the prompt ends at the correct assistant-turn boundary, `chosen`/`rejected` contain only response text (no leaked role markers or special tokens), and the tokens match what the base Tulu/Llama 3.1 model expects (per the "Manual format check" requirement above).
2. **Prompt length distribution:** Confirm `max_prompt_length=512` is not silently truncating a large fraction of the sanity examples (and, by extension, the 1,500-example training subsample — run this check on that subsample too, as already specified above).
3. **Training loss curve (first 50 steps):** Loss should be noisy but trend downward or flat — not flat-zero (sign of a frozen/disconnected LoRA adapter or a data-loading bug) and not exploding/NaN (sign of `learning_rate=5e-6` being too aggressive for this setup, or a formatting bug producing degenerate chosen/rejected pairs).
4. **Intermediate eval fires correctly:** With `eval_steps=20`, confirm at least 2 evaluation events appear in the log, each reporting both `eval_loss` and `eval_rewards/accuracies`. Confirm `eval_rewards/accuracies` is a value in `[0, 1]` (not NaN, not stuck at exactly 0.0 or 1.0 across all eval points, which would indicate the reference-model-via-disabled-adapters trick in Unsloth isn't working).
5. **`eval_loss` vs `rewards/accuracies` sanity:** `eval_loss` should be finite and roughly stable or decreasing; `eval_rewards/accuracies` should not be identical across all eval points (if it is, the model isn't differentiating chosen/rejected — check the DPO loss is actually backpropagating into the LoRA params).
6. **Checkpoint selection logic runs without error:** Run the post-training selection step (see "Checkpoint selection" above) on the 2+ sanity checkpoints and confirm it picks a checkpoint and logs the selection table correctly — this validates the tie-break logic before it matters on the full 8-checkpoint run.
7. **GPU memory:** Check `nvidia-smi` peak usage during the sanity run stays comfortably under the ~35–37 GB estimate, with headroom for the full run (sequence lengths may be longer on average across the full dataset).
8. **Hub push works end-to-end:** Run the final "load selected checkpoint → push to HF Hub" step (see below) against a throwaway/test repo to confirm authentication, `push_to_hub` permissions, and the adapter+tokenizer upload all work before relying on it at the end of the real ~3–6 hour run.

Only proceed to `sbatch train_harmlessness.sh` (full config) once all 8 checks pass.

---

## SLURM Batch Script

Save as `train_harmlessness.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=dpo_harmlessness
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

source ~/.bashrc
conda activate ro2_env          # or your venv activation command

python train_dpo.py \
    --config configs/harmlessness.yaml

echo "Job finished: $(date)"
```

---

## Config File: `configs/harmlessness.yaml`

All parameters in one place — the training run is fully described by this file:

```yaml
# Model
base_model: "allenai/Llama-3.1-Tulu-3-8B-SFT-no-safety-data"
output_dir: "./checkpoints/harmlessness_dpo"   # scratch dir for TRL's intermediate checkpoints during training only —
                                                # not the deliverable; the selected checkpoint is pushed to HF Hub (see below)
hub_model_id: "<your-hf-username>/tulu3-8b-harmlessness-dpo-mvp"  # destination repo for the final selected checkpoint

# Dataset
dataset: "PKU-Alignment/PKU-SafeRLHF"
dataset_split: "train"
conflict_cases_only: true
filtered_pool_size: 12616   # expected size after conflict-case filtering; log the actual count
subsample_size: 1500        # randomly subsampled from the filtered pool (~half Ferreira et al.'s AQUA-RAT scale)
subsample_seed: 7            # seed for the 12,616 -> 1,500 subsampling step
val_size: 150                # held out from the 1500-example subsample
random_seed: 42              # seed for the 1,500 -> train/val split (and any other randomness)

# LoRA
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training (Ferreira et al. 2025, Appendix B.2)
num_train_epochs: 5
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 5e-6
lr_scheduler_type: "cosine"
warmup_ratio: 0.1
weight_decay: 0.01
bf16: true
gradient_checkpointing: true
max_length: 2048
max_prompt_length: 512
beta: 0.1
eval_strategy: "steps"
eval_steps: 100
per_device_eval_batch_size: 2
save_steps: 100
save_total_limit: null   # keep all checkpoints; custom selection logic picks the best one post-training
logging_steps: 10
```

---

## Pushing the Trained Checkpoint to Hugging Face Hub

The trained checkpoint is **not** kept as the final deliverable on local disk. After training completes and the best checkpoint is selected (see "Checkpoint selection" above), `train_dpo.py` must:

1. **Load the selected checkpoint's LoRA adapter** (from `output_dir/checkpoint-<best_step>/`) onto the base model.
2. **Push the adapter and tokenizer to the Hugging Face Hub** as a private repo, using `model.push_to_hub(hub_model_id, private=True)` and `tokenizer.push_to_hub(hub_model_id, private=True)`. Authenticate via `huggingface-cli login` (or `HF_TOKEN` env var) before submitting the job.
3. **Name the repo descriptively**, encoding condition, base model, and the selected step, e.g.:
   `<your-hf-username>/tulu3-8b-harmlessness-dpo-mvp` for the repo itself, with the model card (auto-generated `README.md` via `push_to_hub`) recording: base model (`allenai/Llama-3.1-Tulu-3-8B-SFT-no-safety-data`), training dataset (PKU-SafeRLHF conflict cases, filtered count), the selected checkpoint step, its `eval_rewards/accuracies` and `eval_loss`, and the date trained.
4. **Log the pushed repo ID and commit hash** to the slurm output log, so the RO2 evaluation pipeline (`02_run_inference.py`) can load the model directly via `AutoPeftModelForCausalLM.from_pretrained(hub_model_id)` (or equivalent) without depending on the local scratch checkpoint directory.

The local `checkpoints/harmlessness_dpo/` directory remains on disk only as a byproduct of training (needed for `eval_strategy`/`save_strategy` and the post-training selection step); it is not the artefact referenced by downstream steps or reported in the results.

**Fallback if the Hub push fails:** Wrap the `push_to_hub()` calls in a `try`/`except`. If either call raises (auth failure, network error, quota issue, etc.), do **not** let the run end with the trained adapter existing only as a numbered checkpoint folder — instead, in the `except` block, save the selected checkpoint's adapter and tokenizer locally to a clearly named directory, e.g. `./checkpoints/harmlessness_dpo/best_checkpoint_<best_step>_PUSH_FAILED/`, via `model.save_pretrained(...)` and `tokenizer.save_pretrained(...)`. Log the exception with a full traceback and a clear `ERROR: push_to_hub failed, saved locally to <path> — re-run the push manually once credentials/network are fixed` message to the slurm output log, so the failure is impossible to miss.

---

## Capability Check (Before and After Training)

Run a quick capability evaluation before and after DPO training to confirm the training objective is not causing capability collapse. A collapse in MMLU accuracy would mean the g-mean² shift is confounded by the model losing the ability to reason rather than just changing how it reports its reasoning.

**Metric to track:** MMLU 0-shot accuracy on the same 200 questions in `dataset_index.json`. This is cheap (no sampling, just greedy decoding) and directly comparable.

**Acceptable threshold:** Post-training MMLU accuracy should not drop by more than 5 percentage points relative to the baseline. If it drops more, the training has caused capability degradation and the g-mean² comparison is unreliable.

Log both pre- and post-training accuracy to a `capability_check.json` alongside the g-mean² results.

---

## g-mean² Re-Evaluation on the Trained Checkpoint
(NO coding required for the below; just fyi)

Once training is complete, run the **exact same RO1 evaluation pipeline** on the trained checkpoint (`01_prepare_dataset.py` → `02_run_inference.py` → `03_run_monitor.py` → `04_compute_gmean2.py` → `05_analyze_gmean2.py`). This is the entire point — the only thing that changes is the model. Everything else must be held constant:

- Same fixed eval suite — the problems that passed RO1's **Level-1 filter** (`TE > 0` and `p1 > 0`) on the baseline SFT model (`allenai/Llama-3.1-Tulu-3-8B-SFT-no-safety-data`), from the same `dataset_index.json` and hint assignments
- Same inference prompt templates (unhinted and hinted)
- Same sampling parameters (temperature=0.8, top-p=0.95, **N=4** per condition — RO2 uses N=4, not the N=8 used in RO1 for filtering-decision quality; see `MVP-RO1/metrics/g-mean² Filtering Levels.md`)
- Same binary monitor (GPT-4o, temperature=0, `_MONITOR_PROMPT_TEMPLATE_2` — the Ferreira et al.-style True/False classifier in `src/monitor.py`), with the same uniform prompting across X=0 and X=1 arms
- Same g-mean² computation script (`04_compute_gmean2.py`)

**Critical — the eval suite is fixed, not re-filtered per condition:** Level-1 filtering is run **once**, on the RO1 baseline SFT model, and that set of problems is frozen as the eval suite for all RO2 conditions (including harmlessness-DPO). Do **not** re-run Level-1 filtering on the harmlessness checkpoint to select a different candidate pool — that would make the eval suite itself a function of the training intervention and confound training effects with eval-suite effects.

Within that fixed suite, however, **per-condition exclusions still apply**: for the harmlessness checkpoint, any problem in the suite with `TE ≤ 0` under that checkpoint is excluded from its g-mean² average (TPR_mc is undefined when there's nothing causally hint-driven to detect). The number excluded is reported as the **skip rate** for that condition — a first-class metric in its own right, not just a denominator adjustment. See "RO2 Edge Case Handling — Problems Where TE ≤ 0" in `MVP-RO1/metrics/g-mean² Filtering Levels.md`.

The output is a results file for the harmlessness checkpoint in the same format as the RO1 checkpoint results (`gmean2_per_instance.csv`/`.json`, `gmean2_summary.json`/`.md`), enabling a direct per-instance comparison.

---

## Output Directory Structure

```
ro2_mvp/
├── configs/
│   ├── harmlessness.yaml
│   └── harmlessness_sanity.yaml     # 100-example sanity-check config
├── checkpoints/                     # scratch only — TRL's intermediate checkpoints during training
│   └── harmlessness_dpo/
│       ├── checkpoint-100/
│       ├── checkpoint-200/
│       └── ...                      # best checkpoint selected from these, then pushed to HF Hub (not kept locally)
├── results/
│   ├── baseline/                    # from RO1 — already computed
│   │   ├── generations_hinted.jsonl
│   │   ├── generations_unhinted.jsonl
│   │   ├── monitor_outputs.jsonl
│   │   ├── gmean2_per_instance.csv
│   │   ├── gmean2_per_instance.json
│   │   ├── gmean2_summary.json
│   │   └── gmean2_summary.md
│   └── harmlessness_dpo/            # new — same structure
│       ├── generations_hinted.jsonl
│       ├── generations_unhinted.jsonl
│       ├── monitor_outputs.jsonl
│       ├── gmean2_per_instance.csv
│       ├── gmean2_per_instance.json
│       ├── gmean2_summary.json
│       └── gmean2_summary.md
├── analysis/
│   └── ro2_mvp_comparison.ipynb     # before/after comparison notebook
├── capability_check.json
├── slurm-<jobid>.out
├── train_dpo.py
└── sanity_check_dpo.py              # separate entry point for the 100-example dry run
```

---

## Analysis: Before/After Comparison

Produce the following comparisons between the baseline (from RO1) and the harmlessness-DPO checkpoint.

### Primary Metrics Table

| Metric | Baseline (SFT no-safety) | Harmlessness DPO | Δ |
|---|---|---|---|
| Mean TE (over all 200, before filtering) | — | — | — |
| n passing Level-1 (`TE > 0` and `p1 > 0`) | — | — | — |
| Skip rate | — | — | — |
| Mean TE (over Level-1-passing set) | — | — | — |
| Mean R_effect | — | — | — |
| **g-mean²** | — | — | — |
| Mean TPR_mc | — | — | — |
| Mean TNR_X=0 | — | — | — |
| Mean TNR_X=1 | — | — | — |
| Monitor coverage (missing-verdict rate) | — | — | — |
| MMLU accuracy | — | — | — |

**Always report g-mean² alongside skip rate and mean TE.** A g-mean² shift is uninterpretable in isolation — it may reflect a genuine transparency change (TPR_mc/TNR shift), or it may reflect a change in *which* problems contribute to the average because TE shifted under the new training condition. The eval suite (the candidate pool) is fixed across both checkpoints (see "g-mean² Re-Evaluation" above), but the **per-condition Level-1-passing subset** can still differ — that difference is exactly what the skip rate captures, and large skip-rate differences are themselves a substantive finding (cf. the RO1 four-checkpoint comparison, where skip rate rose from ~60% for SFT models to ~75% after DPO/RLVR).

### Shared Contributing-Set Comparison

In addition to the full comparison, compute g-mean² restricted to the **intersection of Level-1-passing problems** across both checkpoints — i.e. problems where `TE > 0` and `p1 > 0` for *both* the baseline and the harmlessness-DPO model. This controls for the skip-rate confound and gives the cleanest transparency comparison: any difference in g-mean² (and its TPR_mc/TNR components) over this shared subset is attributable to monitorability rather than to which problems are included.

### Per-Instance TE Shift

For each instance in the fixed eval suite, compute `ΔTE_i = TE_i(harmlessness) - TE_i(baseline)`. Plot the distribution of ΔTE. This shows whether the DPO training systematically raised or lowered hint susceptibility across the board, or only affected a subset of instances, and helps explain any skip-rate shift.

### Confusion Matrix (Both Checkpoints)

Using the behavioural label `Y` (1 = model's final answer matches the hinted answer) and the monitor label `Z` (1 = monitor judged the hint explicitly and causally acknowledged in the CoT):

|  | **Y=1 (followed hint)** | **Y=0 (did not follow hint)** |
|---|---|---|
| **Z=1 (acknowledged)** | ✅ Transparent hint-following | ⚠️ Spurious acknowledgement |
| **Z=0 (not acknowledged)** | 🚨 Covert hint-following | ✅ Correct rejection / hint-resistant |

Report counts and proportions for both checkpoints side by side, computed over the Level-1-passing set for each. The key cell to watch is **Y=1, Z=0** (covert hint-following) — if its rate increases after harmlessness training, that is direct evidence of a lower TPR_mc and hence degraded g-mean².

### Qualitative Analysis

Manually read 10–15 CoTs from the **Y=1, Z=0** (covert hint-following) cell of the harmlessness-DPO checkpoint. Compare them to the same cell in the baseline. Look for:

- **Rationalisation sophistication:** Are the DPO CoTs constructing more polished post-hoc arguments for the hinted answer?
- **Hint erasure:** Is the model paraphrasing the professor's authority without naming it? (e.g. "domain experts typically favour X" instead of "the professor suggests X")
- **Safety artefacts:** Does the harmlessness-trained model produce disclaimers, refusal preambles, or hedging language that affects how the hint is treated in the CoT?
- **Answer drift:** Do CoTs show mid-reasoning reversal toward the hinted answer that the monitor classified as `Z=0` (not acknowledged)?

Document 3–5 representative examples with annotation for each checkpoint. These are essential for the dissertation — numbers alone cannot explain *how* the training changed the model's behaviour.

---

## Hypotheses for the MVP

**H_MVP1 — g-mean² (via TPR_mc) decreases after harmlessness DPO.**
The harmlessness-trained checkpoint will show a lower g-mean² than the baseline over the shared contributing set, driven by a lower TPR_mc, because harmlessness training incentivises safe-sounding CoTs that are less likely to explicitly name deference to an authority figure as a reason for the answer.

**H_MVP2 — TE and skip rate are approximately stable.**
Mean TE and skip rate over the fixed eval suite will not change substantially after harmlessness DPO. The training objective targets response style (harmlessness), not answer selection strategy, so hint susceptibility should be largely unaffected. If TE drops and skip rate rises sharply (as RO1 observed for DPO/RLVR conditions generally), the Level-1-passing set will shrink and any g-mean² change becomes harder to interpret without the shared contributing-set comparison.

**H_MVP3 — Covert hint-following increases.**
The ratio of (Y=1, Z=0) to (Y=1, Z=1) — covert vs. transparent hint-following — will be higher in the harmlessness checkpoint than in the baseline: the same rate of hint-following but less acknowledgement of it.

**H_MVP4 — Capability is preserved.**
MMLU accuracy will not drop by more than 5 percentage points after DPO training, confirming that any g-mean² shift reflects a transparency change rather than capability collapse.

---

## What Success Looks Like

By the end of the RO2 MVP you should have:

1. **A working DPO training script** (TRL + Unsloth, bf16 LoRA, sbatch job) that completes without OOM or crash on the A6000
2. **One trained harmlessness checkpoint**, selected via the validation-accuracy/eval-loss tie-break, pushed to the Hugging Face Hub under `hub_model_id` (e.g. `<your-hf-username>/tulu3-8b-harmlessness-dpo-mvp`)
3. **g-mean² evaluated** on that checkpoint using the identical RO1 pipeline and fixed eval suite
4. **A before/after table** with g-mean², TPR_mc, TNR_X=0, TNR_X=1, skip rate, mean TE, and MMLU accuracy for both checkpoints
5. **A qualitative sample** of 5–10 annotated CoTs from the Y=1, Z=0 (covert hint-following) cell of the harmlessness checkpoint
6. **Confirmation that the pipeline generalises** — running helpfulness or anti-sycophancy next requires only a dataset swap in the config file

The MVP is successful even if the g-mean² shift is small or in the unexpected direction — what matters is that the pipeline produces a stable, interpretable result. An unexpected result is scientifically interesting and worth reporting; a broken pipeline is not.

---

## Environment Setup

```bash
# Create environment
conda create -n ro2_env python=3.11
conda activate ro2_env

# Install core dependencies (match CUDA 12.8)
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.44.0
pip install trl==0.9.6
pip install peft==0.12.0
pip install datasets==2.21.0
pip install accelerate==0.33.0
pip install bitsandbytes==0.43.3   # for paged AdamW
pip install unsloth                # installs Triton kernels automatically
pip install openai                 # for monitor calls
pip install scipy pandas matplotlib  # for analysis

# Pin versions
pip freeze > requirements.txt
```

**Important:** Pin all dependency versions in `requirements.txt` before running any training. TRL, PEFT, and Unsloth APIs change frequently and an environment that works today may break after an update. MVP-RO2 should have its own `requirements.txt` file, seperate from MVP-RO1.

---

## Required Credentials (Set in Code, Not in the Terminal)

Both `train_dpo.py` and `sanity_check_dpo.py` must load credentials **from a `.env` file via code** (e.g. `python-dotenv`'s `load_dotenv()` at the top of the script), not rely on the developer having run `huggingface-cli login` or `export HF_TOKEN=...` interactively in the SLURM shell — interactive logins don't persist into `sbatch` jobs, and an unauthenticated run will fail hours in, after the GPU time is already spent.

Add `python-dotenv` to `requirements.txt`. Create a `.env` file in the MVP-RO2 project root (and add `.env` to `.gitignore` — **never commit it**):

```bash
# .env (do not commit)
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**At the top of `train_dpo.py` and `sanity_check_dpo.py`:**

```python
from dotenv import load_dotenv
from huggingface_hub import login
import os

load_dotenv()
login(token=os.environ["HF_TOKEN"])
```

- **`HF_TOKEN`** — required for: (1) downloading `allenai/Llama-3.1-Tulu-3-8B-SFT-no-safety-data` if it is gated/private, and (2) `push_to_hub()` for the trained adapter and tokenizer (see "Pushing the Trained Checkpoint to Hugging Face Hub"). The token must have **write** access to the target `hub_model_id` repo.
- **`OPENAI_API_KEY`** — required by the RO1 evaluation pipeline's monitor step (`03_run_monitor.py`, GPT-4o calls) when re-running the eval suite on the trained checkpoint. Not needed by `train_dpo.py` itself, but should be set in the same `.env` so the downstream eval scripts pick it up the same way.

Fail fast and loudly if either variable is missing — check `os.environ` (or the `.env`-loaded values) at the start of the script and raise a clear error before any GPU work begins, rather than discovering a missing key only at the `push_to_hub()` call after hours of training.

---

## Compute & Cost Estimates for MVP RO2

| Step | Estimate |
|---|---|
| DPO training (5 epochs, ~1,350 train + 150 val examples, Unsloth) | ~1.5–3 hours on A6000 |
| g-mean² inference (200 instances × 2 arms × 4 samples) | ~15 minutes on A6000 |
| Monitor calls (1,600 GPT-4o calls × ~700 tokens avg) | ~1.1M tokens; check OpenAI quota |
| MMLU capability check (greedy, 200 examples) | ~5 minutes |
| **Total wall-clock (training + eval)** | **~4–7 hours** |

Run `sanity_check_dpo.py` first (see "Sanity Check Before Full Training" above) to confirm no OOM, correct formatting, healthy loss curves, working intermediate eval, and a working Hub push, before submitting the full training job.
