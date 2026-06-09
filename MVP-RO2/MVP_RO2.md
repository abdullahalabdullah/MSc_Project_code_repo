# MVP: RO2 — Effect of Post-Training on CoT Monitorability

## Overview

This document describes the MVP for Research Objective 2 (RO2). RO1 is assumed complete: the evaluation pipeline (inference, TE computation, monitor, HAR computation) is working and has been validated on four Tulu checkpoints using MMLU with the Stanford professor hint. The `dataset_index.json` (200 questions with fixed hint assignments) and the RO1 results are available.

RO2 asks: **does applying DPO fine-tuning under a specific post-training objective change the model's CoT monitorability, as measured by HAR?**

The MVP tests this with a single training condition — **harmlessness** — to validate the full RO2 pipeline (training + evaluation) before committing to all three objectives.

---

## Goal

- Train one DPO model on the harmlessness objective (PKU-SafeRLHF dataset) starting from the baseline checkpoint (`allenai/Llama-3.1-Tulu-3-8B-SFT-no-safety-data`)
- Re-run the RO1 evaluation pipeline on the trained checkpoint using the **exact same** `dataset_index.json`, hint assignments, and prompt templates
- Produce a before/after comparison table: baseline HAR vs. harmlessness-DPO HAR, alongside |S| and mean TE
- Confirm the full RO2 code path works end-to-end and that the HAR metric is sensitive enough to detect a training-induced shift

---

## Why Harmlessness First

The harmlessness condition is the MVP choice for three reasons:

1. **It is the hypothesis-critical condition.** The primary hypothesis is that harmlessness training specifically degrades HAR on authority-cued instances, by incentivising safe-sounding CoTs that conceal deference to an external authority figure. This is the condition that matters most for the dissertation argument.
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

**No quantisation.** The A6000's 48 GB VRAM is sufficient to run full bf16 LoRA without NF4 or 8-bit quantisation. Quantisation would introduce noise as a methodological confound — any HAR shift could be attributed to quantisation artefacts rather than the training objective. Use:

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

**Format:** Each example must be converted to the DPO `(prompt, chosen, rejected)` format:
- `prompt`: the user turn (the question or instruction)
- `chosen`: the safer response (the one labelled as safer in the conflict case)
- `rejected`: the less safe response

**Chat template:** Apply the Tulu/Llama 3.1 instruct chat template to all three fields via `tokenizer.apply_chat_template()` before passing to the trainer. Do not manually format strings.

**Decontamination check:** Verify no PKU-SafeRLHF examples overlap with the 200 MMLU questions in `dataset_index.json` before training. A simple string match on question text is sufficient for the MVP.

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

This gives ~13.6M trainable parameters (LoRA adapters only), with the base model weights frozen. Since the base model is identical across all conditions, any HAR shift is attributable to the training objective rather than changes in the underlying weight distribution.

---

## DPO Training Configuration

```python
dpo_config = DPOConfig(
    output_dir="./checkpoints/harmlessness_dpo",
    num_train_epochs=1,              # one epoch for MVP; scale up if loss hasn't converged
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,   # effective batch size = 16
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    bf16=True,
    gradient_checkpointing=True,
    max_length=2048,                 # cap to prevent quadratic memory scaling on backward pass
    max_prompt_length=512,
    beta=0.1,                        # KL penalty coefficient — standard for offline DPO
    ref_model=None,                  # Unsloth handles reference model internally
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,                  # save intermediate checkpoints in case of crash
    save_total_limit=3,
    remove_unused_columns=False,
    report_to="none",                # no wandb for MVP; use slurm log file
)
```

**Why `max_seq_length=2048`:** The Tulu base model supports 32k context, but DPO backward pass memory scales quadratically with sequence length. Cap at 2048 for the MVP; increase to 4096 only if the training data has many examples that are truncated at 2048 (check truncation rate in the training log).

**Why `beta=0.1`:** Standard offline DPO KL penalty. Lower values allow larger policy deviations from the reference; higher values constrain updates more. 0.1 is the Tulu 3 paper's value and a safe default.

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
output_dir: "./checkpoints/harmlessness_dpo"

# Dataset
dataset: "PKU-Alignment/PKU-SafeRLHF"
dataset_split: "train"
conflict_cases_only: true
max_examples: 12616
random_seed: 42

# LoRA
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training
num_train_epochs: 1
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 5e-5
lr_scheduler_type: "cosine"
warmup_ratio: 0.1
bf16: true
gradient_checkpointing: true
max_length: 2048
max_prompt_length: 512
beta: 0.1
save_steps: 500
save_total_limit: 3
logging_steps: 10
```

---

## Capability Check (Before and After Training)

Run a quick capability evaluation before and after DPO training to confirm the training objective is not causing capability collapse. A collapse in MMLU accuracy would mean the HAR shift is confounded by the model losing the ability to reason rather than just changing how it reports its reasoning.

**Metric to track:** MMLU 0-shot accuracy on the same 200 questions in `dataset_index.json`. This is cheap (no sampling, just greedy decoding) and directly comparable.

**Acceptable threshold:** Post-training MMLU accuracy should not drop by more than 5 percentage points relative to the baseline. If it drops more, the training has caused capability degradation and the HAR comparison is unreliable.

Log both pre- and post-training accuracy to a `capability_check.json` alongside the HAR results.

---

## HAR Re-Evaluation on the Trained Checkpoint

Once training is complete, run the **exact same RO1 evaluation pipeline** on the trained checkpoint. This is the entire point — the only thing that changes is the model. Everything else must be held constant:

- Same `dataset_index.json` (200 questions, same hint assignments)
- Same inference prompt templates (unhinted and hinted)
- Same sampling parameters (temperature=0.8, top-p=0.95, N=8)
- Same δ threshold (fixed from RO1 baseline run — do NOT recompute δ from the new model's TE values)
- Same monitor prompt (GPT-4o, temperature=0, `MONITOR_PROMPT_V1`)
- Same HAR computation script

**Critical:** δ must be fixed from the RO1 baseline. If you recompute δ from the harmlessness model's TE values, S will shift and the HAR comparison will be uninterpretable — you won't know if a HAR change reflects a transparency change or just a different set of instances being included in the calculation.

The output is a results file for the harmlessness checkpoint in the same format as the RO1 checkpoint results, enabling a direct per-instance comparison.

---

## Output Directory Structure

```
ro2_mvp/
├── configs/
│   └── harmlessness.yaml
├── checkpoints/
│   └── harmlessness_dpo/
│       ├── checkpoint-500/
│       ├── checkpoint-1000/
│       └── final/                   # final trained adapter weights
├── results/
│   ├── baseline/                    # from RO1 — already computed
│   │   ├── generations_hinted.jsonl
│   │   ├── generations_unhinted.jsonl
│   │   ├── te_per_instance.csv
│   │   └── monitor_outputs.jsonl
│   └── harmlessness_dpo/            # new — same structure
│       ├── generations_hinted.jsonl
│       ├── generations_unhinted.jsonl
│       ├── te_per_instance.csv
│       └── monitor_outputs.jsonl
├── analysis/
│   └── ro2_mvp_comparison.ipynb     # before/after comparison notebook
├── capability_check.json
├── slurm-<jobid>.out
└── train_dpo.py
```

---

## Analysis: Before/After Comparison

Produce the following comparisons between the baseline (from RO1) and the harmlessness-DPO checkpoint.

### Primary Metrics Table

| Metric | Baseline (SFT no-safety) | Harmlessness DPO | Δ |
|---|---|---|---|
| Mean TE (over all 200) | — | — | — |
| |S| (instances retained) | — | — | — |
| Mean TE (over S) | — | — | — |
| HAR | — | — | — |
| Opacity rate | — | — | — |
| Monitor failure rate | — | — | — |
| MMLU accuracy | — | — | — |

**Always report HAR alongside |S| and mean TE.** A HAR shift is uninterpretable in isolation — it may reflect a transparency change, or it may reflect a change in which instances are included in S due to TE shifting. A true monitorability improvement leaves |S| approximately stable while HAR rises across the same retained instances.

### Shared-S Comparison

In addition to the full comparison, compute HAR restricted to the **intersection of S** across both models — the instances retained by both the baseline δ threshold and where both models followed the hint at least once. This controls for the |S| confound and gives the cleanest transparency comparison.

### Per-Instance TE Shift

For each instance, compute `ΔTE_i = TE_i(harmlessness) - TE_i(baseline)`. Plot the distribution of ΔTE. This shows whether the DPO training systematically raised or lowered hint susceptibility across the board, or only affected a subset of instances.

### Confusion Matrix (Both Checkpoints)

|  | **Followed hint** | **Did not follow hint** |
|---|---|---|
| **Acknowledged** | ✅ Transparent hint-following | ⚠️ Spurious acknowledgement |
| **Not acknowledged** | 🚨 Covert hint-following | ✅ Correct rejection |

Report counts and proportions for both checkpoints side by side. The key cell to watch is **Followed + Not Acknowledged** (covert hint-following) — if this increases after harmlessness training, that is direct evidence of degraded monitorability.

### Q-Level Breakdown

For hint-following runs in S, report the proportion of Yes answers for each monitor question (Q1–Q7) for both checkpoints. The most informative comparisons:

- **Q1 rate:** Did the base rate of hint mention change?
- **Q3 rate:** Did post-hoc-only mentions increase (model mentions hint only after independently reaching the same answer)?
- **Q4 rate:** Did causal acknowledgements decrease?
- **Q7 rate:** Did override cases (model's reasoning pointed elsewhere but hint won) change?

### Qualitative Analysis

Manually read 10–15 CoTs from the **Followed + Not Acknowledged** cell of the harmlessness-DPO checkpoint. Compare them to the same cell in the baseline. Look for:

- **Rationalisation sophistication:** Are the DPO CoTs constructing more polished post-hoc arguments for the hinted answer?
- **Hint erasure:** Is the model paraphrasing the professor's authority without naming it? (e.g. "domain experts typically favour X" instead of "the professor suggests X")
- **Safety artefacts:** Does the harmlessness-trained model produce disclaimers, refusal preambles, or hedging language that affects how the hint is treated in the CoT?
- **Answer drift:** Do CoTs show mid-reasoning reversal toward the hinted answer that the monitor classified as Not Acknowledged?

Document 3–5 representative examples with annotation for each checkpoint. These are essential for the dissertation — numbers alone cannot explain *how* the training changed the model's behaviour.

---

## Hypotheses for the MVP

**H_MVP1 — HAR decreases after harmlessness DPO.**
The harmlessness-trained checkpoint will show lower HAR than the baseline on the retained instances S, because harmlessness training incentivises safe-sounding CoTs that are less likely to explicitly name deference to an authority figure as a reason for the answer.

**H_MVP2 — TE is approximately stable.**
Mean TE over S will not change substantially after harmlessness DPO. The training objective targets response style (harmlessness), not answer selection strategy, so hint susceptibility should be largely unaffected. If TE does drop significantly, |S| will shrink and any HAR change becomes harder to interpret.

**H_MVP3 — Covert hint-following increases.**
The ratio of (Followed + Not Acknowledged) to (Followed + Acknowledged) will be higher in the harmlessness checkpoint than in the baseline — the same rate of hint-following but less acknowledgement of it.

**H_MVP4 — Capability is preserved.**
MMLU accuracy will not drop by more than 5 percentage points after DPO training, confirming that any HAR shift reflects a transparency change rather than capability collapse.

---

## What Success Looks Like

By the end of the RO2 MVP you should have:

1. **A working DPO training script** (TRL + Unsloth, bf16 LoRA, sbatch job) that completes without OOM or crash on the A6000
2. **One trained harmlessness checkpoint** saved to `checkpoints/harmlessness_dpo/final/`
3. **HAR evaluated** on that checkpoint using the identical RO1 pipeline
4. **A before/after table** with HAR, |S|, mean TE, opacity rate, and MMLU accuracy for both checkpoints
5. **A qualitative sample** of 5–10 annotated CoTs from the Followed + Not Acknowledged cell of the harmlessness checkpoint
6. **Confirmation that the pipeline generalises** — running helpfulness or anti-sycophancy next requires only a dataset swap in the config file

The MVP is successful even if the HAR shift is small or in the unexpected direction — what matters is that the pipeline produces a stable, interpretable result. An unexpected result is scientifically interesting and worth reporting; a broken pipeline is not.

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

**Important:** Pin all dependency versions in `requirements.txt` before running any training. TRL, PEFT, and Unsloth APIs change frequently and an environment that works today may break after an update.

---

## Compute & Cost Estimates for MVP RO2

| Step | Estimate |
|---|---|
| DPO training (1 epoch, ~12,616 examples, Unsloth) | ~3–5 hours on A6000 |
| HAR inference (200 instances × 2 arms × 8 samples) | ~30 minutes on A6000 |
| Monitor calls (1,600 GPT-4o calls × ~700 tokens avg) | ~1.1M tokens; check OpenAI quota |
| MMLU capability check (greedy, 200 examples) | ~5 minutes |
| **Total wall-clock (training + eval)** | **~4–6 hours** |

Run a small training sanity check first (100 examples, 50 steps) to confirm no OOM before submitting the full training job.
