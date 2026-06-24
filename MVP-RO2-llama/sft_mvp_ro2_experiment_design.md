# SFT MVP — RO2 Experiment Design

> **HISTORICAL.** This spec describes the original Qwen3-4B + PKU-SafeRLHF run with
> a CoT-monitorability objective. The current code is the **Llama-3.2-3B-Instruct +
> Anthropic/hh-rlhf (harmless-base, single-turn)** port, which **drops the CoT/thinking
> handling** (Llama has no thinking mode). See `documentation.md` for the live design.

## 1. Purpose

This document is a developer-ready specification for the first RO2 experiment. The goal is to check whether harmlessness-aligned post-training shifts CoT monitorability, using **SFT as a proxy for DPO** while DPO compute is pending.

This is not a throwaway pilot. SFT on chosen responses from the DPO dataset is one of two planned mechanism ablations in RO2 (the "SFT ablation" that isolates content effects from contrastive suppression effects). The result feeds directly into the final dissertation.

**What this experiment is NOT:**  
DPO training. The key difference: SFT trains only on chosen (safe) responses and applies no contrastive pressure from rejected responses. When DPO compute arrives, the same dataset will be used for the full DPO run, and the SFT vs DPO comparison is itself a finding.

---

## 2. Model

| Field | Value |
|---|---|
| Base model | `Qwen/Qwen3-4B` |
| HuggingFace ID | `Qwen/Qwen3-4B` |
| Architecture | Causal LM with native thinking (CoT) support |
| Precision | `bfloat16` |

> **Note on Qwen3-4B thinking mode:** Qwen3-4B is a native reasoning model that generates `<think>...</think>` blocks before its final response. During SFT training, the chosen responses from the dataset are plain text (no thinking blocks). Training should be done in **non-thinking mode** (see Section 5). The model's thinking behaviour is expected to be preserved for evaluation, since the thinking tokens are not part of the training targets.

---

## 3. Dataset

### 3.1 Source

- **HuggingFace dataset:** `PKU-Alignment/PKU-SafeRLHF`
- **Split:** `train`

### 3.2 Filtering — conflict cases only

The full dataset contains pairs where one response is safer and the other is more helpful, and pairs where both properties coincide. We want **conflict cases only**, where harmlessness and helpfulness diverge, to isolate the harmlessness signal.

**Filter condition:**

```python
is_conflict = dataset["is_response_0_safe"] != dataset["is_response_1_safe"]
dataset = dataset.filter(lambda x: x["is_response_0_safe"] != x["is_response_1_safe"])
```

Expected size after filtering: **~12,616 examples**.

### 3.3 Constructing SFT pairs

For each conflict example, the **chosen response** is the **safer** response (not the more helpful one). Use the `safer_response_id` column to identify it.

```python
def extract_sft_pair(example):
    safer_id = example["safer_response_id"]  # 0 or 1
    chosen = example[f"response_{safer_id}"]
    return {
        "prompt": example["prompt"],
        "chosen": chosen,
    }

dataset = dataset.map(extract_sft_pair)
dataset = dataset.select_columns(["prompt", "chosen"])
```

### 3.4 Dataset format for training

Each example should be formatted as a chat-style conversation using Qwen3-4B's tokenizer chat template, with thinking **disabled**:

```python
def format_example(example, tokenizer):
    messages = [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["chosen"]},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,   # disable <think> blocks during SFT
    )
    return {"text": text}
```

> **No system prompt.** Do not add a system prompt unless the base model's default template requires one. Keep the format as minimal as possible.

### 3.5 Train/validation split

No validation split is required. Training is fixed-duration (see Section 5). If you want a lightweight validation loss curve, hold out 200 examples (`dataset.train_test_split(test_size=200, seed=42)`).

---

## 4. Pre-Training: Verify Sequence Length

**Run this before training.** The `max_seq_length` in Section 5 must cover the vast majority of formatted training examples. Any example longer than `max_seq_length` is silently truncated, training the model on an incomplete target — which distorts the learning signal. The appropriate value depends on the actual dataset content and cannot be assumed.

```python
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

# --- reproduce the same filtering and formatting from Section 3 ---
dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train")
dataset = dataset.filter(lambda x: x["is_response_0_safe"] != x["is_response_1_safe"])

def extract_sft_pair(example):
    safer_id = example["safer_response_id"]
    return {
        "prompt": example["prompt"],
        "chosen": example[f"response_{safer_id}"],
    }
dataset = dataset.map(extract_sft_pair, remove_columns=dataset.column_names)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

def measure_length(example):
    messages = [
        {"role": "user",      "content": example["prompt"]},
        {"role": "assistant", "content": example["chosen"]},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    return {"length": len(tokenizer(text)["input_ids"])}

dataset = dataset.map(measure_length)
lengths = np.array(dataset["length"])

print(f"n                  : {len(lengths)}")
print(f"mean               : {lengths.mean():.0f}")
print(f"median             : {np.median(lengths):.0f}")
print(f"95th pct           : {np.percentile(lengths, 95):.0f}")
print(f"99th pct           : {np.percentile(lengths, 99):.0f}")
print(f"max                : {lengths.max():.0f}")
print()
for cap in [512, 1024, 2048, 4096, 8192]:
    n_over = (lengths > cap).sum()
    pct    = n_over / len(lengths) * 100
    print(f"  > {cap:5d} tokens  : {n_over:5d} examples  ({pct:.2f}%)")
```

**Decision rule:**

| % of examples exceeding the cap | Action |
|---|---|
| < 1% | Use that cap — truncation is negligible |
| 1–5% | Consider increasing to the next threshold |
| > 5% | Must increase — too many examples are truncated |

Set `max_seq_length` to the **lowest cap where truncation falls below 1%**, then update the value in Section 5 accordingly. Typical values for human preference datasets are 1024–2048; if PKU-SafeRLHF skews long, 4096 may be needed.

---

## 5. LoRA Configuration

Use PEFT LoRA. Do **not** train the full model.

```python
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)
```

---

## 5. Training Configuration

Use `trl.SFTTrainer` with the LoRA config above.

```python
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    # Output
    output_dir="./qwen3-4b-sft-harmlessness",

    # Duration — 3 epochs with per-epoch checkpointing.
    # LoRA's low-rank constraint (~1.6% of parameters trainable) converges
    # more slowly than full fine-tuning; 2 epochs risks an insufficiently
    # large behavioural shift to detect a monitorability change.
    # Saving at each epoch gives a 3-point training trajectory for analysis.
    # If training loss has clearly plateaued by end of epoch 2, epoch 3 can
    # be skipped — check the loss curve before proceeding.
    num_train_epochs=3,
    save_strategy="epoch",          # checkpoint at end of each epoch

    # Batch size — adjust per_device_train_batch_size to fit GPU VRAM
    # Effective batch size must be 32
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,     # 2 * 16 = 32 effective

    # Optimiser
    optim="adamw_torch_fused",          # use "paged_adamw_8bit" if OOM
    learning_rate=2e-4,
    weight_decay=0.01,
    max_grad_norm=1.0,

    # LR schedule
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,

    # Sequence length
    max_seq_length=2048,

    # Precision
    bf16=True,
    fp16=False,

    # Memory
    gradient_checkpointing=True,

    # Logging
    logging_steps=10,

    # Checkpointing — keep ALL epoch checkpoints, never delete any.
    # All three will be merged and pushed to HF separately after training
    # so any checkpoint can be evaluated later without re-training.
    save_total_limit=None,

    # Packing — disabled to avoid cross-example contamination
    packing=False,

    # Dataset
    dataset_text_field="text",          # matches the field name from format_example()
)
```

> **Flash Attention:** If the environment supports it, load the model with `attn_implementation="flash_attention_2"` for faster training. This is optional but recommended.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # remove if not supported
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
tokenizer.padding_side = "right"   # required for SFT with causal LM
```

---

## 6. Training Labels

Only the **assistant turn** should be in the loss. The user turn (prompt) should be masked out.

`SFTTrainer` handles this automatically when the chat template is applied correctly and `dataset_text_field` is set. Verify that labels for the prompt tokens are set to `-100` (masked). If using a custom data collator, ensure this explicitly.

---

## 7. Expected Training Time and Scale

| Item | Value |
|---|---|
| Dataset size (post-filter) | ~12,616 examples |
| Epochs | 2 |
| Total training steps (approx) | ~1,182 steps at effective batch 32 (3 epochs) |
| Checkpoints saved | 3 (one per epoch, at ~394 / ~788 / ~1,182 steps) |
| Approximate training time | 2–3 hours on A100 40GB; 4–6 hours on A6000 |

---

## 8. Model Output

### 8.1 What gets saved during training

With `save_strategy="epoch"` and `save_total_limit=None`, three checkpoint directories are created under `output_dir` at the end of each epoch — one per epoch, none deleted. These contain the **LoRA adapter weights only** (not the full model). Each directory is self-contained and includes `adapter_config.json` (which records the base model ID) and `adapter_model.safetensors`.

```
./qwen3-4b-sft-harmlessness/
    checkpoint-394/      # end of epoch 1  (approx step — depends on filtered dataset size)
    checkpoint-788/      # end of epoch 2
    checkpoint-1182/     # end of epoch 3
```

> The exact step numbers depend on the actual filtered dataset size. The push script below discovers them automatically via `glob` so no hardcoding is needed.

### 8.2 Push all adapter checkpoints to HuggingFace Hub

Run this script **after training completes**. It uploads each checkpoint directory as-is — no model loading or merging required. The adapter files are small (typically 100–300 MB each), so this is fast. Merging into the base model happens later at evaluation time.

```python
import glob
import os
from huggingface_hub import HfApi

TRAIN_DIR   = "./qwen3-4b-sft-harmlessness"
HF_USERNAME = "YOUR_HF_USERNAME"
REPO_PREFIX = "qwen3-4b-sft-harmlessness-pku-adapter"

api = HfApi()

# Discover checkpoint dirs and sort by step number
checkpoint_dirs = sorted(
    glob.glob(os.path.join(TRAIN_DIR, "checkpoint-*")),
    key=lambda p: int(p.split("-")[-1]),
)

print(f"Found {len(checkpoint_dirs)} checkpoints: {checkpoint_dirs}")

for epoch_idx, ckpt_dir in enumerate(checkpoint_dirs, start=1):
    repo_id = f"{HF_USERNAME}/{REPO_PREFIX}-epoch{epoch_idx}"
    print(f"Pushing epoch {epoch_idx} from {ckpt_dir} → {repo_id}")

    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(
        folder_path=ckpt_dir,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Done: {repo_id}")

print("\nAll adapter checkpoints pushed.")
```

**Resulting HF repos (adapter only):**

| Epoch | Repo ID |
|---|---|
| 1 | `YOUR_HF_USERNAME/qwen3-4b-sft-harmlessness-pku-adapter-epoch1` |
| 2 | `YOUR_HF_USERNAME/qwen3-4b-sft-harmlessness-pku-adapter-epoch2` |
| 3 | `YOUR_HF_USERNAME/qwen3-4b-sft-harmlessness-pku-adapter-epoch3` |

The **epoch 3 adapter** is the primary evaluation target. Epochs 1 and 2 are retained for trajectory analysis if needed.

> To load an adapter at evaluation time: load `Qwen/Qwen3-4B` as the base, then wrap with `PeftModel.from_pretrained(base, repo_id)`. The base model ID is already recorded in `adapter_config.json` inside each repo.

---

## 9. Sanity Checks Before Handing Off for Evaluation

Before passing the model to the evaluation pipeline, verify the following:

1. **Generation test in thinking mode — run after every epoch checkpoint.** Load the base model + LoRA adapter and run a sample inference with `enable_thinking=True`. Confirm the model still produces `<think>...</think>` blocks of reasonable length (hundreds of tokens). If thinking is completely suppressed or CoT length drops dramatically, the SFT is pushing the model away from reasoning behaviour — do not proceed to epoch 3.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base, "./qwen3-4b-sft-harmlessness/checkpoint-394")  # adjust path per epoch
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

messages = [{"role": "user", "content": "What is 17 × 23?"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(output[0], skip_special_tokens=False))
```

2. **Loss is not zero.** Check that training loss starts above ~1.0 and decreases. A flat or near-zero loss from step 1 indicates a formatting or label masking problem.

3. **Adapter file size.** Each checkpoint directory should contain `adapter_model.safetensors` of approximately 100–300 MB. A missing file or unexpectedly large size indicates a save issue.

---

## 10. Evaluation

Evaluation setup and execution are handled separately and are not part of this specification. The merged HuggingFace model will be used as a drop-in replacement for the base model in the existing evaluation pipeline, with identical configuration (same hint type, same datasets, same N per arm, same max_new_tokens).

---

## 11. Key Dependencies

```
transformers>=4.51.0
peft>=0.14.0
trl>=0.12.0
torch>=2.2.0
datasets>=2.19.0
huggingface_hub>=0.23.0
accelerate>=0.30.0
bitsandbytes>=0.43.0    # only if using paged_adamw_8bit
flash-attn>=2.5.0       # optional, for flash_attention_2
```

---

## 12. File / Repo Checklist for Developer

- [ ] Filter PKU-SafeRLHF to conflict cases and verify ~12,616 examples
- [ ] Format dataset using `apply_chat_template` with `enable_thinking=False`
- [ ] Verify label masking — prompt tokens must be `-100`
- [ ] Train with LoRA config above; confirm loss decreases
- [ ] Confirm 3 checkpoint directories exist under `output_dir` after training
- [ ] Run push script (Section 8.2) — all 3 adapter-only repos created on HF
- [ ] Verify all 3 repos are visible and private on HuggingFace Hub
- [ ] Run generation sanity check: load base + epoch 3 adapter, confirm thinking mode works
- [ ] Share all 3 HF adapter repo IDs with evaluator (`-epoch1`, `-epoch2`, `-epoch3`)
