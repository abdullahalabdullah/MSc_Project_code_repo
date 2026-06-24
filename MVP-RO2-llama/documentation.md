# MVP-RO2 — SFT (Harmlessness) Training: Code Documentation

This documents the code that implements the **RO2 SFT MVP**: LoRA supervised
fine-tuning of `meta-llama/Llama-3.2-3B-Instruct` on the **chosen (more-harmless)
responses** of Anthropic/hh-rlhf `harmless-base` *single-turn* examples, saving
one adapter checkpoint per epoch (4 epochs → 4 checkpoints) and pushing all four
(private) to the Hugging Face Hub for the existing RO1 eval pipeline.

`sft_mvp_ro2_experiment_design.md` in this folder is the original spec from the
Qwen3 + PKU-SafeRLHF version of this experiment; it is kept for history. This
Llama port targets a different model and dataset and **drops the chain-of-thought
(CoT) monitorability objective** — Llama-3.2-3B-Instruct has no thinking mode, so
all `<think>`-block handling, the thinking-mode generation checks, and the
think-token statistics have been removed. Evaluation is out of scope — the
adapters are the hand-off.

---

## 1. Directory structure

```
MVP-RO2-llama/
├── documentation.md                   # this file
├── sft_mvp_ro2_experiment_design.md   # original (Qwen/PKU) spec — historical
├── requirements.txt                   # pinned modern stack (no Unsloth)
├── .env.example                       # credential template -> copy to .env
├── .gitignore                         # ignores .env, outputs/, caches, weights
├── run_sft.sh                         # one-shot launcher: preflight -> gate -> train -> push
├── run_all.sh                         # same, plus an explicit token preflight
│
├── configs/
│   ├── sft_harmlessness.yaml          # FULL run config — the whole run is described here
│   └── sft_harmlessness_sanity.yaml   # tiny-slice overrides for the sanity gate
│
├── sft/                               # importable package — all logic lives here
│   ├── __init__.py
│   ├── config.py                      # YAML load + deep-merge; Hub repo-id helpers
│   ├── logging_utils.py               # stdout logger
│   ├── credentials.py                 # load .env, fail fast on missing HF_TOKEN, hf login
│   ├── data.py                        # hh-rlhf load/filter, completion-only masking, verify, seq-length report
│   ├── model.py                       # Llama-3.2-3B + tokenizer load, LoRA config
│   ├── training.py                    # SFTConfig + SFTTrainer assembly; run_training()
│   ├── push.py                        # discover epoch checkpoints -> push adapters + model cards
│   └── merge.py                       # merge an adapter into the base -> standalone model for RO1
│
└── scripts/                           # thin entry points (argparse); import the sft package
    ├── _bootstrap.py                  # puts repo root on sys.path
    ├── 00_check_seq_length.py         # preflight -> recommends max_length
    ├── 01_train_sft.py                # main training run
    ├── 02_push_adapters.py            # push all 4 epoch adapters
    ├── 04_merge_adapter.py            # merge one adapter -> standalone model (generic CLI)
    ├── 04_merge_and_push.py           # merge an epoch's adapter + push merged model for RO1
    └── sanity_check_sft.py            # ENVIRONMENT GATE — tiny end-to-end run + assertions
```

> Plain PEFT + TRL `SFTTrainer`, **no Unsloth**.

---

## 2. Code structure and data flow

All logic is in the `sft/` package; `scripts/` are thin wrappers. The sanity gate
reuses the exact same `run_training()` path as the real run, just on a tiny slice.

```
run_training(cfg, logger)                         # sft/training.py
  ├─ check_credentials(logger)                    # .env -> HF_TOKEN (fail fast) + hf login
  ├─ load_tokenizer(cfg) / load_model(cfg)        # Llama-3.2-3B bf16, padding_side=right, use_cache=False
  ├─ get_peft_model(model, LoraConfig)            # LoRA r16/α32; enable_input_require_grads()
  ├─ _prepare_data(cfg, tokenizer)                # sft/data.py:
  │     load_hh_harmless         -> filter single-turn (one "\n\nHuman:" in `chosen`)
  │     _parse_chosen            -> {prompt, chosen reply} from the transcript
  │     build_tokenized_dataset  -> tokenize + COMPLETION-ONLY MASK (prompt -> -100)
  │     verify_masking           -> assert prompt masked / response supervised
  │     split_train_eval         -> 90% train / 10% eval (seed 42)
  ├─ SFTTrainer(processing_class=tokenizer, data_collator=DataCollatorForSeq2Seq, peft_config=None)
  ├─ trainer.train()                              # save_strategy="epoch" -> 1 adapter ckpt / epoch
  └─ tokenizer.save_pretrained(output_dir)
```

### Dataset + masking (`sft/data.py`) — the correctness-critical part
- **Single-turn filter.** hh-rlhf stores a full conversation in the `chosen` /
  `rejected` strings as repeated `\n\nHuman:` / `\n\nAssistant:` turns. We keep
  rows with exactly one `\n\nHuman:` marker in `chosen` (matches the project's
  `hh_rlhf_stats.py`); on `harmless-base` train that is **12,254** examples
  (logged, not hard-failed; `expected_single_turn_size` in the config).
- **Target = chosen reply.** `_parse_chosen` splits the transcript into the user
  prompt and the assistant reply from the `chosen` (more-harmless) transcript.
- **Completion-only label masking.** The full conversation is rendered with the
  Llama-3 chat template, then split at the assistant header
  `<|start_header_id|>assistant<|end_header_id|>\n\n`. The prompt scaffold and the
  response are tokenized **separately and concatenated**, so the mask boundary is
  exact by construction. Prompt/scaffold tokens get label `-100`; only the
  response (+ its closing `<|eot_id|>`, so the model learns to stop) is supervised.
  `verify_masking()` decodes and asserts this on a few examples.

### Training (`sft/training.py`)
- `SFTConfig` with the corrections in §4 below. LoRA is applied via
  `get_peft_model` here (not via `peft_config`) so we can call
  `enable_input_require_grads()` for gradient checkpointing.
- The dataset is pre-tokenized, so `dataset_kwargs={"skip_prepare_dataset": True}`
  tells TRL not to re-process it; `DataCollatorForSeq2Seq` pads dynamically and
  pads `labels` with `-100`.
- `save_only_model=True` → each `checkpoint-*/` is a clean adapter dir (no
  optimizer/RNG state), so the push is small and fast.
- **4 epochs, `save_strategy="epoch"` → exactly 4 evenly-spaced adapter
  checkpoints** (one per epoch, ~25/50/75/100% of training).

### Push (`sft/push.py`)
- Discovers `checkpoint-*` dirs (sorted by step = epoch order), maps them to
  `er-abd/<repo_prefix>-epoch{1,2,3,4}`, creates each repo **private**, uploads
  only adapter/config/tokenizer files + an auto-generated model card, and logs the
  commit hash. Each push is wrapped in try/except — a failure is reported loudly
  and the local checkpoint is left intact for a manual retry.

---

## 3. Setup — what YOU must provide

### 3.1 Credentials (`.env`)
Copy the template and fill in real values (`.env` is git-ignored — never commit it):
```bash
cp .env.example .env      # then edit HF_TOKEN
```
- **`HF_TOKEN`** — Hugging Face token with **write** access. Used to (a) download
  `meta-llama/Llama-3.2-3B-Instruct` (a **gated model**), (b) download the public
  hh-rlhf dataset, and (c) create and push the private adapter repos.
- **`OPENAI_API_KEY`** — *not* needed for training (only the downstream RO1
  monitor uses it). A warning is logged if absent; training proceeds.

### 3.2 Accept the Llama-3.2 license
`meta-llama/Llama-3.2-3B-Instruct` is **gated**. Visit its Hugging Face page while
logged in as the account behind `HF_TOKEN` and accept the license, or the model
download will 403. (hh-rlhf itself is public and needs no terms acceptance.)

### 3.3 Set your Hub account / repo names
In `configs/sft_harmlessness.yaml`: `hf_username` (default `er-abd`) and
`repo_prefix`. The four repos created are
`<hf_username>/<repo_prefix>-epoch{1,2,3,4}` plus a throwaway `…-sanitytest`.
The repos are created automatically (private) — just ensure the token has write scope.

### 3.4 RunPod environment
1. Start a CUDA GPU pod (A100 40GB or A6000/L40 etc.; ≥24 GB is comfortable for a
   3B model + LoRA + gradient checkpointing).
2. Keep this repo and `output_dir` under `/workspace` so they survive pod restarts.
3. Install deps (torch usually ships in the pod image — check first):
   ```bash
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   pip install -r requirements.txt
   ```
4. (Optional) For `attn_implementation: flash_attention_2`, install a matching
   `flash-attn` wheel; otherwise leave the default `sdpa` (no extra install).

---

## 4. Implementation notes (verified against the pinned stack)

Verified against transformers 4.56.2 / trl 0.22.2 / peft 0.18.1:

| # | Topic | Decision |
|---|---|---|
| 1 | Sequence length | Use `SFTConfig(max_length=...)` (the old `max_seq_length` was removed) |
| 2 | Completion-only loss | A plain `dataset_text_field` trains on prompt+response; instead we build explicit `{input_ids, labels}` with the prompt masked to `-100` and `verify_masking()` proves it |
| 3 | Trainer API | `SFTTrainer(processing_class=tokenizer, ...)`; pre-tokenized data + `DataCollatorForSeq2Seq` |
| 4 | Device placement | No `device_map="auto"` for training (that is the inference/model-parallel path and conflicts with the Trainer); the Trainer/accelerate place the model |
| 5 | Gradient checkpointing | `use_reentrant=False` + `enable_input_require_grads()` so grads flow with LoRA |
| 6 | Epochs / checkpoints | **4 epochs → 4 per-epoch checkpoints** at equal intervals |
| 7 | Push payload | `save_only_model=True` + `allow_patterns` → adapter-only (no optimizer state) |
| 8 | Mask boundary | Split the canonical full rendering at the Llama assistant header rather than relying on `add_generation_prompt` |

---

## 5. How to run (on RunPod, in order)

One-shot (preflight → sanity gate → train → push):
```bash
cd MVP-RO2-llama
bash run_sft.sh
```

Or step by step:
```bash
# 1) Sequence-length preflight: confirm/adjust max_length in the config.
python scripts/00_check_seq_length.py --config configs/sft_harmlessness.yaml

# 2) ENVIRONMENT GATE: tiny run + masking/loss/push assertions. Must PASS first.
python scripts/sanity_check_sft.py

# 3) Full training (4 epochs; 4 adapter checkpoints under output_dir).
python scripts/01_train_sft.py --config configs/sft_harmlessness.yaml

# 4) Push all 4 epoch adapters to private Hub repos.
python scripts/02_push_adapters.py --config configs/sft_harmlessness.yaml

# 5) (For RO1 eval) merge an epoch's adapter into a standalone model and push it.
python scripts/04_merge_and_push.py --config configs/sft_harmlessness.yaml --epoch 4
```

---

## 6. Outputs

- **Local:** `outputs/llama-3.2-3b-sft-harmlessness/checkpoint-*` — one LoRA
  adapter per epoch (adapter weights + config + tokenizer; no optimizer state).
- **Hub (private, adapter-only):**
  - `er-abd/llama-3.2-3b-sft-harmlessness-hh-adapter-epoch1`
  - `er-abd/llama-3.2-3b-sft-harmlessness-hh-adapter-epoch2`
  - `er-abd/llama-3.2-3b-sft-harmlessness-hh-adapter-epoch3`
  - `er-abd/llama-3.2-3b-sft-harmlessness-hh-adapter-epoch4` ← primary eval target

Load at eval time:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", torch_dtype="bfloat16")
model = PeftModel.from_pretrained(base, "er-abd/llama-3.2-3b-sft-harmlessness-hh-adapter-epoch4")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
```

---

## 7. Key decisions (and why)

- **Modern, verified stack (no Unsloth).** The code is written against and
  import-verified on transformers 4.56.2 / trl 0.22.2 / peft 0.18.1;
  `requirements.txt` pins them.
- **Self-implemented completion-only masking** rather than trusting an auto-mask:
  it is version-independent and explicitly verified by `verify_masking()`.
- **Adapter-only push of all 4 epoch checkpoints** (no best-checkpoint selection)
  — the SFT trajectory is itself of interest, and epoch 4 is the primary target.
- **Sanity gate first.** The tiny end-to-end run validates environment, data
  formatting, masking, loss behaviour, and the Hub push cheaply before the
  multi-hour run.
- **No CoT/thinking handling.** Llama-3.2-3B-Instruct has no thinking mode, so
  training simply supervises the chosen plain-text response; there is no
  `<think>` block to preserve or measure.
