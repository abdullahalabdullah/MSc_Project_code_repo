# MVP-RO2 — SFT (Harmlessness) Training: Code Documentation

This documents the code that implements the **RO2 SFT MVP**: LoRA supervised
fine-tuning of `Qwen/Qwen3-4B` on the **safe (chosen) responses** of
PKU-SafeRLHF *conflict cases*, saving one adapter checkpoint per epoch and
pushing all three (private) to the Hugging Face Hub for the existing RO1 eval
pipeline.

The spec is `sft_mvp_ro2_experiment_design.md` in this folder. The spec's code
is illustrative; this implementation **corrects** several issues in it (see §4).
Evaluation is out of scope (spec §10) — the adapters are the hand-off.

---

## 1. Directory structure

```
MVP-RO2/
├── sft_mvp_ro2_experiment_design.md   # the spec (input)
├── documentation.md                   # this file
├── requirements.txt                   # pinned modern stack (no Unsloth)
├── .env.example                       # credential template -> copy to .env
├── .gitignore                         # ignores .env, outputs/, caches, weights
├── run_sft.sh                         # one-shot launcher: preflight -> gate -> train -> push
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
│   ├── data.py                        # PKU load/filter, completion-only masking, verify, seq-length report
│   ├── model.py                       # Qwen3-4B + tokenizer load, LoRA config
│   ├── training.py                    # SFTConfig + SFTTrainer assembly; run_training()
│   ├── push.py                        # discover epoch checkpoints -> push adapters + model cards
│   └── generate.py                    # thinking-mode generation check
│
└── scripts/                           # thin entry points (argparse); import the sft package
    ├── _bootstrap.py                  # puts repo root on sys.path
    ├── 00_check_seq_length.py         # spec §4 preflight -> recommends max_length
    ├── 01_train_sft.py                # main training run
    ├── 02_push_adapters.py            # spec §8.2 push all 3 epoch adapters
    ├── 03_sanity_generation.py        # spec §9.1 per-checkpoint CoT preservation check
    └── sanity_check_sft.py            # ENVIRONMENT GATE — tiny end-to-end run + assertions
```

> The earlier DPO attempt lives in the sibling `../MVP-RO2-old/`. This SFT code is
> a fresh, modular reimplementation (plain PEFT + TRL `SFTTrainer`, **no Unsloth**).

---

## 2. Code structure and data flow

All logic is in the `sft/` package; `scripts/` are thin wrappers. The sanity gate
reuses the exact same `run_training()` path as the real run, just on a tiny slice.

```
run_training(cfg, logger)                         # sft/training.py
  ├─ check_credentials(logger)                    # .env -> HF_TOKEN (fail fast) + hf login
  ├─ load_tokenizer(cfg) / load_model(cfg)        # Qwen3-4B bf16, padding_side=right, use_cache=False
  ├─ get_peft_model(model, LoraConfig)            # LoRA r16/α32; enable_input_require_grads()
  ├─ _prepare_data(cfg, tokenizer)                # sft/data.py:
  │     load_pku_conflict        -> filter is_response_0_safe != is_response_1_safe
  │     _extract_safer           -> chosen = response_{safer_response_id}
  │     build_tokenized_dataset  -> tokenize + COMPLETION-ONLY MASK (prompt -> -100)
  │     verify_masking           -> assert prompt masked / response supervised (spec §6)
  │     split_train_eval         -> 90% train / 10% eval (seed 42)
  ├─ SFTTrainer(processing_class=tokenizer, data_collator=DataCollatorForSeq2Seq, peft_config=None)
  ├─ trainer.train()                              # save_strategy="epoch" -> 1 adapter ckpt / epoch
  └─ tokenizer.save_pretrained(output_dir)
```

### Dataset + masking (`sft/data.py`) — the correctness-critical part
- **Conflict filter** (spec §3.2): keep rows where exactly one response is safe
  (`is_response_0_safe != is_response_1_safe`). This is the definition that yields
  the spec's expected ~12,616 examples (logged, not hard-failed).
- **Chosen = safer** response (`response_{safer_response_id}`), spec §3.3.
- **Completion-only label masking.** The full conversation is rendered with the
  Qwen3 chat template (`enable_thinking=False`, spec §2/§3.4), then split at the
  assistant header `<|im_start|>assistant\n`. The prompt scaffold and the response
  are tokenized **separately and concatenated**, so the mask boundary is exact by
  construction. Prompt tokens get label `-100`; only the response (+ its closing
  `<|im_end|>`) is supervised. `verify_masking()` decodes and asserts this on a few
  examples (satisfies spec §6 and the checklist item).

### Training (`sft/training.py`)
- `SFTConfig` per spec §5, with the corrections in §4 below. LoRA is applied via
  `get_peft_model` here (not via `peft_config`) so we can call
  `enable_input_require_grads()` for gradient checkpointing.
- The dataset is pre-tokenized, so `dataset_kwargs={"skip_prepare_dataset": True}`
  tells TRL not to re-process it; `DataCollatorForSeq2Seq` pads dynamically and
  pads `labels` with `-100`.
- `save_only_model=True` → each `checkpoint-*/` is a clean adapter dir (no
  optimizer/RNG state), so the push is small and fast.

### Push (`sft/push.py`)
- Discovers `checkpoint-*` dirs (sorted by step = epoch order), maps them to
  `er-abd/<repo_prefix>-epoch{1,2,3}`, creates each repo **private**, uploads only
  adapter/config/tokenizer files + an auto-generated model card, and logs the
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
  `Qwen/Qwen3-4B`, (b) download the **gated** PKU-SafeRLHF dataset, and (c) create
  and push the private adapter repos.
- **`OPENAI_API_KEY`** — *not* needed for training (only the downstream RO1
  monitor uses it). A warning is logged if absent; training proceeds.

### 3.2 Accept the PKU-SafeRLHF dataset terms
`PKU-Alignment/PKU-SafeRLHF` is **gated**. Visit its Hugging Face page while logged
in as the account behind `HF_TOKEN` and accept the terms, or the dataset download
will 403.

### 3.3 Set your Hub account / repo names
In `configs/sft_harmlessness.yaml`: `hf_username` (default `er-abd`) and
`repo_prefix`. The three repos created are
`<hf_username>/<repo_prefix>-epoch{1,2,3}` plus a throwaway `…-sanitytest`.
The repos are created automatically (private) — just ensure the token has write scope.

### 3.4 RunPod environment
1. Start a CUDA GPU pod (A100 40GB or A6000/L40 etc.; ≥40 GB is comfortable for a
   4B model + LoRA + gradient checkpointing).
2. Keep this repo and `output_dir` under `/workspace` so they survive pod restarts.
3. Install deps (torch usually ships in the pod image — check first):
   ```bash
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   pip install -r requirements.txt
   ```
4. (Optional) For `attn_implementation: flash_attention_2`, install a matching
   `flash-attn` wheel; otherwise leave the default `sdpa` (no extra install).

### 3.5 Security note (action required)
`../MVP-RO2-old/.env.example` contains a **real-looking `HF_TOKEN`** that was
committed to git. **Revoke that token** in your Hugging Face settings and issue a
new one for `.env` here. This repo's `.env.example` contains placeholders only.

---

## 4. Corrections made to the spec's example code

The spec's code is illustrative. Verified against the pinned stack
(transformers 4.56.2 / trl 0.22.2 / peft 0.18.1), the following were fixed:

| # | Spec example | Problem | Fix in this code |
|---|---|---|---|
| 1 | `SFTConfig(max_seq_length=2048)` | Field removed in current TRL → `TypeError` | Use `max_length` (`sft/training.py`) |
| 2 | "`dataset_text_field="text"` masks the prompt automatically" (§6) | **False** — a plain text field trains on prompt+response | Explicit completion-only masking + `verify_masking()` (`sft/data.py`) |
| 3 | `SFTTrainer(tokenizer=...)`, `DataCollatorForCompletionOnlyLM` | `tokenizer=` renamed; collator not importable from `trl` 0.22 | `processing_class=`; pre-tokenized data + `DataCollatorForSeq2Seq` |
| 4 | `device_map="auto"` for training | Wrong for single-GPU training (inference/model-parallel path); conflicts with the Trainer | No `device_map`; the Trainer places the model (`sft/model.py`) |
| 5 | `gradient_checkpointing=True` only | Grads don't flow with LoRA unless configured | `use_reentrant=False` + `enable_input_require_grads()` |
| 6 | §7 says "Epochs: 2" | Contradicts the config/checkpoint list (3) | **3 epochs → 3 checkpoints** |
| 7 | push uploads the whole checkpoint dir | Would include large optimizer state | `save_only_model=True` + `allow_patterns` → adapter-only |
| 8 | `add_generation_prompt=True` to find the mask boundary | With `enable_thinking=False`, Qwen3 injects an empty `<think></think>` only into the generation prompt → boundary mismatch | Split the canonical full rendering at the assistant header instead |

---

## 5. How to run (on RunPod, in order)

One-shot (preflight → sanity gate → train → push):
```bash
cd MVP-RO2
bash run_sft.sh
```

Or step by step:
```bash
# 1) Sequence-length preflight (spec §4): confirm/adjust max_length in the config.
python scripts/00_check_seq_length.py --config configs/sft_harmlessness.yaml

# 2) ENVIRONMENT GATE: tiny run + masking/loss/push assertions. Must PASS first.
python scripts/sanity_check_sft.py

# 3) Full training (~3 epochs; 3 adapter checkpoints under output_dir).
python scripts/01_train_sft.py --config configs/sft_harmlessness.yaml

# 4) Push all 3 epoch adapters to private Hub repos.
python scripts/02_push_adapters.py --config configs/sft_harmlessness.yaml

# 5) Per-epoch thinking-mode check (spec §9.1): confirm <think> blocks survive.
python scripts/03_sanity_generation.py --config configs/sft_harmlessness.yaml \
    --checkpoint er-abd/qwen3-4b-sft-harmlessness-pku-adapter-epoch3
```

---

## 6. Outputs

- **Local:** `outputs/qwen3-4b-sft-harmlessness/checkpoint-*` — one LoRA adapter
  per epoch (adapter weights + config + tokenizer; no optimizer state).
- **Hub (private, adapter-only):**
  - `er-abd/qwen3-4b-sft-harmlessness-pku-adapter-epoch1`
  - `er-abd/qwen3-4b-sft-harmlessness-pku-adapter-epoch2`
  - `er-abd/qwen3-4b-sft-harmlessness-pku-adapter-epoch3` ← primary eval target

Load at eval time:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", torch_dtype="bfloat16")
model = PeftModel.from_pretrained(base, "er-abd/qwen3-4b-sft-harmlessness-pku-adapter-epoch3")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
```

---

## 7. Key decisions (and why)

- **Modern, verified stack (no Unsloth).** The spec gives version floors only.
  The code is written against and import-verified on transformers 4.56.2 /
  trl 0.22.2 / peft 0.18.1; `requirements.txt` pins them.
- **Self-implemented completion-only masking** rather than trusting an
  auto-mask the spec mis-attributes to `SFTTrainer`. It is version-independent and
  explicitly verified, directly satisfying spec §6.
- **Adapter-only push of all 3 epoch checkpoints** (no best-checkpoint selection,
  unlike the old DPO run) — the SFT trajectory is itself of interest, and epoch 3
  is the primary target.
- **Sanity gate first.** The tiny end-to-end run validates environment, data
  formatting, masking, loss behaviour, and the Hub push cheaply before the
  multi-hour run — mirroring the old DPO repo's gating philosophy.
- **`enable_thinking=False` for training, `=True` for the generation check.**
  Training targets are plain safe responses; the model's reasoning ability is
  preserved (think tokens are never training targets) and re-checked per epoch.
