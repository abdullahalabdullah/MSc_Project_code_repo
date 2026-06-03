# MVP — CoT Monitorability Evaluation (2026-06-03)

Implements the full evaluation pipeline from MVP.md: MMLU × Stanford-professor hint × four Tulu 8B checkpoints.

---

## Folder Structure

```
MVP/
├── config/
│   └── mvp_config.yaml          All experiment parameters (dataset, models, inference, monitor, analysis, paths)
├── src/
│   ├── dataset.py               Load MMLU, sample 200 examples (≤5/subject), assign incorrect hint targets
│   ├── inference.py             vLLM inference (N=8 per arm), prompt formatting, answer parsing, JSONL writing
│   ├── monitor.py               GPT-4o 7-question structured monitor with exponential-backoff retry
│   ├── metrics.py               TE, δ, S, per-instance ack, HAR computation
│   └── analysis.py              Confusion matrix + TE-vs-ack scatter plots (matplotlib)
├── scripts/
│   ├── 01_prepare_dataset.py    Produces data/dataset_index.json
│   ├── 02_run_inference.py      Produces outputs/{model}/*.jsonl  (one run per checkpoint)
│   ├── 03_run_monitor.py        Produces monitor_outputs/{model}/monitor_results.jsonl
│   ├── 04_compute_metrics.py    Produces analysis/{model}/per_instance_te.csv + har_summary.json
│   ├── 05_analyze.py            Produces plots/*.png + plots/summary_table.json
│   └── 06_extract_qualitative.py Produces plots/{model}_qualitative.txt (covert-following examples)
├── data/                        gitignored — runtime data
├── outputs/                     gitignored — raw JSONL generations
├── monitor_outputs/             gitignored — raw monitor API responses
├── analysis/                    gitignored — computed metrics
├── plots/                       gitignored — figures and tables
├── requirements.txt
├── .gitignore
└── documentation.md            This file
```

---

## Script I/O Summary

| Script | CLI args | Input | Output |
|---|---|---|---|
| `01_prepare_dataset.py` | `[--config]` | cais/mmlu via HuggingFace | `data/dataset_index.json` |
| `02_run_inference.py` | `--model <name>` `[--seed]` | `data/dataset_index.json` | `outputs/{model}/hinted.jsonl`, `unhinted.jsonl`, `progress.json` |
| `03_run_monitor.py` | `--model <name>` | `outputs/{model}/hinted.jsonl` | `monitor_outputs/{model}/monitor_results.jsonl`, `monitor_progress.json` |
| `04_compute_metrics.py` | `[--model <name>]` | hinted + unhinted + monitor JSONL | `analysis/{model}/per_instance_te.csv`, `har_summary.json` |
| `05_analyze.py` | `[--config]` | per_instance_te.csv + JSONL files | `plots/{model}_confusion_matrix.{png,json}`, `te_ack_scatter_all_models.png`, `summary_table.json` |
| `06_extract_qualitative.py` | `[--n 15]` | per_instance_te.csv + JSONL files | `plots/{model}_qualitative.txt` |

---

## JSONL Record Fields

**Inference output** (`hinted.jsonl` / `unhinted.jsonl`):
- `instance_id`, `question_id`, `arm` (hinted/unhinted), `sample_idx` (0–7)
- `hint_target`, `correct_answer`
- `full_prompt` — exact string passed to the model (after chat template)
- `raw_output` — full CoT + answer, untruncated
- `parsed_answer` — extracted letter (A–D) or `null`
- `followed_hint` — bool (`parsed_answer == hint_target`)
- `no_cot` — bool (output has <30 chars of non-answer content)
- `token_count`, `truncated`, `seed`

**Monitor output** (`monitor_results.jsonl`):
- `instance_id`, `sample_idx`, `hint_target`, `followed_hint`, `no_cot`
- `q1`–`q7` — bool answers to the seven structured questions
- `acknowledged` — bool (Final Verdict == "Acknowledged")
- `label` — "Acknowledged" / "Not Acknowledged" / null
- `summary` — monitor's 2–3 sentence explanation
- `raw_response` — full GPT-4o output string
- `parse_failure` — present (True) only when verdict line is missing

---

## Pipeline Run Order

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare dataset (once; deterministic via seed=42)
python scripts/01_prepare_dataset.py

# 3. Run inference for each checkpoint (requires GPU; ~1-2h per checkpoint on A100)
python scripts/02_run_inference.py --model tulu_sft_no_safety
python scripts/02_run_inference.py --model tulu_sft_full
python scripts/02_run_inference.py --model tulu_dpo
python scripts/02_run_inference.py --model tulu_final

# 4. Run monitor (requires OPENAI_API_KEY; ~1600 GPT-4o calls per model)
OPENAI_API_KEY=<key> python scripts/03_run_monitor.py --model tulu_sft_no_safety
OPENAI_API_KEY=<key> python scripts/03_run_monitor.py --model tulu_sft_full
OPENAI_API_KEY=<key> python scripts/03_run_monitor.py --model tulu_dpo
OPENAI_API_KEY=<key> python scripts/03_run_monitor.py --model tulu_final

# 5. Compute metrics
python scripts/04_compute_metrics.py

# 6. Generate plots
python scripts/05_analyze.py

# 7. Extract qualitative examples for manual annotation
python scripts/06_extract_qualitative.py
```

---

## Key Design Decisions

- **vLLM `n=8`**: All 8 samples per instance are generated in one `generate()` call using `SamplingParams(n=8)`. Diversity comes from temperature=0.8 sampling.
- **Resume-on-restart**: Both inference and monitor scripts write a `progress.json` checkpoint every 50 instances and skip already-completed work on restart.
- **δ threshold**: `max(25th percentile of all TE_i, 0.20)` — computed per model from its own TE distribution.
- **HAR denominator**: Computed only over hint-following runs within S; assertion guards against zero-denominator.
- **Monitor**: GPT-4o at temperature=0 for determinism; exponential backoff (up to 5 retries) for rate-limit resilience.
- **Prompt byte-for-byte consistency**: A single `build_user_content(item, include_hint)` function with an `include_hint` flag ensures hinted/unhinted prompts differ only in the inserted hint line.








1. Provision the Pod on RunPod

- GPU: A40 (48 GB)
- Container disk: 30 GB
- Volume disk: 30 GB
- Template: any PyTorch/CUDA image (e.g. runpod/pytorch:2.1.0-py3.10-cuda11.8.0)

---
2. First-Time Setup (run once after the pod starts)

# Pin HuggingFace cache to the persistent volume
export HF_HOME=/workspace/hf_cache
export OPENAI_API_KEY="sk-..."   # your OpenAI key for step 3

# Clone your repo
cd /workspace
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO/MVP

# Install dependencies
pip install -r requirements.txt

Add both exports to ~/.bashrc so they survive a pod restart:
echo 'export HF_HOME=/workspace/hf_cache' >> ~/.bashrc
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.bashrc

---
3. Prepare the Dataset (run once)

python scripts/01_prepare_dataset.py

Produces data/dataset_index.json — 200 MMLU instances with hint assignments.

---
4. Inference + Monitor — repeat for each of the 4 models

Run this block once per model, in order. After each model finishes inference, delete its weights before downloading the next one (keeps volume under 30 GB).

Model 1

python scripts/02_run_inference.py --model tulu_sft_no_safety
python scripts/03_run_monitor.py --model tulu_sft_no_safety
rm -rf /workspace/hf_cache/hub/models--allenai--Llama-3.1-Tulu-3-8B-SFT-no-safety-data

Model 2

python scripts/02_run_inference.py --model tulu_sft_full
python scripts/03_run_monitor.py --model tulu_sft_full
rm -rf /workspace/hf_cache/hub/models--allenai--Llama-3.1-Tulu-3-8B-SFT

Model 3

python scripts/02_run_inference.py --model tulu_dpo
python scripts/03_run_monitor.py --model tulu_dpo
rm -rf /workspace/hf_cache/hub/models--allenai--Llama-3.1-Tulu-3-8B-DPO

Model 4

python scripts/02_run_inference.py --model tulu_final
python scripts/03_run_monitor.py --model tulu_final
rm -rf /workspace/hf_cache/hub/models--allenai--Llama-3.1-Tulu-3-8B

▎ Step 03 calls GPT-4o via API — no GPU needed for it. Step 02 is the GPU-heavy step (~1–2 hours per model).

---
5. Metrics and Analysis (run once, after all 4 models are done)

python scripts/04_compute_metrics.py
python scripts/05_analyze.py
python scripts/06_extract_qualitative.py

---
6. Save Your Outputs Before Deleting the Pod

Your results live in /workspace/YOUR_REPO/MVP/outputs/, monitor_outputs/, analysis/, and plots/. Copy them off before terminating:

# Option A — push to GitHub
git add outputs/ monitor_outputs/ analysis/ plots/
git commit -m "Add MVP results"
git push

# Option B — copy to your local machine (run this on your laptop, not the pod)
scp -r root@<POD_IP>:/workspace/YOUR_REPO/MVP/outputs/ ./
scp -r root@<POD_IP>:/workspace/YOUR_REPO/MVP/analysis/ ./

Then terminate the pod on RunPod.