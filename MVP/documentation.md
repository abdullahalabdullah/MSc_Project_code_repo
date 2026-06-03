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
