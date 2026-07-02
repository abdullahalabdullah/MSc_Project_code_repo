"""Central configuration for the harmlessness preference-pair pipeline.

All stages import their constants from here so a smoke run only needs edits in one
place (drop SAMPLE / N / MAX_TOKENS). Paths are all relative to this file's dir so the
pipeline runs the same on a laptop or a RunPod box.
"""
import os

# ── Reproducibility ────────────────────────────────────────────────────────
SEED = 0                      # RHiE convention (utils.set_seed)

# ── Sampling / generation ──────────────────────────────────────────────────
SAMPLE = 300                 # number of unique prompts to sample
N = 8                        # responses generated per prompt
MAX_TOKENS = 8192             # per-response token budget (truncation surfaced live)

# Prompts per vLLM generate() call. Large so vLLM's continuous batching keeps the GPU
# saturated and we pay only one decode drain-tail per batch (not per prompt). Truncation
# is still reported after each batch, so 300/100 = 3 batches gives periodic live feedback
# while staying near single-call throughput.
GEN_BATCH_PROMPTS = 100

# Qwen3 thinking-mode decoding preset (official recommendation).
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20

# ── vLLM engine tuning (see MVP-RO1-v2/src/inference_qwen.py::load_qwen_model) ──
# gpu_memory_utilization: fraction of VRAM given to vLLM's KV cache. 0.90 is safe on a
#   dedicated inference pod and maximizes concurrency on a large-VRAM GPU.
GPU_MEMORY_UTILIZATION = 0.90
# max_model_len = MAX_TOKENS + prompt budget. Capping it well below Qwen3's native 40960
#   context multiplies KV-cache concurrency (more sequences resident at once = higher
#   throughput). PKU-SafeRLHF prompts are short (<~300 tokens); 2048 is ample headroom.
#   Do NOT set below MAX_TOKENS + observed_max_prompt_len or outputs truncate silently.
PROMPT_TOKEN_BUDGET = 2048
MAX_MODEL_LEN = MAX_TOKENS + PROMPT_TOKEN_BUDGET

# ── Models ─────────────────────────────────────────────────────────────────
GEN_MODEL = "Qwen/Qwen3-4B"
GUARD_MODEL = "Qwen/Qwen3Guard-Gen-4B"
COST_MODEL = "PKU-Alignment/beaver-7b-unified-cost"

# ── Classifier / scorer batch sizes ────────────────────────────────────────
GUARD_BATCH = 32              # (prompt, answer) pairs per Qwen3Guard generate() call
GUARD_MAX_NEW_TOKENS = 128    # enough for "Safety: ...\nCategories: ..."
COST_BATCH = 8               # (prompt, answer) pairs per Beaver forward pass

# ── Dataset ────────────────────────────────────────────────────────────────
DATASET = "PKU-Alignment/PKU-SafeRLHF-30K"
DATASET_SPLIT = "train"
PROMPT_COLUMN = "prompt"

# ── Paths ──────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "outputs")

PROMPTS_PATH = os.path.join(OUTPUT_DIR, "01_prompts.joblib")
RESPONSES_PATH = os.path.join(OUTPUT_DIR, "02_responses.joblib")
LABELED_PATH = os.path.join(OUTPUT_DIR, "03_labeled.joblib")
SCORED_PATH = os.path.join(OUTPUT_DIR, "04_scored.joblib")
FINAL_PAIRS_PATH = os.path.join(OUTPUT_DIR, "final_pairs.joblib")

REPORT_PATH = os.path.join(OUTPUT_DIR, "report.md")
SAMPLE_PAIRS_PATH = os.path.join(OUTPUT_DIR, "sample_pairs.json")
COST_SPREAD_FIG_PATH = os.path.join(OUTPUT_DIR, "cost_spread.png")


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
