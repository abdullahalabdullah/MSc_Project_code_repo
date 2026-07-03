"""Central configuration for the harmlessness preference-pair pipeline.

All stages import their constants from here so a smoke run only needs edits in one
place (drop SAMPLE / N / MAX_TOKENS). Paths are all relative to this file's dir so the
pipeline runs the same on a laptop or a RunPod box.
"""
import os

# ── Reproducibility ────────────────────────────────────────────────────────
SEED = 0                      # RHiE convention (utils.set_seed)

# ── Sampling / generation ──────────────────────────────────────────────────
SAMPLE = 300                  # number of unique prompts to sample
N = 10                        # responses generated per prompt
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
#   throughput). WildJailbreak `adversarial_harmful` prompts are LONG (multi-paragraph
#   jailbreak wrappers), so give a generous 4096-token budget — a prompt longer than
#   (MAX_MODEL_LEN - MAX_TOKENS) would otherwise error/truncate. Still ~37x concurrency.
PROMPT_TOKEN_BUDGET = 4096
MAX_MODEL_LEN = MAX_TOKENS + PROMPT_TOKEN_BUDGET

# ── Models ─────────────────────────────────────────────────────────────────
GEN_MODEL = "Qwen/Qwen3-4B"
GUARD_MODEL = "Qwen/Qwen3Guard-Gen-4B"
COST_MODEL = "PKU-Alignment/beaver-7b-unified-cost"
WILDGUARD_MODEL = "allenai/wildguard"   # sanity-check reference ranker (Mistral-7B based)

# beaver-7b-unified-cost is LLaMA-1 based: a 2048-token context. Inputs longer than this
# are LEFT-truncated (drop the prompt head, keep the ASSISTANT answer + final token that
# the score head reads), and every such truncation is counted in the sanity report.
BEAVER_MAX_CTX = 2048

# ── Classifier / scorer batch sizes ────────────────────────────────────────
GUARD_BATCH = 32              # (prompt, answer) pairs per Qwen3Guard generate() call
GUARD_MAX_NEW_TOKENS = 128    # enough for "Safety: ...\nCategories: ..."
COST_BATCH = 8               # (prompt, answer) pairs per Beaver forward pass
WILDGUARD_BATCH = 16         # (prompt, answer) pairs per WildGuard generate() call
WILDGUARD_MAX_NEW_TOKENS = 32  # enough for the 3-line "Harmful request/refusal/response" verdict

# ── Dataset ────────────────────────────────────────────────────────────────
# WildJailbreak is GATED: accept the terms on the HF page and make HF_TOKEN available on
# the machine (see .env / env_setup.py). Stored as TSV, so load with delimiter="\t" and
# keep_default_na=False (empty cells must stay "" not NaN).
DATASET = "allenai/wildjailbreak"
DATASET_CONFIG = "train"          # "train" config (has the data_type column)
# We use ONLY the two harmful subsets. Each maps to the column that holds its prompt text:
# vanilla_harmful -> the plain request; adversarial_harmful -> the jailbreak-wrapped request.
WILDJAILBREAK_TYPE_TO_COLUMN = {
    "vanilla_harmful": "vanilla",
    "adversarial_harmful": "adversarial",
}

# ── Paths ──────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
# Output dir is overridable so a new run never clobbers an old one. run_all.sh sets
# RUN_OUTPUT_DIR to a fresh timestamped folder and exports it to every stage; if unset,
# it defaults to ./outputs. All stages in one run must share the same RUN_OUTPUT_DIR.
OUTPUT_DIR = os.environ.get("RUN_OUTPUT_DIR") or os.path.join(HERE, "outputs")

PROMPTS_PATH = os.path.join(OUTPUT_DIR, "01_prompts.joblib")
PROMPTS_META_PATH = os.path.join(OUTPUT_DIR, "01_prompts_meta.joblib")  # per prompt: subset + vanilla seed
RESPONSES_PATH = os.path.join(OUTPUT_DIR, "02_responses.joblib")
LABELED_PATH = os.path.join(OUTPUT_DIR, "03_labeled.joblib")
ALL_LABELED_PATH = os.path.join(OUTPUT_DIR, "03_all_labeled.joblib")  # every prompt+labels (debug)
SCORED_PATH = os.path.join(OUTPUT_DIR, "04_scored.joblib")
FINAL_PAIRS_PATH = os.path.join(OUTPUT_DIR, "final_pairs.joblib")

REPORT_PATH = os.path.join(OUTPUT_DIR, "report.md")
SAMPLE_PAIRS_PATH = os.path.join(OUTPUT_DIR, "sample_pairs.json")
COST_SPREAD_FIG_PATH = os.path.join(OUTPUT_DIR, "cost_spread.png")

# Beaver ranker sanity-check outputs (sanity_beaver.py)
SANITY_REPORT_PATH = os.path.join(OUTPUT_DIR, "beaver_sanity_report.md")  # standalone results (.md)
SANITY_DISAGREEMENT_PATH = os.path.join(OUTPUT_DIR, "beaver_sanity_disagreements.jsonl")
SANITY_SCORED_PATH = os.path.join(OUTPUT_DIR, "beaver_sanity_scored.joblib")


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
