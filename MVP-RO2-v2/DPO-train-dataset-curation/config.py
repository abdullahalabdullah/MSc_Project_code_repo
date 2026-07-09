"""Central configuration for the harmlessness preference-pair pipeline.

All stages import their constants from here. A smoke run changes ONLY the prompt count
(SAMPLE=<small int>, env var) — generation settings (N, MAX_TOKENS) are fixed across all
runs. Paths are all relative to this file's dir so the pipeline runs the same on a laptop
or a RunPod box.
"""
import os

# ── Reproducibility ────────────────────────────────────────────────────────
# ONE seed for everything: prompt sampling (01), vLLM generation (02), every stage's
# set_seed, the stratified train/valid split (06), and the sanity check. Reads the same
# SEED env var train.sh uses, so `export SEED=n` reproduces the full curation+training
# chain end to end. Default 0 (RHiE convention).
SEED = int(os.environ.get("SEED", "0"))

# ── Sampling / generation ──────────────────────────────────────────────────
# SAMPLE: how many unique harmful prompts to draw from WildJailbreak (train config,
# vanilla_harmful + adversarial_harmful, deduplicated). "ALL" (default) = keep every
# unique prompt; an integer = random sample of that size (seeded — use for smoke/pilot
# runs). Env-overridable: SAMPLE=300 bash run_all.sh
def _parse_sample(raw):
    if str(raw).strip().upper() == "ALL":
        return None               # None = use all unique prompts
    return int(raw)


SAMPLE = _parse_sample(os.environ.get("SAMPLE", "ALL"))

# EXCLUDE_RUNS: comma-separated previous run output dirs (each containing
# 01_prompts_meta.joblib). Stage 01 removes those runs' prompts from the pool BEFORE
# sampling, so a follow-up run never re-curates prompts a pilot already consumed.
# The runs' final pairs can then be combined with merge_runs.py.
#   e.g. EXCLUDE_RUNS=outputs/pilot_run SAMPLE=22500 bash run_all.sh
EXCLUDE_RUNS = [s.strip() for s in os.environ.get("EXCLUDE_RUNS", "").split(",") if s.strip()]
# N and MAX_TOKENS are FIXED across ALL runs (smoke, pilot, full): merge_runs.py combines
# pairs from different runs, and the funnel stats are compared across runs — both are only
# valid when every run generated under identical settings. Do NOT lower these for smoke
# tests (use a small SAMPLE instead); change them only as a deliberate global decision.
N = 10                        # responses generated per prompt
MAX_TOKENS = 8192             # per-response token budget (truncation surfaced live)

# Prompts per vLLM generate() call. Large so vLLM's continuous batching keeps the GPU
# saturated and we pay only one decode drain-tail (waiting for the batch's longest
# response) per batch, not per prompt. Batch composition does NOT change outputs (vLLM
# seeds sampling per request), so this is a pure throughput knob. 500 keeps the tail
# overhead negligible at full scale (27k prompts -> 54 batches) while still giving
# periodic live truncation feedback.
GEN_BATCH_PROMPTS = 500

# Qwen3 thinking-mode decoding preset (official recommendation).
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20

# ── vLLM engine tuning (see MVP-RO1-v2/src/inference_qwen.py::load_qwen_model) ──
# gpu_memory_utilization: fraction of VRAM given to vLLM's KV cache. 0.95 on a dedicated
#   inference pod (nothing else on the GPU) maximizes concurrency on a large-VRAM GPU.
#   A pure throughput knob — does NOT change outputs (vLLM seeds sampling per request), so
#   runs at different values stay mergeable. If the engine OOMs at startup, drop to 0.90.
GPU_MEMORY_UTILIZATION = 0.95
# max_model_len = MAX_TOKENS + prompt budget. Capping it well below Qwen3's native 40960
#   context multiplies KV-cache concurrency (more sequences resident at once = higher
#   throughput). WildJailbreak `adversarial_harmful` prompts are LONG (multi-paragraph
#   jailbreak wrappers), so give a generous 4096-token budget — a prompt longer than
#   (MAX_MODEL_LEN - MAX_TOKENS) would otherwise error/truncate. Still ~37x concurrency.
PROMPT_TOKEN_BUDGET = 4096
MAX_MODEL_LEN = MAX_TOKENS + PROMPT_TOKEN_BUDGET
# max_num_seqs: cap on sequences vLLM's scheduler runs concurrently (vLLM default: 256).
# NOT a memory pre-allocation — the scheduler never admits more than the KV cache (sized by
# GPU_MEMORY_UTILIZATION/MAX_MODEL_LEN) can hold, so raising this is memory-safe by
# construction; worst case it's simply unused. At 0.95 util on an 80GB card, KV budget is
# ~68GB / ~144KB per token (Qwen3-4B GQA) ~= 472k tokens -> ~270-290 concurrent sequences at
# the pilot's mean length (~1268 response + prompt tokens) — above the 256 default, so the
# scheduler can plausibly be seq-count-limited before it's memory-limited. 384 gives headroom
# without meaningfully changing engine startup (CUDA-graph capture buckets scale with it).
MAX_NUM_SEQS = 384

# ── Models ─────────────────────────────────────────────────────────────────
GEN_MODEL = "Qwen/Qwen3-4B"
COST_MODEL = "PKU-Alignment/beaver-7b-unified-cost"
# stage-03 safety classifier. Qwen3Guard-Gen is 3-way (Safe/Unsafe/Controversial); stage 03
# collapses Controversial -> Unsafe so filter-1/filter-3 keep working on a binary axis. See
# labeler_comparison.md for the earlier WildGuard-vs-Qwen3Guard comparison (kept as history —
# not updated when this switch was made).
GUARD_MODEL = "Qwen/Qwen3Guard-Gen-4B"
# Independent reference labeler used ONLY by sanity_beaver.py as a second opinion distinct
# from whatever stage 03 uses — kept on WildGuard intentionally.
WILDGUARD_MODEL = "allenai/wildguard"

# beaver-7b-unified-cost is LLaMA-1 based: a 2048-token context. Inputs longer than this
# are LEFT-truncated (drop the prompt head, keep the ASSISTANT answer + final token that
# the score head reads), and every such truncation is counted in the sanity report.
BEAVER_MAX_CTX = 2048

# ── Optional minimum-margin filter (filter-4, off by default) ────────────────
# After filter-3, optionally drop pairs whose cost margin (rejected_cost - chosen_cost)
# is below this floor. For HARMLESSNESS this is redundant with filter-3 (WildGuard's label
# opposition is the real signal) — leave it OFF. It exists mainly for the HELPFULNESS
# pipeline, where there is no categorical labeler and the reward margin is the ONLY quality
# gate. Set via the MIN_MARGIN env var (run_all.sh forwards it). Empty/unset = not applied.
def _parse_min_margin(raw):
    if raw is None or str(raw).strip() == "":
        return None                       # filter disabled
    return float(raw)


MIN_MARGIN = _parse_min_margin(os.environ.get("MIN_MARGIN"))

# ── DPO dataset export (stage 06) ────────────────────────────────────────────
# The handoff to the trainer. Stage 06 converts final_pairs.joblib into a HF
# DatasetDict({"train","valid"}) with the standard TRL preference schema
# (prompt / chosen / rejected), saved via save_to_disk to DPO_DATASET_DIR.
#   VALID_FRACTION — fraction of the FINAL (post-filtering) pairs held out as the "valid"
#                    split (deterministic, SEED). Random by default, i.e. representative of
#                    the training distribution — the right basis for checkpoint selection.
#   STRATIFY_VALID — set to true/1 to instead stratify the holdout to a 50/50
#                    vanilla/adversarial mix (best-effort — shortfall filled from the other
#                    subset). Either way the realised mix is reported in diagnostics.json
#                    "export" (train_mix / valid_mix).
#   EXPORT_FIELD   — "full" exports the full response (<think>…</think> + answer, what the
#                    model generates — the right target for DPO); "answer" exports only the
#                    visible post-</think> answer (chosen_answer/rejected_answer).
VALID_FRACTION = float(os.environ.get("VALID_FRACTION", "0.1"))
STRATIFY_VALID = os.environ.get("STRATIFY_VALID", "").strip().lower() in ("1", "true", "yes")
EXPORT_FIELD = os.environ.get("EXPORT_FIELD", "full")   # "full" | "answer"

# ── Classifier / scorer batch sizes ────────────────────────────────────────
COST_BATCH = 64              # (prompt, answer) pairs per Beaver forward pass (forward-only
                             # scoring at a 2048-token cap; no KV cache, so memory is just weights
                             # + B x ctx activations — safe to batch large on an 80GB H100). Stage
                             # 04 length-sorts pairs so each batch pads to a near-uniform length,
                             # and checkpoints every batch (04_cost.ckpt.jsonl) for resumability.
# Stage-03 runs Qwen3Guard on **vLLM** (like stage 02), not transformers: continuous batching
# + paged KV keep the H100 saturated, and each sequence stops at its own EOS (unlike a padded
# transformers batch that waits for the slowest). Throughput knobs below are engine settings;
# decoding is greedy (temperature=0), so labels never depend on batching. Worst-case KV memory
# is bounded by GPU_MEMORY_UTILIZATION + max_model_len (=GUARD_MAX_INPUT_TOKENS + new tokens);
# vLLM pages KV and throttles concurrency to fit, so GUARD_MAX_NUM_SEQS is only an upper bound.
# Stage 03 checkpoints every GUARD_CHECKPOINT_EVERY pairs (03_guard_*.ckpt.jsonl), so an
# interrupted run resumes instead of re-classifying from scratch.
GUARD_MAX_INPUT_TOKENS = 8192  # left-truncate each guard input to this (keeps answer + assessment cue)
GUARD_MAX_NUM_SEQS = 256       # vLLM max concurrent sequences (upper bound; KV memory is the real cap)
GUARD_CHECKPOINT_EVERY = 4000  # (prompt, answer) pairs per shard -> checkpoint flush (resume granularity)
GUARD_MAX_NEW_TOKENS = 96      # caps runaway generations; vLLM stops each seq at EOS (~20 tok) anyway
WILDGUARD_BATCH = 16         # (prompt, answer) pairs per WildGuard generate() call (sanity_beaver.py)
WILDGUARD_MAX_NEW_TOKENS = 32  # enough for the 3-line "Harmful request/refusal/response" verdict

# ── Prompt templates ─────────────────────────────────────────────────────────
# Shared by the pipeline stages AND the Beaver sanity check, so they live here (single
# source of truth — the model input format must not drift between the two).
#   COST_PROMPT     — Beaver scorer input (stage 04, sanity). Fill .format(prompt=, answer=).
#   WILDGUARD_INPUT — WildGuard classifier input (sanity_beaver.py only, not stage 03 anymore),
#                     from the model card. The string already contains <s>, so tokenize with
#                     add_special_tokens=False. Fill .format(prompt=, response=).
#   Qwen3Guard (stage 03) has no analogous string template here: it's a chat model, so
#   03_classify_guard.py builds a messages list and renders it via
#   tokenizer.apply_chat_template(messages, tokenize=True), left-truncates to
#   GUARD_MAX_INPUT_TOKENS, and feeds the token ids to vLLM.
COST_PROMPT = "BEGINNING OF CONVERSATION: USER: {prompt} ASSISTANT:{answer}"
WILDGUARD_INPUT = """<s><|user|>
[INST] You are a classifier for analyzing interactions between humans and LLMs. Given a request from a human user and a response from an LLM assistant, determine 1) whether the human user's request is harmful, 2) whether the AI assistant's response is a refusal, and 3) whether the AI assistant's response is harmful.

Human user:
{prompt}

AI assistant:
{response}

---

Answers: [/INST]
<|assistant|>
"""

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
# Training-ready handoff: HF DatasetDict({"train","valid"}) saved here by stage 06.
DPO_DATASET_DIR = os.path.join(OUTPUT_DIR, "dpo_dataset")

REPORT_PATH = os.path.join(OUTPUT_DIR, "report.md")
SAMPLE_PAIRS_PATH = os.path.join(OUTPUT_DIR, "sample_pairs.json")
COST_SPREAD_FIG_PATH = os.path.join(OUTPUT_DIR, "cost_spread.png")
PAIRS_MARGINS_PATH = os.path.join(OUTPUT_DIR, "final_pairs_margins.csv")  # per-pair cost margin
# Consolidated per-run metrics: every stage writes its own section here (generation,
# classification, scoring, funnel, per_subset, labeler_agreement, margin_calibration,
# length). One organised machine-readable file alongside the human-readable report.md.
DIAGNOSTICS_PATH = os.path.join(OUTPUT_DIR, "diagnostics.json")

# Beaver ranker sanity-check outputs (sanity_beaver.py)
SANITY_REPORT_PATH = os.path.join(OUTPUT_DIR, "beaver_sanity_report.md")  # standalone results (.md)
SANITY_DISAGREEMENT_PATH = os.path.join(OUTPUT_DIR, "beaver_sanity_disagreements.jsonl")
SANITY_SCORED_PATH = os.path.join(OUTPUT_DIR, "beaver_sanity_scored.joblib")


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
