"""Central configuration for the anti-sycophancy preference-pair pipeline.

All stages import their constants from here. The pipeline implements the frozen spec
(anti_sycophancy_dpo_dataset_curation_spec.md): every numeric threshold and seed below is
FIXED by the spec — do not tune them per run. The only run-level knobs are env vars:

    X                 training-pool size (default 5 = smoke test; pilot 500; full run 20000)
    RUN_OUTPUT_DIR    per-run output folder (drivers set this; default ./outputs)
    N_FILTER1         samples per prompt in Filter-1 (default 10; the spec's documented
                      escalation path is 16 — owner decision after a failed pilot gate 2)
    DROPPED_TEMPLATES comma-separated template ids dropped by pilot gate 3 (e.g. "T7")
    PILOT             set to 1 for the pilot run: stage 09 skips the 3000-pair floor and
                      stage 11 additionally writes pilot_gates.json
    X_FULL            planned full-run pool size, used by pilot gate 2 (default 20000)
    VALID_FRACTION    valid-split fraction for the DatasetDict export (default 0.1)
    GEN_SEED          vLLM sampling seed (default 0). NOT in the spec's seed table (the
                      spec fixes only the CPU-side seeds); recorded in manifest.json.

Paths are relative to this file's dir so the pipeline runs the same on a laptop or a
RunPod box. Stage outputs are cached: a stage is skipped when its stage-marker hash
matches the current config (see utils.check_stage_cache); pass --force to re-run.
"""
import os

# ── Spec §2 seeds (frozen — one seed per stochastic step) ────────────────────
SEED_SAMPLING = 7      # Stage A: eval slice first, then training pool (same RNG stream)
SEED_TEMPLATE = 1234   # Stage E: template assignment
SEED_PAIR = 1701       # Stage G: chosen/rejected selection within class
SEED_SHAM = 69         # Stage I: sham chosen/rejected swap
SEED_BALANCE = 2003    # Stage H: direction downsampling + final shuffle
GEN_SEED = int(os.environ.get("GEN_SEED", "0"))  # vLLM SamplingParams seed (manifest-recorded)

# ── Pool / slice sizes ───────────────────────────────────────────────────────
X = int(os.environ.get("X", "5"))               # training-pool size (5 = smoke default)
X_FULL = int(os.environ.get("X_FULL", "20000")) # planned full-run X (pilot gate 2)
EVAL_SLICE_SIZE = 1500                          # spec §3: drawn FIRST, before the pool
PILOT = os.environ.get("PILOT", "").strip().lower() in ("1", "true", "yes")
MIN_PAIRS_FULL_RUN = 3000                       # spec §8 post-balance floor (full run only)

# ── Generation (spec §2 — one fixed sampling config for ALL generation stages) ─
N = 10                                          # samples per prompt, Filter-0
N_FILTER1 = int(os.environ.get("N_FILTER1", "10"))  # Filter-1 (16 = owner escalation path)
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 20
MAX_NEW_TOKENS = 16384
GEN_MODEL = "Qwen/Qwen3-4B"                     # thinking mode ON at render time

# ── vLLM engine tuning (mirrors the harmlessness module / MVP-RO1-v2) ────────
# Trivia questions + templates are short (<300 tokens even with the format instruction),
# so a 2048-token prompt budget is generous. Capping max_model_len below Qwen3's native
# 40960 multiplies KV-cache concurrency. Throughput knobs only — outputs are unchanged
# (vLLM seeds sampling per request), so runs at different values stay comparable.
PROMPT_TOKEN_BUDGET = 2048
MAX_MODEL_LEN = MAX_NEW_TOKENS + PROMPT_TOKEN_BUDGET   # 18432
GPU_MEMORY_UTILIZATION = 0.95
MAX_NUM_SEQS = 384
GEN_BATCH_PROMPTS = 500     # prompts per llm.generate() call (checkpoint granularity)

# ── Filter thresholds (spec §2) ──────────────────────────────────────────────
FILTER0_MIN_PARSEABLE = 9        # ≥ 9/10 samples parseable
FILTER0_MIN_CONSISTENCY = 0.70   # ≥ 70% of parseable samples CORRECT
CONSISTENCY_STRATA = ((0.70, "70-80"), (0.80, "80-90"), (0.90, "90-99"), (1.00, "100"))

# ── gpt-5-mini grader / wrong-answer generator (spec §6 / §5) ────────────────
# Follows the monitorability monitor pattern (RO1 monitorability-eval/scripts/03_run_monitor.py):
# ThreadPoolExecutor + semaphore gate that steps concurrency down on 429s, resume via a
# progress file, plus persistent per-call caches. NOTE: the spec says temperature=0, but
# gpt-5-mini rejects non-default temperature — it is OMITTED from every call (recorded in
# manifest.json), matching how the existing monitor calls gpt-5-mini.
GRADER_MODEL = "gpt-5-mini"
GRADER_MAX_RETRIES = 3           # spec §6 (the monitor uses 5; spec pins 3 here)
GRADER_BACKOFF_FACTOR = 2.0
GRADER_TIMEOUT = 60
NUM_WORKERS = 64                 # starting concurrency (auto-reduces on 429s)
MIN_WORKERS = 16                 # concurrency floor

# ── Templates (spec §5 Stage E) ──────────────────────────────────────────────
# DROPPED_TEMPLATES: ids removed by pilot gate 3 (rejected-class rate < 2%). Empty for
# the pilot itself; set for the full run only if the pilot says so.
DROPPED_TEMPLATES = [s.strip() for s in os.environ.get("DROPPED_TEMPLATES", "").split(",") if s.strip()]

# ── Datasets ─────────────────────────────────────────────────────────────────
TRIVIAQA_DATASET = "mandarjoshi/trivia_qa"
TRIVIAQA_CONFIG = "rc.nocontext"
TRIVIAQA_SPLIT = "train"
# SycophancyEval "Are you sure?" release (Sharma et al., 2023) — used verbatim as Eval set 2.
OOD_EVAL_URL = "https://raw.githubusercontent.com/meg-tong/sycophancy-eval/main/datasets/are_you_sure.jsonl"

# ── Export (DatasetDict handoff to train_dpo.py — same contract as harmlessness) ─
VALID_FRACTION = float(os.environ.get("VALID_FRACTION", "0.1"))
# Train/valid split seed. Not in the spec's seed table (the split is a training handoff
# detail, out of the spec's scope); fixed here and recorded in manifest.json. The sham
# export reuses the exact same split indices as the main export.
SPLIT_SEED = 2003

# ── Paths ────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.environ.get("RUN_OUTPUT_DIR") or os.path.join(HERE, "outputs")

RAW_DIR = os.path.join(OUTPUT_DIR, "raw")
RESERVED_DIR = os.path.join(OUTPUT_DIR, "reserved")
FILTER0_DIR = os.path.join(OUTPUT_DIR, "filter0")
FILTER1_DIR = os.path.join(OUTPUT_DIR, "filter1")
PAIRS_DIR = os.path.join(OUTPUT_DIR, "pairs")
AUDIT_DIR = os.path.join(OUTPUT_DIR, "audit")
MARKERS_DIR = os.path.join(OUTPUT_DIR, "stage_markers")

TRAINING_POOL_PATH = os.path.join(RAW_DIR, "training_pool.jsonl")
EVAL_ID_PARTIAL_PATH = os.path.join(RESERVED_DIR, "eval_id_triviaqa.partial.jsonl")  # no `w` yet
EVAL_ID_PATH = os.path.join(RESERVED_DIR, "eval_id_triviaqa.jsonl")                  # frozen (stage 05)
EVAL_OOD_PATH = os.path.join(RESERVED_DIR, "eval_ood_sycophancyeval.jsonl")
EVAL_OOD_PROVENANCE_PATH = os.path.join(RESERVED_DIR, "eval_ood_provenance.json")

FILTER0_GENERATIONS_PATH = os.path.join(FILTER0_DIR, "generations.jsonl")
FILTER0_GEN_CKPT_PATH = os.path.join(FILTER0_DIR, "generations.ckpt.jsonl")
FILTER0_GRADES_PATH = os.path.join(FILTER0_DIR, "grades.jsonl")
FILTER0_GRADE_PROGRESS_PATH = os.path.join(FILTER0_DIR, "grade_progress.json")
FILTER0_SUMMARY_PATH = os.path.join(FILTER0_DIR, "summary.jsonl")     # per-question verdicts
WRONG_ANSWERS_PATH = os.path.join(FILTER0_DIR, "wrong_answers.jsonl") # Stage D output (pool)

INSTANCES_PATH = os.path.join(FILTER1_DIR, "instances.jsonl")         # Stage E output
FILTER1_GENERATIONS_PATH = os.path.join(FILTER1_DIR, "generations.jsonl")
FILTER1_GEN_CKPT_PATH = os.path.join(FILTER1_DIR, "generations.ckpt.jsonl")
FILTER1_GRADES_PATH = os.path.join(FILTER1_DIR, "grades.jsonl")
FILTER1_GRADE_PROGRESS_PATH = os.path.join(FILTER1_DIR, "grade_progress.json")
FILTER1_SUMMARY_PATH = os.path.join(FILTER1_DIR, "summary.jsonl")     # per-instance classes

# Persistent API-call caches (spec §2/§6 caching rules; shared across re-runs of a run dir)
GRADER_CACHE_PATH = os.path.join(OUTPUT_DIR, "grader_cache.jsonl")          # (qid, prompt_type, sha1)
WRONG_ANSWER_CACHE_PATH = os.path.join(OUTPUT_DIR, "wrong_answer_cache.jsonl")  # question_id

PAIRS_PATH = os.path.join(PAIRS_DIR, "anti_syc_dpo.jsonl")
SHAM_PAIRS_PATH = os.path.join(PAIRS_DIR, "anti_syc_dpo_sham.jsonl")

MANIFEST_PATH = os.path.join(OUTPUT_DIR, "manifest.json")
QA_REPORT_PATH = os.path.join(OUTPUT_DIR, "qa_report.md")
PILOT_GATES_PATH = os.path.join(OUTPUT_DIR, "pilot_gates.json")
DIAGNOSTICS_PATH = os.path.join(OUTPUT_DIR, "diagnostics.json")
SYC_FRACTION_FIG_PATH = os.path.join(OUTPUT_DIR, "syc_fraction_hist.png")

AUDIT_SAMPLE_PATH = os.path.join(AUDIT_DIR, "grader_audit_sample.csv")
AUDIT_RESULTS_PATH = os.path.join(AUDIT_DIR, "grader_audit_results.json")

DPO_DATASET_DIR = os.path.join(OUTPUT_DIR, "dpo_dataset")
DPO_DATASET_SHAM_DIR = os.path.join(OUTPUT_DIR, "dpo_dataset_sham")


def ensure_output_dirs():
    for d in (OUTPUT_DIR, RAW_DIR, RESERVED_DIR, FILTER0_DIR, FILTER1_DIR,
              PAIRS_DIR, AUDIT_DIR, MARKERS_DIR):
        os.makedirs(d, exist_ok=True)


def sampling_config():
    """The frozen §2 sampling config, as a dict (manifest + stage-hash input)."""
    return {
        "model": GEN_MODEL,
        "enable_thinking": True,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "max_new_tokens": MAX_NEW_TOKENS,
        "gen_seed": GEN_SEED,
    }
