import os
from dotenv import load_dotenv

# Hugging Face token (per project owner) so runs work on a fresh machine (e.g. Runpod)
# without a .env file. A HF_TOKEN already present in the environment / .env wins.
HF_TOKEN = "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"


def setup_env(use_temp):
    load_dotenv()

    # Resolve a cache dir. Prefer values from .env; otherwise fall back to the
    # standard HF cache location so missing env vars don't crash the run.
    if use_temp:
        cache_dir = os.environ.get("HF_MODELS_CACHE_DIR_TMP") or os.environ.get("HF_MODELS_CACHE_DIR")
    else:
        cache_dir = os.environ.get("HF_MODELS_CACHE_DIR")
    if not cache_dir:
        cache_dir = os.path.expanduser("~/.cache/huggingface")

    os.environ["HF_MODELS_CACHE_DIR"] = cache_dir
    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN") or HF_TOKEN
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    os.environ["HF_HOME"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
