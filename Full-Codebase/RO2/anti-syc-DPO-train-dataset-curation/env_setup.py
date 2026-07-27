"""HuggingFace cache / token setup (verbatim copy of the harmlessness module's).

Reads everything from the environment or a local .env — NO hardcoded secrets.
Set HF_TOKEN (and optionally HF_MODELS_CACHE_DIR) in your shell or a .env file.
The API stages additionally need OPENAI_API_KEY (checked in grader.get_client);
load_dotenv() picks it up from the same .env.

    HF_TOKEN=hf_xxx
    HF_MODELS_CACHE_DIR=/workspace/hf_cache
    OPENAI_API_KEY=sk-xxx
"""
import os

try:
    from dotenv import load_dotenv
except ImportError:  # dotenv is optional
    def load_dotenv(*a, **k):
        return False


def setup_env():
    load_dotenv()

    cache_dir = os.environ.get("HF_MODELS_CACHE_DIR") or os.path.expanduser(
        "~/.cache/huggingface"
    )
    os.environ["HF_MODELS_CACHE_DIR"] = cache_dir
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    token = os.environ.get("HF_TOKEN")
    if not token:
        print(
            "WARNING: HF_TOKEN is not set. Public models will still download, but "
            "gated ones will fail. Set it in your shell or a .env file."
        )
    return cache_dir
