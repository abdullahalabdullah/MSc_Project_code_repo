import datasets
from transformers import AutoTokenizer
import os

DATASET_PATH = "outputs/aqua/qwen3-1.7B/dpo/seed_0/data/cot_negative_sk-llama_original"
MODEL = "Qwen/Qwen3-1.7B"
MAX_LENGTH = 10240

assert os.path.exists(DATASET_PATH), f"Dataset not found at {DATASET_PATH}"

print(f"Loading tokenizer from {MODEL} ...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL, cache_dir=os.environ.get("HF_MODELS_CACHE_DIR")
)

print(f"Loading dataset from {DATASET_PATH} ...")
dataset = datasets.load_from_disk(DATASET_PATH)

print(f"Train size: {len(dataset['train'])}  Valid size: {len(dataset['valid'])}\n")

lengths = []
for split in ("train", "valid"):
    for ex in dataset[split]:
        for completion in ("chosen", "rejected"):
            seq = ex["prompt"] + ex[completion]
            lengths.append(len(tokenizer(seq, add_special_tokens=False)["input_ids"]))

lengths.sort()
n = len(lengths)
truncated = sum(l > MAX_LENGTH for l in lengths)

print(f"Total sequences checked : {n}")
print(f"Max length              : {lengths[-1]}")
print(f"p99                     : {lengths[int(n * 0.99)]}")
print(f"p95                     : {lengths[int(n * 0.95)]}")
print(f"p50 (median)            : {lengths[n // 2]}")
print(f"Truncated (> {MAX_LENGTH})  : {truncated} / {n}  ({100*truncated/n:.1f}%)")
