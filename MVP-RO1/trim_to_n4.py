#!/usr/bin/env python3
"""
One-time script: trim all inference and monitor outputs from N=8 to N=4.
Keeps only sample_idx 0, 1, 2, 3. Rewrites files in-place (with backups).
"""

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).parent
KEEP = {0, 1, 2, 3}

MODELS = [
    "tulu_sft_no_safety",
    "tulu_sft_full",
    "tulu_dpo",
    "tulu_final",
]


def trim_jsonl(path: Path, keep_idx: set, key: str = "sample_idx") -> tuple[int, int]:
    """Rewrite path keeping only records where record[key] in keep_idx."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    before = len(records)
    kept = [r for r in records if r[key] in keep_idx]

    backup = path.with_suffix(".jsonl.bak")
    shutil.copy2(path, backup)

    with open(path, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return before, len(kept)


def trim_monitor_progress(path: Path, keep_idx: set):
    """Remove done_keys entries where sample_idx not in keep_idx."""
    if not path.exists():
        return
    with open(path) as f:
        data = json.load(f)

    before = len(data.get("done_keys", []))
    # done_keys is list of [instance_id, sample_idx]
    filtered = [k for k in data.get("done_keys", []) if k[1] in keep_idx]

    backup = path.with_suffix(".json.bak")
    shutil.copy2(path, backup)

    with open(path, "w") as f:
        json.dump({"done_keys": filtered}, f)

    print(f"  monitor_progress: {before} → {len(filtered)} done_keys")


def main():
    for model in MODELS:
        print(f"\n=== {model} ===")

        # Inference outputs
        out_dir = ROOT / "outputs" / model
        for arm in ("hinted", "unhinted"):
            p = out_dir / f"{arm}.jsonl"
            if not p.exists():
                print(f"  SKIP {arm}.jsonl (not found)")
                continue
            before, after = trim_jsonl(p, KEEP)
            print(f"  {arm}.jsonl: {before} → {after} records (backup: {arm}.jsonl.bak)")

        # Monitor outputs (only tulu_sft_no_safety has results so far)
        mon_dir = ROOT / "monitor_outputs" / model
        results_path = mon_dir / "monitor_results.jsonl"
        progress_path = mon_dir / "monitor_progress.json"

        if results_path.exists():
            before, after = trim_jsonl(results_path, KEEP)
            print(f"  monitor_results.jsonl: {before} → {after} records (backup: monitor_results.jsonl.bak)")

        if progress_path.exists():
            trim_monitor_progress(progress_path, KEEP)

    print("\nDone. Original files backed up as *.bak")
    print("Next: update config/mvp_config.yaml n_samples: 8 -> 4")


if __name__ == "__main__":
    main()
