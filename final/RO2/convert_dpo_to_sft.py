"""Convert an exported DPO preference dataset into the SFT-arm dataset.

The SFT arm is a controlled ablation of DPO (see SFT_Arm_Developer_Spec.md): train on the
chosen responses only, i.e. DPO with the negative (rejected) term removed. The conversion is
therefore deliberately trivial — anything cleverer would break the ablation:

    DatasetDict({"train","valid"}) prompt / chosen / rejected
        -> DatasetDict({"train","valid"}) prompt / chosen

No re-filtering, re-judging, deduplication, truncation, shuffling, or re-splitting: row order
and the existing seeded train/valid boundary are preserved exactly, so the SFT arm trains on
the identical chosen set (and evals on the identical valid split) as the DPO arm.

Works on any dataset following the curation modules' export contract (harmlessness stage 06,
anti-syc stage 12): three string columns prompt/chosen/rejected, chosen = full response
including its <think>…</think> CoT.

Provenance (spec §2.1): writes <output>/provenance.json recording sha256 hashes of the source
and derived datasets so the lineage is auditable at train time (train_sft.py re-verifies).

Usage:
    python convert_dpo_to_sft.py path/to/dpo_dataset [--output path/to/sft_dataset] [--limit N]

Default output is a sibling directory named "sft_dataset" next to the source. --limit builds a
small smoke-test dataset (marked non-canonical in provenance.json — never train the real run
on one).
"""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys

import datasets


# ── Self-contained helpers (this file has no dependency on any sibling module) ──
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_dir(root, exclude=("provenance.json", ".verify_passed")):
    """Deterministic hash of a directory tree: sha256 over sorted '<relpath>  <file-sha>' lines.

    provenance.json is excluded so the derived dataset's hash is stable after the provenance
    file itself is written into it; .verify_passed (train_sft.sh's verification marker) is
    excluded for the same reason.
    """
    lines = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in sorted(filenames):
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, root)
            if os.path.basename(rel) in exclude:
                continue
            lines.append(f"{rel}  {sha256_file(path)}")
    digest = hashlib.sha256("\n".join(sorted(lines)).encode("utf-8")).hexdigest()
    return digest


def git_commit_or_none():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dpo_dataset", help="path to the exported dpo_dataset (DatasetDict dir)")
    ap.add_argument("--output", default=None,
                    help="output dir (default: sibling 'sft_dataset' next to the source)")
    ap.add_argument("--limit", type=int, default=None,
                    help="keep only the first N rows per split (SMOKE TESTS ONLY — "
                         "marked non-canonical in provenance.json)")
    args = ap.parse_args()

    src = os.path.normpath(args.dpo_dataset)
    out = os.path.normpath(args.output or os.path.join(os.path.dirname(src), "sft_dataset"))
    if os.path.abspath(out) == os.path.abspath(src):
        sys.exit("ERROR: output dir must differ from the source dir")
    if os.path.exists(out) and os.listdir(out):
        sys.exit(f"ERROR: output dir already exists and is not empty: {out}\n"
                 "       Remove it first — silent overwrites would corrupt provenance.")

    print(f"Loading DPO dataset: {src}")
    dset = datasets.load_from_disk(src)

    # The export contract this converter (and train_sft.py) relies on.
    assert isinstance(dset, datasets.DatasetDict), f"expected a DatasetDict, got {type(dset)}"
    assert set(dset.keys()) == {"train", "valid"}, f"expected splits {{train, valid}}, got {sorted(dset.keys())}"
    for split, d in dset.items():
        missing = {"prompt", "chosen", "rejected"} - set(d.column_names)
        assert not missing, f"split '{split}' is missing columns {missing} — is this a dpo_dataset export?"

    source_rows = {split: len(d) for split, d in dset.items()}
    source_sha = sha256_dir(src)

    canonical = args.limit is None
    if args.limit is not None:
        print(f"[smoke] --limit {args.limit}: truncating each split (NON-canonical dataset)")
        dset = datasets.DatasetDict(
            {split: d.select(range(min(args.limit, len(d)))) for split, d in dset.items()})

    # The entire conversion: drop `rejected`, keep prompt + chosen. Row order untouched.
    sft = datasets.DatasetDict(
        {split: d.select_columns(["prompt", "chosen"]) for split, d in dset.items()})

    for split in sft:
        expected = source_rows[split] if canonical else min(args.limit, source_rows[split])
        assert len(sft[split]) == expected, \
            f"row-count mismatch in '{split}': {len(sft[split])} != {expected}"
        assert sft[split].column_names == ["prompt", "chosen"]

    print(f"Saving SFT dataset -> {out}")
    sft.save_to_disk(out)

    provenance = {
        "source_path": src,
        "source_sha256": source_sha,
        "derived_sha256": sha256_dir(out),
        "rows": {split: len(d) for split, d in sft.items()},
        "source_rows": source_rows,
        "columns": ["prompt", "chosen"],
        "canonical": canonical,
        "limit": args.limit,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": git_commit_or_none(),
        "converter": os.path.basename(__file__),
    }
    prov_path = os.path.join(out, "provenance.json")
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)

    print(f"Provenance -> {prov_path}")
    print(json.dumps({k: provenance[k] for k in ("rows", "source_sha256", "derived_sha256", "canonical")},
                     indent=2))
    if not canonical:
        print("WARNING: this is a --limit smoke dataset. Do NOT use it for the real training run.")


if __name__ == "__main__":
    main()
