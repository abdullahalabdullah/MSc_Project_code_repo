"""One-off utility — back up the heavy, gitignored stage joblibs to a private HF Hub dataset
repo before terminating a pod. These files are excluded from git (see MVP-RO2-v2/.gitignore:
`*.joblib`) because they're large and regenerable by rerunning the pipeline, but regenerating
them means redoing stage 02 (the expensive vLLM generation) from scratch. Uploading them here
lets you resume experimentation (e.g. a different MIN_MARGIN or EXPORT_FIELD) from disk instead.

Not part of the pipeline (run_all.sh / run_from_03.sh never call this) — run it by hand:

    python backup_raw_outputs_to_hf.py                       # backs up outputs/main
    python backup_raw_outputs_to_hf.py --run-dir outputs/foo  # a different run folder
    python backup_raw_outputs_to_hf.py --repo you/custom-name

Uploads to a PRIVATE dataset repo (created if missing) named
"<your-hf-username>/ro2-dpo-curation-raw-<run-dir-name>" by default. Only the heavy joblibs
that git ignores are uploaded (02_responses / 03_labeled / 03_all_labeled / 04_scored) — the
small tracked ones (01_prompts_meta, final_pairs) are already in git and skipped.
"""
import argparse
import os

import env_setup

RAW_FILES = [
    "02_responses.joblib",
    "03_labeled.joblib",
    "03_all_labeled.joblib",
    "04_scored.joblib",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="outputs/main",
                     help="Run folder to back up (default: outputs/main)")
    ap.add_argument("--repo", default=None,
                     help="Target HF dataset repo id (default: <username>/ro2-dpo-curation-raw-<run-dir-name>)")
    args = ap.parse_args()

    env_setup.setup_env()
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("ERROR: HF_TOKEN not set (check .env) — needed to push to your HF account.")

    api = HfApi(token=token)
    who = api.whoami()
    run_name = os.path.basename(os.path.normpath(args.run_dir))
    repo_id = args.repo or f"{who['name']}/ro2-dpo-curation-raw-{run_name}"

    print(f"Target repo: {repo_id} (private)")
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)

    found = []
    for fname in RAW_FILES:
        path = os.path.join(args.run_dir, fname)
        if os.path.exists(path):
            found.append((path, fname))
        else:
            print(f"  skip (not found): {path}")
    if not found:
        raise SystemExit(f"ERROR: none of {RAW_FILES} found under {args.run_dir}")

    for path, fname in found:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"Uploading {path} ({size_mb:.0f} MB) -> {repo_id}/{fname} ...")
        api.upload_file(
            path_or_fileobj=path, path_in_repo=fname,
            repo_id=repo_id, repo_type="dataset",
        )
        print(f"  done: {fname}")

    print(f"\nAll files backed up -> https://huggingface.co/datasets/{repo_id}")
    print("Restore later with:")
    print(f"  huggingface_hub.hf_hub_download(repo_id='{repo_id}', repo_type='dataset', filename='02_responses.joblib', local_dir='{args.run_dir}')")


if __name__ == "__main__":
    main()
