"""Move a run dir between machines via a private HF dataset repo.

The pipeline's GPU phases (03, 07) and API phases (04/05/06, 08-12) run on different
boxes; this utility carries the run dir across (analogous to the harmlessness module's
backup_raw_outputs_to_hf.py, but bidirectional and whole-dir, since every JSONL here is
an input to a later stage).

    python backup_run_to_hf.py --push outputs/pilot            # GPU box, before terminating
    python backup_run_to_hf.py --pull outputs/pilot            # local/API box, before phase 2
    python backup_run_to_hf.py --push outputs/pilot --repo you/custom-name

Default repo: <hf-username>/ro2-antisyc-curation-<run-dir-name> (private, created on push).
"""
import argparse
import os
import sys

import env_setup


def main():
    ap = argparse.ArgumentParser(description="Push/pull a run dir to/from a private HF dataset repo.")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--push", metavar="RUN_DIR", help="Upload the run dir.")
    group.add_argument("--pull", metavar="RUN_DIR", help="Download into the run dir.")
    ap.add_argument("--repo", default=None,
                    help="HF dataset repo id (default: <username>/ro2-antisyc-curation-<run-name>)")
    args = ap.parse_args()

    env_setup.setup_env()
    from huggingface_hub import HfApi, snapshot_download

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("ERROR: HF_TOKEN not set (check .env) — needed for the private repo.")
    api = HfApi(token=token)

    run_dir = args.push or args.pull
    run_name = os.path.basename(os.path.normpath(run_dir))
    repo_id = args.repo or f"{api.whoami()['name']}/ro2-antisyc-curation-{run_name}"

    if args.push:
        if not os.path.isdir(run_dir):
            sys.exit(f"ERROR: run dir not found: {run_dir}")
        print(f"Uploading {run_dir} -> {repo_id} (private)")
        api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(folder_path=run_dir, repo_id=repo_id, repo_type="dataset",
                          commit_message=f"backup {run_name}")
        print(f"Done -> https://huggingface.co/datasets/{repo_id}")
    else:
        print(f"Downloading {repo_id} -> {run_dir}")
        os.makedirs(run_dir, exist_ok=True)
        snapshot_download(repo_id=repo_id, repo_type="dataset", token=token,
                          local_dir=run_dir)
        print(f"Done -> {run_dir}")


if __name__ == "__main__":

    main()
