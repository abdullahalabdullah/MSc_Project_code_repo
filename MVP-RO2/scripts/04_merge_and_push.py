"""Merge an SFT LoRA adapter into Qwen3-4B and push the merged model for MVP-RO1.

The merged repo is a full standalone 16-bit model (~8 GB) that MVP-RO1's vLLM
pipeline can load by hf_id. Run once per epoch you want to evaluate (epoch 3 is
the primary target).

Examples:
    # merge the epoch-3 adapter (Hub repo) and push the merged model
    python scripts/04_merge_and_push.py --config configs/sft_harmlessness.yaml --epoch 3

    # merge from a LOCAL checkpoint dir instead of the Hub adapter (faster on the pod)
    python scripts/04_merge_and_push.py --config configs/sft_harmlessness.yaml --epoch 3 \
        --adapter ./outputs/qwen3-4b-sft-harmlessness/checkpoint-915

    # local merge only, no push
    python scripts/04_merge_and_push.py --config configs/sft_harmlessness.yaml --epoch 3 --no-push
"""

import argparse

import _bootstrap  # noqa: F401
from sft.config import epoch_repo_id, load_config
from sft.credentials import check_credentials
from sft.logging_utils import make_logger
from sft.merge import merge_adapter, push_merged

_DEFAULT_MERGED_PREFIX = "qwen3-4b-sft-harmlessness-pku-merged"


def main():
    parser = argparse.ArgumentParser(description="Merge an SFT adapter into the base model + push.")
    parser.add_argument("--config", default="configs/sft_harmlessness.yaml")
    parser.add_argument("--epoch", type=int, required=True, help="Epoch index (1-3).")
    parser.add_argument("--adapter", default=None,
                        help="Override: adapter Hub repo id OR local checkpoint dir. "
                             "Default: the pushed -epoch<N> adapter repo.")
    parser.add_argument("--merged-repo", default=None,
                        help="Override the target merged repo id.")
    parser.add_argument("--out-dir", default=None, help="Override the local merge dir.")
    parser.add_argument("--no-push", action="store_true", help="Merge locally, do not push.")
    parser.add_argument("--public", action="store_true", help="Push as a public repo.")
    args = parser.parse_args()

    logger = make_logger("merge_and_push")
    cfg = load_config(args.config)
    check_credentials(logger)

    adapter = args.adapter or epoch_repo_id(cfg, args.epoch)
    prefix = cfg.get("merged_repo_prefix", _DEFAULT_MERGED_PREFIX)
    merged_repo = args.merged_repo or f"{cfg['hf_username']}/{prefix}-epoch{args.epoch}"
    out_dir = args.out_dir or f"{cfg['output_dir']}-merged-epoch{args.epoch}"

    logger.info("Merging epoch %d: adapter=%s -> merged=%s", args.epoch, adapter, merged_repo)
    merge_adapter(cfg["base_model"], adapter, out_dir, logger)

    if args.no_push:
        logger.info("--no-push set; merged model is at %s. Done.", out_dir)
        return
    push_merged(out_dir, merged_repo, logger, private=not args.public)
    logger.info("Done. Add this to MVP-RO1 config/mvp_config.yaml under models::\n"
                "  - name: \"qwen3_4b_sft_harmless_e%d\"\n"
                "    hf_id: \"%s\"\n"
                "    role: \"+sft_harmlessness\"", args.epoch, merged_repo)


if __name__ == "__main__":
    main()
