"""LoRA-adapter merging for the harmlessness inference pipeline.

Self-contained: duplicated verbatim from monitorability-eval/src/merge.py so the
harmlessness eval never imports the monitorability-eval code. RO2 checkpoints are
pushed to the Hub as *adapter-only* repos (adapter_config.json +
adapter_model.safetensors, no config.json), which vLLM cannot load directly. This
module merges such an adapter into its base model and saves a full,
vLLM-loadable model to a local directory.

Used by scripts/02_run_inference.py --merge (merge locally, then run inference
on the merged dir; the merge base comes from the config `merge:` block).
"""

from pathlib import Path


def cleanup_hf_cache(repo_id: str) -> None:
    """Best-effort delete of `repo_id`'s snapshot from the local HF hub cache.

    Called after a base model has been merged into an adapter and saved to
    `out_dir` — the cache copy is now redundant (its weights live on disk in
    the merged output and in the vLLM process's GPU memory), so freeing it
    keeps small container disks (e.g. RunPod's default 20-30GB) from filling
    up before WildGuard's own ~14.5GB download. Never raises: a failed cleanup
    just leaves disk usage higher, it doesn't affect correctness.
    """
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id == repo_id:
                revisions = {rev.commit_hash for rev in repo.revisions}
                strategy = cache_info.delete_revisions(*revisions)
                print(f"Freeing {strategy.expected_freed_size_str} from HF cache "
                      f"for {repo_id} (merged; no longer needed on disk) ...")
                strategy.execute()
    except Exception as e:
        print(f"WARNING: could not clean HF cache for {repo_id}: {e}")


def merge_adapter(base: str, adapter: str, out_dir, force: bool = False,
                  dtype: str = "bfloat16") -> Path:
    """Merge a LoRA adapter into `base` and save the full model to `out_dir`.

    base:    HF id (or local dir) of the base model the adapter is merged into.
    adapter: HF repo id (or local dir) of the adapter-only checkpoint.
    out_dir: local directory to write the merged model + tokenizer to.
    force:   re-merge even if `out_dir` already holds a merged model. Default
             False: an existing merge (detected via config.json) is reused if
             present. In normal operation `out_dir` won't be there on the next
             call — the caller (02_run_inference.py --merge) deletes it right
             after vLLM loads it, and this function frees the base model's own
             HF cache too (see cleanup_hf_cache) — trading re-download/re-merge
             time on the next run for not needing a large container disk. This
             fast path only fires if that cleanup didn't happen (e.g. a crash
             between merge and vLLM load).

    The tokenizer is taken from the adapter repo when it ships one (it carries
    any training-time chat template / added tokens), falling back to the base's.

    Returns `out_dir` as a Path — pass it wherever an HF model id is expected.
    """
    out_dir = Path(out_dir)
    if not force and (out_dir / "config.json").exists():
        print(f"Merged model already present at {out_dir} — reusing (use force=True to re-merge).")
        return out_dir

    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Sanity check: warn (don't fail) when the adapter was trained on a
    # different base than the one it is being merged into.
    adapter_base = PeftConfig.from_pretrained(adapter).base_model_name_or_path
    if adapter_base and adapter_base != base:
        print(f"WARNING: adapter {adapter} was trained on base '{adapter_base}' "
              f"but is being merged into '{base}'.")

    print(f"Loading base model {base} ({dtype}) ...")
    base_model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=getattr(torch, dtype))

    print(f"Attaching adapter {adapter} ...")
    model = PeftModel.from_pretrained(base_model, adapter)

    print("Merging adapter into base weights (merge_and_unload) ...")
    merged = model.merge_and_unload()

    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_dir, safe_serialization=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base)
    tokenizer.save_pretrained(out_dir)
    print(f"Merged model + tokenizer saved to {out_dir}")

    del base_model, model, merged
    cleanup_hf_cache(base)

    return out_dir
