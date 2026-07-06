"""LoRA-adapter merging for the inference pipeline.

RO2 checkpoints are pushed to the Hub as *adapter-only* repos (adapter_config.json
+ adapter_model.safetensors, no config.json), which vLLM cannot load directly.
This module merges such an adapter into its base model and saves a full,
vLLM-loadable model to a local directory.

Used by:
  - scripts/02_run_inference_qwen.py --merge   (merge locally, then run inference
    on the merged dir; the merge base comes from the config `merge:` block)
  - scripts/merge_and_push.py                   (merge locally, then push to Hub)

Heavy imports (torch/peft/transformers) happen inside the function so importing
this module is cheap and safe, matching the rest of src/.
"""

from pathlib import Path


def merge_adapter(base: str, adapter: str, out_dir, force: bool = False,
                  dtype: str = "bfloat16") -> Path:
    """Merge a LoRA adapter into `base` and save the full model to `out_dir`.

    base:    HF id (or local dir) of the base model the adapter is merged into.
    adapter: HF repo id (or local dir) of the adapter-only checkpoint.
    out_dir: local directory to write the merged model + tokenizer to.
    force:   re-merge even if `out_dir` already holds a merged model. Default
             False: an existing merge (detected via config.json) is reused, so
             repeated --merge runs of the same adapter don't redo the work.

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
    return out_dir
