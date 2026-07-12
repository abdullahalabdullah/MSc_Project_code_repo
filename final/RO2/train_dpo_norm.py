# ---------------------------------------------------------------------------
# Length-normalized DPO — "dpo_norm".
#
# Standard DPO scores a completion by the SUM of its per-token log-probs, which
# mechanically rewards longer generations (the mechanism behind the observed post-DPO
# response-length inflation — see RO1/notes/harmless_dpo_response_length_inflation.md).
# Length-normalized DPO instead uses the length-AVERAGED per-token log-prob
# (sum / #completion-tokens), removing that length bias while keeping DPO's
# reference-based sigmoid loss (unlike SimPO, which is reference-free and adds a margin).
#
# Length normalization for DPO was noted in the original DPO work (Rafailov et al.) and
# established as a strong, well-tuned variant by Meng et al. (2024, "SimPO", which reports
# length-normalized DPO as a baseline).
#
# The length-normalization here follows AllenAI open-instruct's `dpo_norm`:
#   https://github.com/allenai/open-instruct/blob/main/open_instruct/dpo_utils.py
# whose `_get_batch_logps(average_log_prob=True)` computes
#     (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1).clamp(min=1)
# (i.e. mean per-completion-token log-prob) and then applies the standard reference-based
# `dpo_loss` (reference_free=False). TRL's DPOTrainer already computes this exact quantity
# for loss_type="ipo" (`all_logps / loss_mask.sum(-1)`) but not for the sigmoid loss; we
# apply that same normalization to the sigmoid DPO loss via a thin DPOTrainer subclass.
#
# Added ALONGSIDE standard DPO — train_dpo.py is NOT modified; train.sh selects via
# METHOD=dpo|dpo_norm. All other machinery (LoRA, eval/checkpoint schedule, per-step Hub
# push, best-checkpoint selection, metric logging, dataset formatting) is shared with
# train_dpo.py and imported from it, so the two methods stay directly comparable.
# ---------------------------------------------------------------------------
"""LoRA length-normalized DPO training on the harmlessness preference pairs.

Consumes the same DatasetDict exported by the curation module (stage 06):
    DatasetDict({"train","valid"}) with columns prompt / chosen / rejected.

Run via train.sh (one command):
    METHOD=dpo_norm bash train.sh <.../dpo_dataset>
or directly:
    python train_dpo_norm.py --dataset_path <.../dpo_dataset> --model_name_or_path Qwen/Qwen3-4B \
        --beta 5 ...   # loss_type stays "sigmoid" (DPO); logps are length-averaged
"""
import json
import os

import datasets
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer, ModelConfig, TrlParser, get_peft_config

# Reuse ALL shared machinery from train_dpo.py unchanged (that file is not modified).
# Importing it also runs its module-level setup_hf_env(), so HF caches/token are configured
# on import — we call it again in main() too (idempotent) for clarity.
from train_dpo import (
    set_seed,
    setup_hf_env,
    gpu_memory_stats,
    BestCheckpointCallback,
    StepwiseEvalCallback,
    EvalMetricsLoggerCallback,
    PushStepCheckpointCallback,
    ScriptArguments,
)


class LengthNormalizedDPOTrainer(DPOTrainer):
    """DPOTrainer with length-normalized (mean) completion log-probs — open-instruct `dpo_norm`.

    Standard DPOTrainer sums per-token log-probs for the sigmoid loss; here we divide that sum
    by the number of completion tokens, so neither chosen nor rejected is favoured for being
    longer. Equivalent to open-instruct's `_get_batch_logps(average_log_prob=True)`:
        (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1).clamp(min=1)

    Implementation: the parent `concatenated_forward` returns SUMMED chosen/rejected logps; we
    recompute the completion lengths exactly the way the parent does (via `concatenated_inputs`,
    reading `completion_attention_mask`) and divide. Applied to BOTH the policy and the reference
    forward pass (no `is_ref_model` gate) so each logratio is consistently length-normalized on
    both sides — matching open-instruct, which averages policy and reference log-probs alike.

    Assumption: completions are not truncated by `max_length` (train.sh sets max_length=12288,
    far above the ~1.5k-token completions), so `completion_attention_mask` here equals the
    parent's internal per-token loss mask. If you lower `max_length` below the completion length,
    revisit this (the normalization denominator would then differ from the tokens actually scored).
    """

    def concatenated_forward(self, model, batch, is_ref_model=False):
        output = super().concatenated_forward(model, batch, is_ref_model=is_ref_model)
        num_examples = batch["prompt_input_ids"].shape[0]
        # Same call the parent makes; cheap tensor concat/pad, no extra model forward.
        concatenated_batch = self.concatenated_inputs(batch, padding_value=self.padding_value)
        # #completion tokens per example, ordered [chosen; rejected]. clamp(min=1) mirrors
        # open-instruct and guards against a hypothetical empty completion.
        completion_lengths = concatenated_batch["completion_attention_mask"].sum(dim=-1).clamp(min=1)
        lengths = completion_lengths.to(output["chosen_logps"].dtype)
        output["chosen_logps"] = output["chosen_logps"] / lengths[:num_examples]
        output["rejected_logps"] = output["rejected_logps"] / lengths[num_examples:]
        return output


def main(script_args, training_args, model_config):

    setup_hf_env()
    set_seed(training_args.seed, full_determinism=True)

    # Load dataset (the handoff from the curation module — columns: prompt/chosen/rejected)
    print("Loading dataset from: ", script_args.dataset_path)
    dataset = datasets.load_from_disk(script_args.dataset_path)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        cache_dir=os.environ.get("HF_MODELS_CACHE_DIR"),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    if tokenizer.pad_token is None:
        # Model-agnostic fallback (Qwen3 already defines a pad token, so this no-ops).
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'

    # Prepare training data — IDENTICAL formatting to train_dpo.py so DPO and length-normalized
    # DPO see the exact same prompt/chosen/rejected text (kept in sync by hand; it's ~10 lines).
    def format_assistant_turn(content):
        # Format a single assistant turn WITHOUT routing through apply_chat_template:
        # Qwen3's chat template strips <think>...</think> from assistant messages, but for
        # thinking-mode DPO we need the full reasoning response preserved in the
        # chosen/rejected completions.
        return f"<|im_start|>assistant\n{content}<|im_end|>\n"

    def format_dpo_sample(sample):
        # Harmlessness pairs have a single user turn (the harmful prompt). The chosen/rejected
        # completions already contain the <think>…</think> reasoning from generation.
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": sample["prompt"]}],
            tokenize=False,
            enable_thinking=True,
        )
        chosen = format_assistant_turn(sample["chosen"])
        rejected = format_assistant_turn(sample["rejected"])
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

    dataset = dataset.map(format_dpo_sample)
    dataset = dataset.select_columns(['prompt', 'chosen', 'rejected'])

    # Eval/checkpoint schedule is fully owned by StepwiseEvalCallback (step 0, every ~20% of
    # total training steps, and the final step) — force off the trainer's own interval-based
    # eval/save so the two mechanisms can't double-trigger or disagree (identical to DPO).
    training_args.eval_strategy = "no"
    training_args.save_strategy = "no"

    # LoRA config
    peft_config = get_peft_config(model_config)
    assert peft_config
    # ref_model=None + PEFT: the reference is the base model with the LoRA adapter disabled
    # (no separate model copy). Length normalization is applied to that reference pass too,
    # via LengthNormalizedDPOTrainer.concatenated_forward.
    ref_model = None

    best_checkpoint_callback = BestCheckpointCallback()
    stepwise_eval_callback = StepwiseEvalCallback(eval_fraction=0.2)
    # eval_metrics.jsonl — DPO eval metrics per eval point (step 0, each ~20% interval, final
    # step). Behavioural eval (safety + monitorability) is done separately, offline, on the
    # pushed checkpoints — not here.
    metrics_logger = EvalMetricsLoggerCallback(
        os.path.join(training_args.output_dir, "eval_metrics.jsonl"))
    callbacks = [best_checkpoint_callback, stepwise_eval_callback, metrics_logger]

    # Push-to-hub is optional: only wire the push callback when explicitly requested.
    if getattr(training_args, "push_to_hub", False) and getattr(training_args, "hub_model_id", None):
        callbacks.append(PushStepCheckpointCallback())

    print(f"dpo_norm config: loss_type={training_args.loss_type} beta={training_args.beta} "
          f"(length-normalized log-probs; reference-based sigmoid DPO — open-instruct dpo_norm)")

    trainer = LengthNormalizedDPOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=callbacks,
    )
    stepwise_eval_callback.trainer = trainer
    trainer.train()

    # Dump the trainer's FULL structured log history (every logging-step loss + LR + grad norm,
    # and every eval point) as one JSON — the complete, W&B-independent record.
    log_history_path = os.path.join(training_args.output_dir, "train_log_history.json")
    with open(log_history_path, "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print(f"Full training log history -> {log_history_path}")

    # GPU memory summary for batch-size decisions: peak reserved vs total is the OOM-relevant
    # headroom. Saved to gpu_memory.json (per-eval snapshots are in eval_metrics.jsonl).
    mem = gpu_memory_stats()
    if mem:
        mem_path = os.path.join(training_args.output_dir, "gpu_memory.json")
        with open(mem_path, "w", encoding="utf-8") as f:
            json.dump(mem, f, indent=2)
        headroom = mem["gpu_mem_total_gb"] - mem["gpu_mem_peak_reserved_gb"]
        print(f"GPU memory: peak reserved {mem['gpu_mem_peak_reserved_gb']} GB / "
              f"{mem['gpu_mem_total_gb']} GB total -> {headroom:.1f} GB headroom "
              f"(details: {mem_path})")

    # Get the best checkpoint and metrics (selected by eval reward-accuracy; ties by eval loss)
    best_step = best_checkpoint_callback.get_best_checkpoint()
    best_metrics = best_checkpoint_callback.get_best_metrics()
    print(f"Best checkpoint at step {best_step} with metrics: {best_metrics}")

    # Remove all non-best checkpoints and rename the best to checkpoint-best
    best_checkpoint_callback.clean_checkpoints(trainer.args.output_dir)

    # Record WHICH checkpoint won — including its HF repo id when checkpoints were pushed.
    pushed = getattr(training_args, "push_to_hub", False) and getattr(training_args, "hub_model_id", None)
    best_hf_repo = f"{training_args.hub_model_id}-step-{best_step}" if (pushed and best_step is not None) else None
    best_info_path = os.path.join(training_args.output_dir, "best_checkpoint.txt")
    with open(best_info_path, "w", encoding="utf-8") as f:
        f.write("Best checkpoint (selected by eval reward-accuracy; ties broken by eval loss)\n")
        f.write("method: dpo_norm (length-normalized DPO; open-instruct-style avg log-probs)\n")
        f.write(f"beta: {training_args.beta}\n")
        f.write(f"step: {best_step}\n")
        f.write(f"eval_rewards/accuracies: {best_metrics.get('accuracy')}\n")
        f.write(f"eval_loss: {best_metrics.get('loss')}\n")
        f.write(f"local: {os.path.join(training_args.output_dir, 'checkpoint-best')}\n")
        if best_hf_repo:
            f.write(f"hf_repo: {best_hf_repo}\n")
            f.write(f"hf_url: https://huggingface.co/{best_hf_repo}\n")
        else:
            f.write("hf_repo: (checkpoints were not pushed — run with --push_to_hub/HUB_MODEL_ID)\n")
    print(f"Best-checkpoint info -> {best_info_path}"
          + (f" (HF: {best_hf_repo})" if best_hf_repo else ""))

    # Also surface the winner in the W&B run summary (no-op if W&B isn't the reporter).
    try:
        import wandb
        if wandb.run is not None:
            wandb.run.summary["best_step"] = best_step
            wandb.run.summary["best_eval_rewards/accuracies"] = best_metrics.get("accuracy")
            wandb.run.summary["best_eval_loss"] = best_metrics.get("loss")
            if best_hf_repo:
                wandb.run.summary["best_hf_repo"] = best_hf_repo
    except Exception:
        pass


if __name__ == '__main__':
    parser = TrlParser((ScriptArguments, DPOConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()
    print(f"SCRIPT ARGS:\n{script_args}")
    print(f"TRAINING ARGS:\n{training_args}")
    print(f"MODEL CONFIG:\n{model_config}")
    main(script_args, training_args, model_config)
