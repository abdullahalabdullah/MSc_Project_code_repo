# ---------------------------------------------------------------------------
# Length-normalized, reference-free preference training — SimPO.
#   Meng, Xia & Chen (2024), "SimPO: Simple Preference Optimization with a
#   Reference-Free Reward."  https://arxiv.org/abs/2405.14734
#
# Added ALONGSIDE standard DPO — train_dpo.py is NOT modified; train.sh selects
# between the two via METHOD=dpo|simpo. All shared machinery (helpers + the four
# training callbacks) is imported from train_dpo.py so the two methods stay directly
# comparable (identical LoRA, eval/checkpoint schedule, per-step Hub push, best-
# checkpoint selection, metric logging, dataset formatting).
#
# SimPO differs from DPO in three ways, all handled by TRL's CPOTrainer with
# loss_type="simpo":
#   1. Length-normalized reward  r(x,y) = (beta/|y|) * sum_t log p(y_t | x, y_<t)
#      — the AVERAGE (not sum) of per-token log-probs. This removes the sequence-length
#      bias of DPO's unnormalized sum, which is the mechanism behind the observed
#      post-DPO response-length inflation (see
#      RO1/notes/harmless_dpo_response_length_inflation.md).
#   2. Reference-free — no reference-model term, and no ref forward pass at all, so it
#      uses less memory than DPO (relevant on the near-full 80GB H100).
#   3. Target reward margin gamma — loss is -log sigmoid(beta*(r_w - r_l) - gamma),
#      requiring the winner to beat the loser by at least gamma.
#
# cpo_alpha=0 gives PURE SimPO (no auxiliary SFT/NLL term); cpo_alpha>0 gives the
# CPO-SimPO hybrid.
# ---------------------------------------------------------------------------
"""LoRA SimPO training on the harmlessness preference pairs.

Consumes the same DatasetDict exported by the curation module (stage 06):
    DatasetDict({"train","valid"}) with columns prompt / chosen / rejected.

Run via train.sh (one command):
    METHOD=simpo bash train.sh <.../dpo_dataset>
or directly:
    python train_simpo.py --dataset_path <.../dpo_dataset> --model_name_or_path Qwen/Qwen3-4B \
        --loss_type simpo --beta 5 --simpo_gamma 1.5 --cpo_alpha 0 ...
"""
import json
import os

import datasets
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import CPOConfig, CPOTrainer, ModelConfig, TrlParser, get_peft_config

# Reuse ALL shared machinery from train_dpo.py unchanged (that file is not modified).
# Importing it also runs its module-level setup_hf_env(), so HF caches/token are
# configured on import — we call it again in main() too (idempotent) for clarity.
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


def main(script_args, training_args, model_config):

    setup_hf_env()
    set_seed(training_args.seed, full_determinism=True)

    # Load dataset (the handoff from the curation module — columns: prompt/chosen/rejected)
    print("Loading dataset from: ", script_args.dataset_path)
    dataset = datasets.load_from_disk(script_args.dataset_path)

    # Load model and tokenizer. CPOTrainer/SimPO is reference-free, so unlike DPO there is
    # no second (reference) model to load — only this policy model.
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        cache_dir=os.environ.get("HF_MODELS_CACHE_DIR"),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    # Prepare training data — IDENTICAL formatting to train_dpo.py so DPO and SimPO see the
    # exact same prompt/chosen/rejected text (kept in sync by hand; it's ~10 lines).
    def format_assistant_turn(content):
        # Format a single assistant turn WITHOUT routing through apply_chat_template:
        # Qwen3's chat template strips <think>...</think> from assistant messages, but for
        # thinking-mode preference training we need the full reasoning response preserved in
        # the chosen/rejected completions.
        return f"<|im_start|>assistant\n{content}<|im_end|>\n"

    def format_sample(sample):
        # Harmlessness pairs have a single user turn (the harmful prompt). The chosen/rejected
        # completions already contain the <think>…</think> reasoning from generation.
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": sample["prompt"]}],
            tokenize=False,
            enable_thinking=True,
        )
        return {
            "prompt": prompt,
            "chosen": format_assistant_turn(sample["chosen"]),
            "rejected": format_assistant_turn(sample["rejected"]),
        }

    dataset = dataset.map(format_sample)
    dataset = dataset.select_columns(["prompt", "chosen", "rejected"])

    # Eval/checkpoint schedule is fully owned by StepwiseEvalCallback (step 0, every ~20% of
    # total training steps, and the final step) — force off the trainer's own interval-based
    # eval/save so the two mechanisms can't double-trigger or disagree (identical to DPO).
    training_args.eval_strategy = "no"
    training_args.save_strategy = "no"

    # LoRA config
    peft_config = get_peft_config(model_config)
    assert peft_config

    best_checkpoint_callback = BestCheckpointCallback()
    stepwise_eval_callback = StepwiseEvalCallback(eval_fraction=0.2)
    # eval_metrics.jsonl — per eval point (step 0, each ~20% interval, final step). For SimPO
    # these are the CPO/SimPO eval metrics (eval_loss, eval_rewards/accuracies,
    # eval_rewards/margins, eval_logps/*). Behavioural eval (safety + monitorability) is done
    # separately, offline, on the pushed checkpoints — not here.
    metrics_logger = EvalMetricsLoggerCallback(
        os.path.join(training_args.output_dir, "eval_metrics.jsonl"))
    callbacks = [best_checkpoint_callback, stepwise_eval_callback, metrics_logger]

    # Push-to-hub is optional: only wire the push callback when explicitly requested.
    if getattr(training_args, "push_to_hub", False) and getattr(training_args, "hub_model_id", None):
        callbacks.append(PushStepCheckpointCallback())

    print(f"SimPO config: loss_type={training_args.loss_type} beta={training_args.beta} "
          f"simpo_gamma={training_args.simpo_gamma} cpo_alpha={training_args.cpo_alpha} "
          f"(cpo_alpha=0 -> pure SimPO; length-normalized, reference-free)")

    # CPOTrainer is reference-free: no ref_model argument (unlike DPOTrainer). The SimPO
    # length normalization + gamma margin are driven entirely by training_args.loss_type
    # ("simpo"), training_args.beta, and training_args.simpo_gamma.
    trainer = CPOTrainer(
        model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=callbacks,
    )
    stepwise_eval_callback.trainer = trainer
    trainer.train()

    # Dump the trainer's FULL structured log history (every logging-step loss + LR + grad
    # norm, and every eval point) as one JSON — the complete, W&B-independent record.
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
        f.write(f"method: simpo (length-normalized, reference-free; Meng et al. 2024)\n")
        f.write(f"beta: {training_args.beta}  simpo_gamma: {training_args.simpo_gamma}  "
                f"cpo_alpha: {training_args.cpo_alpha}\n")
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
    parser = TrlParser((ScriptArguments, CPOConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()
    print(f"SCRIPT ARGS:\n{script_args}")
    print(f"TRAINING ARGS:\n{training_args}")
    print(f"MODEL CONFIG:\n{model_config}")
    main(script_args, training_args, model_config)
