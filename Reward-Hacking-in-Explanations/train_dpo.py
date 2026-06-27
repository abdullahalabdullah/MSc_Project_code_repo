import math
import os
import shutil
from dataclasses import dataclass
from glob import glob

import datasets
import torch
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    TrlParser,
    get_peft_config,
)

import env_setup
from utils import set_seed

env_setup.setup_env(use_temp=False)

# Authenticate for pushing adapters to the Hub. Token is centralized in env_setup
# (already exported to HF_TOKEN by setup_env above).
login(token=os.environ["HF_TOKEN"])


class BestCheckpointCallback(TrainerCallback):
    def __init__(self):
        self.best_reward = float('-inf')
        self.best_loss = float('inf')
        self.best_checkpoint = None
        
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        current_reward = metrics.get("eval_rewards/accuracies")
        current_loss = metrics.get("eval_loss")

        if (current_reward > self.best_reward) or (current_reward == self.best_reward and current_loss < self.best_loss):
            self.best_reward = current_reward
            self.best_loss = current_loss
            self.best_checkpoint = state.global_step

    def get_best_checkpoint(self):
        return self.best_checkpoint
    
    def get_best_metrics(self):
        return {'accuracy': self.best_reward, 'loss': self.best_loss}

    def clean_checkpoints(self, output_dir):
        """Remove all non-best checkpoints and rename the best one to 'checkpoint-best'"""

        if self.best_checkpoint is None:
            print("No best checkpoint found.")
            return False

        best_checkpoint_path = os.path.join(output_dir, f"checkpoint-{self.best_checkpoint}")
        best_target_path = os.path.join(output_dir, "checkpoint-best")

        # Check that the best checkpoint exists
        if not os.path.exists(best_checkpoint_path):
            print(f"Best checkpoint directory {best_checkpoint_path} not found.")
            return False

        # Get all checkpoint directories
        checkpoint_dirs = [d for d in glob(os.path.join(output_dir, "checkpoint-*")) if os.path.isdir(d)]

        # Remove the target directory if it already exists
        if os.path.isdir(best_target_path):
            print("Checkpoint-best already exists")
            return False
        
        # Copy the best checkpoint to the new name
        try:
            shutil.copytree(best_checkpoint_path, best_target_path)
        except Exception as e:
            print(f"Error copying best checkpoint: {e}")
            return False

        # Remove all other checkpoints
        for checkpoint_dir in checkpoint_dirs:
            try:
                assert os.path.basename(checkpoint_dir).split("-")[0] == "checkpoint"
                shutil.rmtree(checkpoint_dir)
                print(f"Removed checkpoint: {checkpoint_dir}")
            except Exception as e:
                print(f"Error removing checkpoint {checkpoint_dir}: {e}")

        print(f"Successfully renamed checkpoint-{self.best_checkpoint} to checkpoint-best")
        
        return True


def main(script_args, training_args, model_config):

    set_seed(training_args.seed, full_determinism=True)

    # Load dataset
    print("Loading dataset from: ", script_args.dataset_path)
    dataset = datasets.load_from_disk(script_args.dataset_path)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        cache_dir=os.environ.get("HF_MODELS_CACHE_DIR")
    )

    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    if tokenizer.pad_token is None:
        # Model-agnostic fallback (Qwen3 already defines a pad token, so this no-ops).
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'

    # Prepare training data
    def format_assistant_turn(content):
        # Format a single assistant turn WITHOUT routing through apply_chat_template:
        # Qwen3's chat template strips <think>...</think> from assistant messages, but
        # for thinking-mode DPO we need the full reasoning response preserved in the
        # chosen/rejected completions.
        return f"<|im_start|>assistant\n{content}<|im_end|>\n"

    def format_dpo_sample(sample):
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": sample["prompt"]},
                {"role": "user", "content": sample["query"]},
            ],
            tokenize=False,
            enable_thinking=True,
        )
        chosen = format_assistant_turn(sample["chosen"])
        rejected = format_assistant_turn(sample["rejected"])
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

    dataset = dataset.map(format_dpo_sample)
    dataset = dataset.select_columns(['prompt', 'chosen', 'rejected'])

    # Save & push checkpoints twice per epoch (every half epoch), epoch-count-proof.
    # HF only supports a global step interval, so we derive steps-per-epoch from the data
    # and set the save/eval interval to half of it. This overrides any save_steps/eval_steps
    # passed on the CLI (a static fraction like 0.5 would only equal half an epoch at 1 epoch).
    # Assumes a single training process (single-GPU Runpod run).
    n_train = len(dataset["train"])
    per_device_bs = max(1, training_args.per_device_train_batch_size)
    steps_per_epoch = math.ceil(math.ceil(n_train / per_device_bs) / training_args.gradient_accumulation_steps)
    half_epoch_steps = max(1, steps_per_epoch // 2)
    training_args.save_steps = half_epoch_steps
    training_args.eval_steps = half_epoch_steps
    print(f"[checkpointing] {n_train} train pairs -> {steps_per_epoch} steps/epoch; "
          f"saving & pushing every {half_epoch_steps} steps (every half epoch)")

    # Define lora config
    peft_config = get_peft_config(model_config)
    assert peft_config
    ref_model = None

    # Define trainer
    trainer = DPOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[BestCheckpointCallback]
    )
    trainer.train()

    # Get the best checkpoint and metrics
    def get_callback_by_class(trainer, cls):
        for callback in trainer.callback_handler.callbacks:
            if isinstance(callback, cls):
                return callback
        return None

    best_checkpoint_callback = get_callback_by_class(trainer, BestCheckpointCallback)
    best_step = best_checkpoint_callback.get_best_checkpoint()
    best_metrics = best_checkpoint_callback.get_best_metrics()
    print(f"Best checkpoint at step {best_step} with metrics: {best_metrics}")

    # Remove all non best checkpoints and rename to best checkpoint
    best_checkpoint_callback.clean_checkpoints(trainer.args.output_dir)


@dataclass
class ScriptArguments:
    dataset_path: str


if __name__ == '__main__':
    parser = TrlParser((ScriptArguments, DPOConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()
    print(f"SCRIPT ARGS:\n{script_args}")
    print(f"TRAINING ARGS:\n{training_args}")
    print(f"MODEL CONFIG:\n{model_config}")
    main(script_args, training_args, model_config)    
