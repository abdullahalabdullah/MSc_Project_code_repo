"""Tokenization-equality gate for the SFT arm (SFT_Arm_Developer_Spec.md §2.2).

The whole ablation is void if the SFT arm trains on a sequence that differs by even one token
from what the DPO arm trained on for the same chosen response. This script proves, per
example, that the two training pathways produce identical token sequences:

  DPO side (what train_dpo.py + trl DPOTrainer actually trained on):
      prompt  = apply_chat_template([user], enable_thinking=True)          # format_dpo_sample
      chosen  = "<|im_start|>assistant\\n" + chosen + "<|im_end|>\\n"        # format_assistant_turn
      ids     = DPOTrainer.tokenize_row(...)  ->  prompt_input_ids + chosen_input_ids
      (tokenize_row appends eos_token_id to the completion and left-truncates the prompt to
       max_prompt_length — trl 0.22.2, trl/trainer/dpo_trainer.py::tokenize_row)

  SFT side (what train_sft.py + trl SFTTrainer will train on):
      prompt     = same apply_chat_template call                           # format_sft_sample
      completion = "<|im_start|>assistant\\n" + chosen + "<|im_end|>\\n" + eos_token
      then trl's own pathway (trl 0.22.2, trl/trainer/sft_trainer.py::_prepare_dataset):
        add_eos: appends eos_token iff completion doesn't already end with it (must be a no-op)
        prompt_ids = tok(prompt);  pc_ids = tok(prompt + completion)
        completion_mask = [0]*len(prompt_ids) + [1]*(len(pc_ids) - len(prompt_ids))

  Asserts, per example:
    1. pc_ids == dpo prompt_ids + chosen_ids          (full-sequence token equality)
    2. loss tokens (completion_mask==1) == chosen_ids (loss support identical to DPO's)
    3. last loss token == eos_token_id                (EOS included in the loss, spec §2.2)
    4. pc_ids startswith prompt_ids                   (prompt/loss boundary is clean)
    5. len(prompt_ids) <= --max_prompt_length         (else DPO left-truncated the prompt but
                                                       SFT would not — a real divergence)
    6. len(pc_ids) <= --max_length                    (else SFTTrainer right-truncates and the
                                                       EOS silently drops out of the loss)

  Plus converter fidelity when --sft_dataset is given: same row counts per split, and the
  sft_dataset's prompt/chosen columns equal the dpo_dataset's element-wise, in order.

Runs on CPU (tokenizer only). Exit code 0 = all checks passed; non-zero otherwise.

Usage:
    python verify_sft_tokenization.py \
        --dpo_dataset harmlessness-DPO-train-dataset-curation/outputs/main/dpo_dataset \
        --sft_dataset harmlessness-DPO-train-dataset-curation/outputs/main/sft_dataset
    # quick pass: --sample 200
"""
import argparse
import sys

import datasets
import numpy as np
from transformers import AutoTokenizer
from trl import DPOTrainer


def build_formatters(tokenizer):
    # Copies of train_dpo.py::format_assistant_turn/format_dpo_sample and
    # train_sft.py::format_sft_sample. Kept inline (this repo's training scripts are
    # self-contained by policy); any drift is exactly what this script exists to catch.
    def format_assistant_turn(content):
        return f"<|im_start|>assistant\n{content}<|im_end|>\n"

    def templated_prompt(raw_prompt):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": raw_prompt}],
            tokenize=False,
            enable_thinking=True,
        )

    def sft_completion(chosen):
        return format_assistant_turn(chosen) + tokenizer.eos_token

    return format_assistant_turn, templated_prompt, sft_completion


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dpo_dataset", required=True, help="path to the exported dpo_dataset")
    ap.add_argument("--sft_dataset", default=None,
                    help="path to the converted sft_dataset (adds converter-fidelity checks)")
    ap.add_argument("--model", default="Qwen/Qwen3-4B", help="tokenizer to verify with")
    ap.add_argument("--max_length", type=int, default=12288, help="train.sh MAX_LENGTH")
    ap.add_argument("--max_prompt_length", type=int, default=4096, help="train.sh MAX_PROMPT_LENGTH")
    ap.add_argument("--sample", type=int, default=None,
                    help="check only the first N examples per split (default: all)")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    eos = tokenizer.eos_token
    eos_id = tokenizer.eos_token_id
    print(f"Tokenizer: {args.model}  (eos={eos!r}, id={eos_id})")

    dpo = datasets.load_from_disk(args.dpo_dataset)
    assert set(dpo.keys()) == {"train", "valid"}, f"unexpected splits: {sorted(dpo.keys())}"

    # ── Converter fidelity: sft_dataset must be dpo_dataset minus `rejected`, row-for-row ──
    if args.sft_dataset:
        sft = datasets.load_from_disk(args.sft_dataset)
        assert set(sft.keys()) == {"train", "valid"}, f"unexpected sft splits: {sorted(sft.keys())}"
        for split in ("train", "valid"):
            assert sft[split].column_names == ["prompt", "chosen"], \
                f"sft '{split}' columns: {sft[split].column_names}"
            assert len(sft[split]) == len(dpo[split]), \
                f"row-count mismatch in '{split}': sft={len(sft[split])} dpo={len(dpo[split])}"
            for col in ("prompt", "chosen"):
                if sft[split][col] != dpo[split][col]:
                    sys.exit(f"FAIL: sft_dataset '{split}.{col}' differs from dpo_dataset — "
                             "the converter did not preserve rows/order.")
        print(f"Converter fidelity OK: sft_dataset == dpo_dataset minus 'rejected' "
              f"({len(sft['train'])} train / {len(sft['valid'])} valid rows)")

    _, templated_prompt, sft_completion = build_formatters(tokenizer)

    failures = 0
    checked = 0
    seq_lens, prompt_lens = [], []

    for split in ("train", "valid"):
        d = dpo[split]
        n = min(args.sample, len(d)) if args.sample else len(d)
        print(f"Checking {n}/{len(d)} examples in '{split}' ...")
        for i in range(n):
            row = d[i]
            prompt_text = templated_prompt(row["prompt"])
            chosen_text = f"<|im_start|>assistant\n{row['chosen']}<|im_end|>\n"

            # DPO side: the real trl staticmethod, called exactly as DPOTrainer._prepare_dataset
            # does (add_special_tokens=False for decoder-only; prompt left-truncated).
            tok_row = DPOTrainer.tokenize_row(
                {"prompt": prompt_text, "chosen": chosen_text, "rejected": chosen_text},
                tokenizer,
                max_prompt_length=args.max_prompt_length,
                max_completion_length=None,
                add_special_tokens=False,
            )
            dpo_prompt_ids = tok_row["prompt_input_ids"]
            dpo_chosen_ids = tok_row["chosen_input_ids"]
            dpo_ids = dpo_prompt_ids + dpo_chosen_ids

            # SFT side: replicate trl 0.22.2 sft_trainer.py add_eos + tokenize closure.
            completion_text = sft_completion(row["chosen"])
            if not completion_text.endswith(eos):  # add_eos — must be a no-op for us
                completion_text = completion_text + eos
            prompt_ids = tokenizer(text=prompt_text)["input_ids"]
            pc_ids = tokenizer(text=prompt_text + completion_text)["input_ids"]
            completion_ids = pc_ids[len(prompt_ids):]

            errs = []
            if pc_ids != dpo_ids:
                errs.append("full-sequence tokens differ from DPO")
            if completion_ids != dpo_chosen_ids:
                errs.append("loss-token set differs from DPO chosen tokens")
            if not completion_ids or completion_ids[-1] != eos_id:
                errs.append("last loss token is not EOS")
            if pc_ids[:len(prompt_ids)] != prompt_ids:
                errs.append("prompt is not a prefix of prompt+completion")
            if len(prompt_ids) > args.max_prompt_length:
                errs.append(f"prompt {len(prompt_ids)} tokens > max_prompt_length "
                            f"{args.max_prompt_length} (DPO truncated, SFT would not)")
            if len(pc_ids) > args.max_length:
                errs.append(f"sequence {len(pc_ids)} tokens > max_length {args.max_length} "
                            "(SFT right-truncation would drop the EOS from the loss)")

            if errs:
                failures += 1
                if failures <= 10:
                    print(f"  FAIL [{split}#{i}]: {'; '.join(errs)}")
            checked += 1
            seq_lens.append(len(pc_ids))
            prompt_lens.append(len(prompt_ids))

    seq_lens = np.array(seq_lens)
    prompt_lens = np.array(prompt_lens)
    print("\n── Report ──")
    print(f"examples checked : {checked}")
    print(f"failures         : {failures}")
    print(f"sequence tokens  : max={seq_lens.max()}  p99={int(np.percentile(seq_lens, 99))}  "
          f"p50={int(np.percentile(seq_lens, 50))}  (budget {args.max_length})")
    print(f"prompt tokens    : max={prompt_lens.max()}  p99={int(np.percentile(prompt_lens, 99))}  "
          f"(budget {args.max_prompt_length})")

    if failures:
        print(f"\nRESULT: FAIL — {failures}/{checked} examples diverge. Do NOT train the SFT arm.")
        sys.exit(1)
    print(f"\nRESULT: PASS — all {checked} examples token-identical to the DPO chosen pathway, "
          "EOS in loss, no truncation risk.")


if __name__ == "__main__":
    main()
