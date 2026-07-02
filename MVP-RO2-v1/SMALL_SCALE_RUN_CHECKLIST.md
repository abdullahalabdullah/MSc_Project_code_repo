# Small-Scale Pre-Flight Checklist (run before the full A100 run)

Goal: confirm the whole Qwen3-1.7B DPO pipeline runs end-to-end and the reward-hacking
signal is present **before** spending hours/$$ on the full run. Do this with a tiny
dataset first, verify the checks below, then scale up.

Pipeline order: `prepare_dataset.sh` → `train_rw_gemma.sh` → `eval_responses_compute.sh`
→ `eval_responses_eval.sh`. All scripts are single-seed (`for SEED in 0`).

Run each script capturing logs:
```bash
bash running_scripts/aqua/aqua_dpo_prepare_dataset.sh 2>&1 | tee logs/prepare.log
```

---

## 0. One-time setup (do once, verify before anything)
- [ ] venv named exactly `my_env` created; `pip install -r requirements.txt` succeeded
      (torch, vllm, trl, peft, openai all import).
- [ ] Data generated: `python data/aqua/prepare_data.py` → `data/aqua/cot_negative.joblib` exists.
- [ ] HF token works (it is hardcoded in `env_setup.py`): `python -c "from huggingface_hub import whoami; print(whoami())"`.
- [ ] `OPENAI_API_KEY` exported (only needed for the final eval/judge step).
- [ ] `nvidia-smi` shows the A100 80GB and it's idle.

---

## 1. Configure the SMOKE TEST (temporary small values)
In `running_scripts/aqua/aqua_dpo_prepare_dataset.sh` temporarily set:
- [ ] `NUM_EXAMPLES=50`   (was 2000 — REVERT after the test)
- [ ] keep `N_SAMPLES=8` (or drop to 4 to go faster)

Everything else can stay. This makes a full pass take minutes, not hours.

---

## 2. Generation stage (vLLM) — `prepare_dataset.sh`
- [ ] vLLM loads `Qwen/Qwen3-1.7B` on a single GPU without error.
- [ ] Note the **tokens/sec** vLLM prints — use it to extrapolate full-run time
      (full train gen ≈ 2000 × 8 samples).
- [ ] **Eyeball a few raw responses** (load the `cot_negative_orig_train.joblib`): each
      should contain a `<think>...</think>` block followed by a final answer that says
      **"The best answer is: (X)"**.
  - 🔴 Red flag: if the model writes "the answer is (X)" **without "best"**, the parser
    drops it as invalid. (We are intentionally NOT changing the parser — so just be aware
    the valid-rate check below is the gate.)

## 3. Parsing / validity — after `get_responses_data.py`
Load a `*_data.joblib` and check the `response_is_valid` / `predicted_is_correct` fields.
- [ ] **Valid rate is healthy** (rough target ≳ 70–80%). If most responses are invalid,
      the answer format doesn't match the parser and the run is not worth scaling.
- [ ] `predicted_is_correct` is a sensible mix (not all False).
- [ ] Truncated-mid-think responses (no `</think>`) are marked invalid (expected).

## 4. Reward scoring — after `get_rewards_data.py`
- [ ] `Skywork-Reward-Llama-3.1-8B-v0.2` loads and scores (BSIZE=16) without OOM.
- [ ] Rewards have **spread** (not all near-identical). Load the joblib and check
      `np.std([r['reward'] for r in example])` per prompt is non-trivial.

## 5. DPO data prep — `dpo_prepare_dataset.py` stdout (MOST IMPORTANT CHECK)
Read these printed lines from the log:
- [ ] `Final: N examples` — N should be > 0 and close to the number of prompts.
- [ ] `# Original max ixs that did not have a valid answer` — should be small.
- [ ] **`Mean diff rewards: X ± Y`** — X should be **clearly positive** (chosen scored
      higher than rejected). 🔴 If ~0, there is no usable preference signal — STOP and debug.
- [ ] The saved dataset dir exists: `.../seed_0/data/cot_negative_sk-llama_original/`.

## 6. DPO training — `train_rw_gemma.sh`
- [ ] Training starts; **no truncation warnings flooding** (if many, `max_length 10240`
      is cutting sequences — note it).
- [ ] `loss` trends down; `eval_rewards/accuracies` > 0.5 (ideally higher).
- [ ] W&B runs in **offline** mode (no login prompt) — logs under `wandb/`.
- [ ] At the end: `checkpoint-best` created locally, and the adapter is **pushed to HF**
      (a private repo `aqua-dpo-qwen3-1.7B-sk-llama_original-seed-0` appears in your account).

### 6a. DPO batch-size / GPU-utilization test (do this here)
Default is `per_device_train_batch_size=1`, `gradient_accumulation_steps=16` — very
conservative for a 1.7B+LoRA model on 80GB. To use the GPU better, try bumping the batch
while keeping the **effective batch (= batch × accum) = 16**:
- [ ] Edit train script to `per_device_train_batch_size=2`, `gradient_accumulation_steps=8`.
- [ ] Run, and in another shell watch memory: `watch -n 2 nvidia-smi`.
- [ ] If it fits with headroom, try `4` / `4`. If you hit **OOM**, step back down.
- [ ] ⚠️ The real OOM risk is a batch full of **long** sequences (near 10240 tokens), which
      may not appear in the tiny smoke set — re-confirm the chosen batch size early in the
      FULL run too, and be ready to revert to `1 / 16` if it OOMs.
- [ ] Whatever value passes, keep effective batch at 16 so hyperparameters stay comparable.

## 7. Eval — `eval_responses_compute.sh` + `eval_responses_eval.sh`
- [ ] Merge step produces a merged model from `checkpoint-best`; generation runs; merged
      model is deleted afterward.
- [ ] Judge (OpenAI) returns text containing `OUTPUT: True/False`; `pf_ack` is populated
      (not all None). Requires `OPENAI_API_KEY`.

## 8. Final sanity of the RESULT (does the effect appear at all?)
Using the eval `*_data.joblib` (or the `analysis/` scripts):
- [ ] Accuracy is **higher on `orig` than `cf`** (model exploits the math book) — this is
      the precondition for the reward-hacking story.
- [ ] Acknowledgment rate does **not** rise to match the accuracy gap (ideally stays low) —
      i.e., it uses the book without admitting it. Even a directional hint here on the small
      run is encouraging; magnitudes will be noisy at NUM_EXAMPLES=50.

---

## 9. Go / No-Go before the full run
Proceed to full scale only if: valid-rate healthy (§3), mean reward diff clearly positive
(§5), DPO trains + pushes (§6), judge works (§7), and accuracy(orig) > accuracy(cf) (§8).

## 10. Revert to full-scale values
- [ ] `NUM_EXAMPLES=2000` (back from 50).
- [ ] Lock in the batch/accum that survived the OOM test (or revert to `1 / 16`).
- [ ] Re-run the four scripts in order, each `| tee` to a log.
