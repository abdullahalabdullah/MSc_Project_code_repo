# Run on Pod (RunPod, 1× A100 SXM 80 GB)

> **Repo path note:** the commands below use `/workspace/Reward-Hacking-in-Explanations`.
> Adjust the path to wherever you actually cloned/rsynced the repo (e.g. if it's under
> `~/MSc_Project_code_repo/Reward-Hacking-in-Explanations`, use that). Ideally keep it on
> `/workspace` (the volume); the HF cache must be on `/workspace` regardless (see step 3).

---

## PART 2 — One-time setup (run on the pod)

SSH in, then:

```bash
# 0. sanity: GPU present
nvidia-smi

# 1. go to the repo (everything lives on the 80GB volume)
cd /workspace/Reward-Hacking-in-Explanations

# 2. create the venv the scripts expect (repo_root/my_env) + install deps
python3 -m venv my_env
source my_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt        # pulls torch + vllm; takes ~5–15 min

# 3. point the HF cache at the volume (so ~20GB of models don't fill the container disk)
echo "HF_MODELS_CACHE_DIR=/workspace/hf-cache" > .env

# 4. build the base dataset (downloads AQUA-RAT -> data/aqua/cot_negative.joblib)
python data/aqua/prepare_data.py

# 5. logs dir
mkdir -p logs
```

**OpenAI key** (only needed for the judge, i.e. the last eval script — `eval_responses_eval.sh`).
Export it when you're ready to run that step:

```bash
export OPENAI_API_KEY="sk-..."
```

---

## PART 3 — Smoke test (~50 examples, minutes)

```bash
cd /workspace/Reward-Hacking-in-Explanations
source my_env/bin/activate

# shrink to 50 examples
sed -i 's/^NUM_EXAMPLES=.*/NUM_EXAMPLES=50/' running_scripts/aqua/aqua_dpo_prepare_dataset.sh

# stage 1: generate -> parse -> reward -> build DPO pairs
bash running_scripts/aqua/aqua_dpo_prepare_dataset.sh 2>&1 | tee logs/smoke_prepare.log

# stage 2: DPO train (+ pushes adapter to your HF repo)
bash running_scripts/aqua/aqua_dpo_train_rw_gemma.sh 2>&1 | tee logs/smoke_train.log

# stage 3: merge + generate test responses
bash running_scripts/aqua/aqua_dpo_eval_responses_compute.sh 2>&1 | tee logs/smoke_eval_compute.log

# stage 4: judge (needs OPENAI_API_KEY) — run now if you have the key, else later
bash running_scripts/aqua/aqua_dpo_eval_responses_eval.sh 2>&1 | tee logs/smoke_eval_eval.log
```

### Smoke checks (go / no-go) — see `SMALL_SCALE_RUN_CHECKLIST.md` for detail

- In `smoke_prepare.log`: a healthy valid-rate and **`Mean diff rewards:` clearly positive** (not ~0).
- In `smoke_train.log`: the `[checkpointing] … every N steps` line prints, loss trends down, **no OOM**, and your private HF repo `aqua-dpo-qwen3-1.7B-sk-llama_original-seed-0` appears with an adapter.
- Quick valid-rate peek:

```bash
python -c "
import joblib, glob
f=sorted(glob.glob('outputs/aqua/qwen3-1.7B/dpo/seed_0/data/*_train_data.joblib'))[0]
d=joblib.load(f); flat=[r for ex in d for r in ex]
v=sum(r['response_is_valid'] for r in flat)
print(f'{f}\nvalid {v}/{len(flat)} = {100*v/len(flat):.1f}%')
"
```

If valid-rate is very low → the "best answer is" format isn't matching (the known risk;
the parser was already loosened to accept markdown bold like `**C**`).

---

## PART 4 — Full run (2000 examples)

```bash
cd /workspace/Reward-Hacking-in-Explanations
source my_env/bin/activate

# back to full size
sed -i 's/^NUM_EXAMPLES=.*/NUM_EXAMPLES=2000/' running_scripts/aqua/aqua_dpo_prepare_dataset.sh

bash running_scripts/aqua/aqua_dpo_prepare_dataset.sh        2>&1 | tee logs/full_prepare.log
bash running_scripts/aqua/aqua_dpo_train_rw_gemma.sh         2>&1 | tee logs/full_train.log
bash running_scripts/aqua/aqua_dpo_eval_responses_compute.sh 2>&1 | tee logs/full_eval_compute.log
bash running_scripts/aqua/aqua_dpo_eval_responses_eval.sh    2>&1 | tee logs/full_eval_eval.log
```

(Optional speed test before committing: early in `full_train.log`, confirm the DPO batch
size you may have bumped doesn't OOM on long sequences — watch `nvidia-smi`.)

---

## PART 5 — Save results, then terminate

The trained adapter is already safe on HF (pushed every half epoch). But the Volume disk is
wiped on termination, so pull your **logs + eval outputs** to your Mac first (run on **Mac**):

```bash
rsync -avz -e "ssh -p <POD_PORT> -i ~/.ssh/<your_key>" \
  root@<POD_IP>:/workspace/Reward-Hacking-in-Explanations/logs/ ./run_logs/
rsync -avz -e "ssh -p <POD_PORT> -i ~/.ssh/<your_key>" \
  root@<POD_IP>:/workspace/Reward-Hacking-in-Explanations/outputs/ ./run_outputs/
```

Then **terminate the pod** in the console.

---

## Quick reference of what each thing needs

| Step | Needs |
|---|---|
| Setup, data prep, generation, reward, DPO train | GPU + HF token (hardcoded) |
| HF adapter push | HF token (hardcoded) — pushes to a **private** repo |
| Judge (`eval_responses_eval.sh`) | `OPENAI_API_KEY` |
