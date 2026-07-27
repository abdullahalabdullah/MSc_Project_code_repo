# Small-scale HP-validation run — log

> Copy this file (e.g. to `SMOKE_RUN_LOG_<date>.md`), fill it in after the run, and keep it
> with the thesis notes. "Expected" values are pass/fail guides — anything outside them
> needs a note in **Interpretation** and possibly a change in **Decisions**.

## Run metadata

| Field | Value |
|---|---|
| Date / GPU / pod | |
| Repo commit | |
| Curation run dir | `outputs/pilot` |
| SAMPLE / SEED | 4500 / 0 |
| Training OUTPUT_DIR | |
| HUB_MODEL_ID (pilot) | |
| W&B run URL | |
| Wall-clock: curation / training | |

## 1. Curation results (`outputs/pilot/report.md` + `diagnostics.json`)

| Metric | Expected | Observed | OK? |
|---|---|---|---|
| End-to-end survival | ~20–30% (pilot was 23.7%) | | |
| Final pairs | ~900–1350 | | |
| Truncation rate (stage 02) | < 1% | | |
| Sign-convention check | PASS | | |
| Labeler agreement (both) | broadly ≥ 70% | | |
| Pearson(margin, answer-length Δ) | \|r\| well below ~0.5 | | |
| Per-subset survival (van / adv) | both non-degenerate | | |

**`check_max_tokens.py` verdict** (paste): 

**Interpretation / anomalies:**

## 2. Training results (`eval_metrics.jsonl`, `train_log_history.json`, W&B)

| Metric | Expected | Observed | OK? |
|---|---|---|---|
| Total optimizer steps | ~150–250 | | |
| Train loss start → end | ~0.693 → clearly lower, smooth | | |
| Grad norm | stable, no spikes/NaN | | |
| eval_rewards/accuracies (final) | > 0.7 and rising | | |
| eval_rewards/margins | positive, growing | | |
| eval_loss trend | decreasing (no late U-turn) | | |
| Best checkpoint step | not step 0; note where it lands | | |
| Checkpoints pushed to HF | step 0, each ~20%, final — all present | | |
| gpu_mem_peak_reserved_gb / total | record it | | |
| sec / optimizer step (from logs) | record it | | |

**Full-run ETA extrapolation**: `sec/step × (final pair count ÷ 16 × 3 epochs)` = ______

**Interpretation / anomalies:**

## 3. Hyperparameter verdicts

| Knob | Current | Verdict (keep / change to X) | Evidence |
|---|---|---|---|
| learning_rate | 5e-6 | | loss/margin slope, grad norms |
| per_device_batch × accum | 1 × 16 | | peak reserved vs 80 GB (if < ~45–50 GB → 2 × 8) |
| effective batch | 16 | | curve noisiness (only change with deliberate LR retune) |
| epochs | 3 | | where best checkpoint landed; eval_loss U-turn? |
| MAX_LENGTH (trainer) | 12288 | | check_max_tokens trainer-side check |
| MAX_TOKENS (generation) | 8192 | | check_max_tokens verdict (change = global, pre-merge only) |

## 4. Decisions for the full run

- [ ] SAMPLE for top-up run: 5000 ÷ observed survival × 1.15 − pairs already curated ≈ ______
- [ ] Batch config: ______
- [ ] LR: ______
- [ ] Any config/code change needed before the full run? ______
- [ ] Distinct HUB_MODEL_ID + OUTPUT_DIR chosen for the full run (no collision with pilot)?

## 5. Issues / surprises log

| What happened | Root cause | Action taken |
|---|---|---|
| | | |
