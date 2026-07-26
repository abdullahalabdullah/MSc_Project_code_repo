# Mechanistic follow-up: why did DPO raise monitorability?

A small sub-study on top of the monitorability results. We have a shared base
(`Qwen/Qwen3-4B`) and two independent DPO interventions from it:

- **H** = `er-abd/harmless-dpo-qwen3-4B-step-735` (harmlessness)
- **S** = `er-abd/anti-syc-dpo-qwen3-4B-step-520` (anti-sycophancy)

plus controls **SFT** (`harmless-sft-qwen3-4B-step-735`, same data as H, different
objective) and **H147** (`harmless-dpo-qwen3-4B-step-147`, earlier checkpoint of H).

Both DPO runs independently raised hint verbalization / monitorability. The
question: **did they move the model along the *same* direction, and is that
direction the cause?**

## Scripts (run from the `RO1/` directory)

| Script | What it does | Cost |
|---|---|---|
| `lora_diff.py` | Weight-space cosine + subspace overlap of the LoRA `ΔW = (α/r)·B·A` per module, pairwise (H~S, H~SFT, S~SFT, H147~H735). Factor-space math, never materializes ΔW. | seconds (CPU) |
| `act_extract.py` | Paired residual-stream diffs `Δ = mean(act_adapter − act_base)` per layer on the real cued/uncued prompts (adapter-swap on one base via PEFT). Saves `delta_vectors.npz`. | ~28 min (MPS) |
| `analyse_deltas.py` | Splits generic vs hint-specific shift (`δ = Δ_cued − Δ_uncued`) and reports cross-model agreement. Reads `delta_vectors.npz`. | seconds |
| `cross_steer.py` | Injects the Δ/δ vectors into the base model during generation and measures whether hint verbalization rises (sufficiency + interchangeability). Keyword scorer for calibration; monitor scorer TODO. | minutes–hours (MPS, generation) |

Outputs land in `outputs-and-results/mechinterp/`.

## Findings so far

1. **Weight space (`lora_diff.py`):** H and S are near-orthogonal —
   `cos(ΔW_H, ΔW_S) ≈ 0.05` overall (random floor 0.007), vs **0.72** for two
   checkpoints of the same run and **0.18** for H~SFT (same data, different
   objective). The two DPO runs did **not** learn the same weights.

2. **Activation space (`act_extract.py` + `analyse_deltas.py`):** partial
   convergence, concentrated in the hint-specific component.
   - Raw last-token `cos(Δ_H, Δ_S)` ≈ **+0.13** mean over layers (peaks ~0.25–0.52).
   - Each adapter's shift is ~99% generic (`cos(Δ_cued, Δ_uncued) ≈ 0.97–0.99`);
     only ~14–20% of the shift magnitude is hint-specific.
   - The **hint-specific** components agree ~2× more:
     `cos(δ_H, δ_S) ≈ +0.28` mean over layers 5–35, peaking **0.35–0.54 in mid
     layers 6–16** (random 2σ ≈ 0.04).
   - The mean-pool readout is corrupted by Qwen massive-activation sink tokens
     (flips to ≈ −0.5 at deep layers) — use the **last-token** readout only.

   Reading: no single dominant shared axis, but the two interventions converge
   on a shared **hint-response subspace** in mid layers, reached by orthogonal
   weight updates.

3. **Causality (`cross_steer.py`):** in progress — does injecting δ_S/δ_H into
   base reproduce the verbalization gain, and are the two interchangeable?

See `../../../RO1/documentation.md` (Mechanistic follow-up section) for context.
