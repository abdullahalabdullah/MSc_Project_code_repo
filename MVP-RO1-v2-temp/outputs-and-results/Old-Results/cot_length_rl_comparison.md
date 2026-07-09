# CoT Length: Base vs. SafeRL — Comparison & Interpretation

Comparison of chain-of-thought (CoT) token length between the base and RL'd models,
on the same 50 instances (N = 4 runs/instance) for the LSAT tasks.

- **Base:** `Qwen/Qwen3-4B` — run `2026-07-02_15-43-21`
- **RL'd:** `Qwen/Qwen3-4B-SafeRL` — run `2026-07-02_19-56-12`

Source: `gmean2_summary.md` (CoT length section) in each dataset's
`.../analysis/<dataset>/<model>/professor/` folder. Only `lsat_lr` and `lsat_ar`
exist in both runs, so only those are comparable.

---

## Table 1 — SafeRL (`Qwen/Qwen3-4B-SafeRL`)

CoT length (tokens):

| Dataset | Condition          |   Mean |    Std |  Min |   Max | Runs | Inst |
|---------|--------------------|-------:|-------:|-----:|------:|-----:|-----:|
| lsat_lr | Hinted (X=1, all)  | 1432.0 | 1107.2 |  386 |  5446 |  200 |   50 |
| lsat_lr | Unhinted (X=0, all)| 1256.1 |  858.4 |  322 |  4261 |  200 |   50 |
| lsat_ar | Hinted (X=1, all)  | 4968.0 | 2577.4 | 1094 | 14192 |  200 |   50 |
| lsat_ar | Unhinted (X=0, all)| 4618.9 | 2918.9 |  984 | 16238 |  200 |   50 |

## Table 2 — Base (`Qwen/Qwen3-4B`)

| Dataset | Condition          |   Mean |    Std |  Min |   Max | Runs | Inst |
|---------|--------------------|-------:|-------:|-----:|------:|-----:|-----:|
| lsat_lr | Hinted (X=1, all)  | 1859.3 | 1468.0 |  448 |  8254 |  200 |   50 |
| lsat_lr | Unhinted (X=0, all)| 1683.5 | 1449.2 |  388 |  9117 |  200 |   50 |
| lsat_ar | Hinted (X=1, all)  | 5920.1 | 3251.1 | 1268 | 14672 |  199 |   50 |
| lsat_ar | Unhinted (X=0, all)| 5274.1 | 3409.1 |  966 | 16267 |  198 |   50 |

---

## What the RL changes

The clearest and most consistent signal is that **RL makes the model's
chain-of-thought shorter and less variable.**

### 1. Shorter CoT across the board (−12% to −25%)

| Dataset | Condition | Base mean | SafeRL mean |     Δ |
|---------|-----------|----------:|------------:|------:|
| lsat_lr | Hinted    |      1859 |        1432 | −23%  |
| lsat_lr | Unhinted  |      1684 |        1256 | −25%  |
| lsat_ar | Hinted    |      5920 |        4968 | −16%  |
| lsat_ar | Unhinted  |      5274 |        4619 | −12%  |

RL trims reasoning length everywhere; the effect is stronger on the
shorter-reasoning task (lsat_lr) than the long-reasoning one (lsat_ar).

### 2. Tighter, less rambling distributions — mostly on lsat_lr

On **lsat_lr** the spread collapses: std falls from ~1468/1449 → ~1107/858, and
the max plummets from **8254/9117 → 5446/4261**. RL is cutting the long tail of
runaway reasoning.

On **lsat_ar** this does not happen — std stays ~2.5–3.4k and the max is
essentially unchanged (~14–16k). So the tail-trimming is **task-dependent, not
universal**.

### 3. Hint-sensitivity is preserved

Both models still "think" more when hinted (hinted > unhinted in every case). RL
does not remove that sensitivity — on lsat_ar it actually shrinks the
hinted−unhinted gap (+646 → +349 tokens), while on lsat_lr it holds (~+176
either way).

---

## Caveats

- This is **CoT length only** — it says nothing about whether the shorter
  reasoning is more accurate or more *faithful*. To claim RL improved behavior,
  pair this with the accuracy / opacity / g-mean² numbers from the same reports.
- The lsat_ar **min rises** after RL (984→1094, 966→1268 hinted/unhinted),
  consistent with RL enforcing a floor of reasoning while capping the top on the
  easier task.
- Only two datasets overlap between the runs; broader claims need the other
  datasets re-run on the base model.

## Bottom line

The RL training behaves like a **length-regularizer**: it compresses reasoning
(−12% to −25%) and suppresses pathologically long chains on the easier task,
while leaving the model's tendency to reason more under hints intact. Whether
that is beneficial depends on what happens to accuracy and faithfulness, which
these tables do not show.
