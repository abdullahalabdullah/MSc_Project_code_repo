# Eval-set scale — reduce n or N?

Decision for sizing the (smaller) `--data eval` split relative to the `test`
split. Scale of any monitorability run is set by two knobs: `n` (examples
sampled per dataset) and `N` (samples/arm per example). Question: which one
to cut to shrink the eval set.

## Recommendation

**Reduce n, hold N at 8. Don't cut N as the primary lever.**

## Reasoning — fixed-budget math

Monitor cost (≈90% of total cost) scales as `n × N`. The variance of the
headline metric (mean TE / g-mean² across the suite) decomposes as:

```
Var(mean) ≈ Var_between/n + Var_within/(n·N)
```

If total budget is held fixed (`n·N = C`), the second term becomes
`Var_within/C` — **constant, regardless of how the budget is split between n
and N.** Only the first term, `Var_between/n`, depends on the split, and it
only shrinks by increasing `n`. So for a fixed budget, more instances at
adequate N always beats fewer instances at higher N — consistent with the
earlier power analysis finding that N=16 bought nothing over N=8
(between-instance variance was already dominating; see
`project_ro1_experiment_design` memory).

## Caveat — granularity floor on N

The n=50/N=4 pilot was flagged "not final" partly because N=4 makes p0/p1
too coarse (∈{0,.25,.5,.75,1}) for the per-instance L1 filter (needs TE>0,
p1>0) to be reliable. N=8 was the point where that stopped being a problem.
Going below N=8 risks reintroducing that coarseness — a different failure
mode from the variance argument above, and one already paid to characterize
and avoid.

## Concrete recommendation

Keep N=8 unchanged, cut n. The `--data eval` default (`len(test)//2`)
already gives n=100 for most datasets — which happens to be exactly the
n=100/N=8 scale the earlier power analysis validated as the target design
point (±5–6pp CI on the mean-TE gate). The existing default eval-split size
isn't an arbitrary "half of test" — it already lands on the
previously-justified number. Use it as-is rather than re-deriving a new n.

If the eval set needs to be even cheaper (e.g. run after every one of the 4
RO2 checkpoints), drop n further (e.g. n=50, matching the original pilot)
rather than touching N — noisier per-checkpoint estimates, but L1-filter
validity is preserved. Since this is for relative checkpoint comparison
rather than a reported threshold gate, the extra noise is more tolerable
here than on the test-set headline number.

## Purpose (resolved)

Not checkpoint *selection* during training — **post-hoc checkpoint
comparison**. The eval set is run once per checkpoint after training
finishes (across the 4 RO2 checkpoints), to compare monitorability trends
between them, not to gate/pick a checkpoint mid-training.

Two implications for sizing:

- **No iteration-speed pressure.** Since it isn't run repeatedly inside the
  training loop, there's no cost-per-cycle reason to shrink further to
  n=50 — that fallback (from "Concrete recommendation" above) was for a
  frequent, cheap, in-loop check, which doesn't apply here. Default to
  n=100/N=8 (the `--data eval` default) rather than trimming further.
- **Paired design strengthens the case for keeping n=100.** The same eval
  instances are reused across all 4 checkpoints, so checkpoint-vs-checkpoint
  (or checkpoint-vs-base) comparisons are paired differences, not independent
  means. Paired SE depends on `SD_of_differences/√n`, and `SD_of_differences`
  is typically smaller than the raw between-instance SD (`Var_between` above)
  when instance-level difficulty correlates across checkpoints sharing the
  same base architecture/adapters — which it should, since they're LoRA
  checkpoints off the same base model. So the effective precision at n=100 is
  better than the raw ±5–6pp CI figure suggests, comfortably covering
  checkpoint-trend comparisons even though it's not the headline test-set
  number.

Net: use the eval default as-is (n=100, N=8) for this purpose; no further
reduction needed.

## Final decision (2026-07-09)

Confirmed: eval set = test-set design with n=100, N=8. No config changes
needed — `n_examples: 200` (test) and `qwen_inference.n_samples: 8` are
already the config defaults, and `01_prepare_dataset.py --data eval`
defaults `n_examples` to `len(test)//2` = 100 with no `--n-examples`
override, and N=8 applies with no `--N` override either. Running the
documented eval workflow as-is (`01 --data eval` →
`run_inference.sh ... --data eval` → `run_monitor_and_results.sh ... --data
eval`, no extra flags) already produces n=100/N=8. Excludes
`gpqa_diamond`/`lsat_ar` (no eval split — pool too small per
`eval_exclude_datasets`).
