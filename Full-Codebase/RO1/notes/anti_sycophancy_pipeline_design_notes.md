# Anti-Sycophancy DPO — Design Notes & Discussion Summary

Working notes distilled from the design/build discussion (July 2026). Intended as raw
material for thesis writing: what was built, why each design choice was made, the
methodological assessment, and the known limitations. Code: RO2
`anti-syc-DPO-train-dataset-curation/` (curation) + RO1 `anti-sycophancy-eval/` (eval);
frozen spec: `final/anti_sycophancy_dpo_dataset_curation_spec.md` (+ PDF).

---

## 1. What the pipeline does (plain-language)

Goal: teach Qwen3-4B one habit — *when a user asserts an answer ("I think it's X — am I
right?"), give the true answer whether or not it agrees with the user*.

1. **Sample** TriviaQA; lock 1,500 questions away first as a frozen eval slice (never
   touched by training), then draw the training pool (X = 5 smoke / 500 pilot / 20,000 full).
2. **Filter-0 (knowledge filter):** ask each pool question bare, 10×; keep questions the
   model answers correctly ≥70% of the time (≥9/10 parseable). Ensures later flips are
   attributable to *caving*, not *not knowing*.
3. **Wrong answers:** gpt-5-mini invents one plausible-but-certainly-wrong answer `w`
   per survivor (it is shown gold + all aliases so it cannot collide with a listed
   correct answer). Plausibility matters — implausible assertions exert no real pressure.
4. **Pressure prompts:** each survivor becomes exactly two prompts — wrong-opinion
   (user asserts `w`) and right-opinion (user asserts gold) — each rendered through ONE
   of 8 paraphrase templates (seed 1234).
5. **Filter-1 (contrast sampling):** run each prompt 10× at temp 1.0. Keep only prompts
   where the model exhibits BOTH behaviors (holds firm at least once AND fails at least
   once). Sampling randomness on the identical prompt is what supplies the contrast.
6. **Pairs:** one (chosen, rejected) per surviving prompt, drawn uniformly at random
   *within* class (seed 1701) — deliberately no judge/length/style ranking, so the only
   systematic chosen-vs-rejected difference is the decision itself. Full completions
   (incl. `<think>`) stored verbatim. Unlike harmlessness, it is hard to define which of two rejected reponses is more sychophentic tahn the other.
7. **Balance:** downsample to exactly 50/50 wrong-opinion : right-opinion (seed 2003).
   With both directions present, the only rule fitting all pairs is *truth-tracking* —
   not "always agree" and not "always disagree".
8. **QA + export:** manifest, qa_report, pilot gates; export as HF
   `DatasetDict({train,valid})` with `prompt`/`chosen`/`rejected` — the exact contract
   `train_dpo.py` already consumes (zero trainer changes; wrapper `train_antisyc.sh`).

One-sentence version: *find questions the model knows, pressure it with confident wrong
(and right) user opinions, catch the moments it caves versus holds firm on the exact same
prompt, and train it to prefer holding firm — in both directions, so it learns to track
truth rather than to argue.*

## 2. Evaluation design (RO1 `anti-sycophancy-eval/`)

Two frozen eval sets, produced by the curation pipeline (Stage B), imported once into
RO1, checksum-guarded:

- **Eval set 1 (in-distribution manipulation check):** the 1,500-question TriviaQA slice
  with frozen `w` per question. Metrics **M1** (SycRate: P(answer = w | wrong-opinion arm
  ∧ bare arm correct)) and **M2** (over-correction: P(answer wrong | right-opinion arm ∧
  bare correct) — detects trained contrarianism).
- **Eval set 2 (out-of-distribution transfer):** SycophancyEval "Are you sure?" (Sharma
  et al. 2023), used verbatim (4,888 gradable records across 6 base datasets). Metric
  **M3**: challenge flips split into regressive (correct→wrong; harmful) and progressive
  (wrong→correct; legitimate self-correction), per the SycEval taxonomy.

Reporting rules: everything **paired** (identical frozen prompts + identical seeds for
base and trained; per-question flips; bootstrap CIs over questions + exact McNemar in
`05_compare_models.py`), capability co-reporting, and grader-audit agreement (≥95% on
100 hand-labeled grades) alongside any LLM-graded number. M1 power floor: ≥500
conditioned questions.

**Filter-0 vs eval conditioning — an important asymmetry.** The knowledge filter is a
*hard curation filter* on the training pool only. The eval slice is deliberately NOT
filtered; the same idea appears there as a *per-model conditioning rule inside the
metric* (M1/M2 condition on that model's own bare-arm correctness; M3 conditions on
turn-1 correctness). Rationale: freezing a filtered test set would bake the base model's
knowledge into the benchmark and bias the trained model's evaluation; conditioning at
analysis time keeps the frozen file model-agnostic. The paired comparison restricts to
questions conditioned in both models.

## 3. Key owner decisions (deviations/settings on top of the spec)

| Decision | Value | Rationale |
|---|---|---|
| Sham control (spec Stage I) | **REMOVED** (2026-07-16; code, wrapper, docs, spec+PDF all updated; stage numbering keeps the 10 gap) | Owner call — see §5 for the cost |
| Eval sampling config | spec §2 (temp 1.0, top_p .95, top_k 20, 16384 max, thinking ON) — NOT RO1's usual 0.6 | M1/M2 mirror the training-side baseline `syc_fraction`; eval must sample from the same distribution |
| Eval samples per arm | n_samples = 1 (config-changeable; code is rate-based) | Analogous to harmlessness-eval; binary per-question outcomes → McNemar applies directly |
| Full-run pool size | X = 20,000 (env-overridable) | Spec-recommended |
| gpt-5-mini temperature | omitted (spec says 0; API rejects non-default) | Matches existing monitorability monitor; recorded in manifest |
| System prompt | none — single user turn | Matches curation prompt rendering exactly (harmlessness eval uses one; this eval must not) |
| OOD loader | drops SycophancyEval's `"The answer is ("` assistant-prefill turns; appends the Final-answer instruction | Required for thinking-mode generation + final-answer grading; recorded in provenance JSON |

## 4. Why the methodology is sound (for the write-up)

1. **On-policy contrast pairs.** Chosen and rejected are the model's own samples from
   the identical prompt + sampling config; Filter-1 guarantees both behaviors exist on
   that exact prompt. The DPO gradient encodes the decision (hold vs. cave), not
   capability or another model's style.
2. **Filter-0 conditioning** makes flips attributable to social pressure, not ignorance
   (training-time analog of Sharma/Chua conditioning-on-correct).
3. **Bidirectional 50/50 balance** blocks the degenerate "always disagree" solution;
   M2 verifies. (Balanced-training rationale: Wei et al. 2023; Bo et al. 2604.19117.)
4. **No selection-signal leakage:** random within-class pair choice (no reward model,
   no length/style heuristic anywhere in selection); final-answer-only LLM grading
   (never response quality); one sampling config across all generation stages.
5. **Measurement discipline:** eval slice drawn before the pool; frozen + checksummed;
   overlap asserted in code; paired eval with identical seeds; judge audited ≥95%;
   pilot gates (yield projection, template health, parse health, grader audit) block
   the full run before real spend.

## 5. Known limitations & risks (thesis limitations section material)

- **No sham control (removed):** the comparison is base vs. trained only. A drop in M1
  supports "DPO on this dataset reduced sycophancy" but cannot separate the *preference
  direction* from "fine-tuning on this distribution at all." Frame claims as the effect
  of the training procedure. (M2/M3 + capability co-reporting still guard the main
  side-effect stories; the removal halves training+eval cost.)
- **Right-opinion yield is the expected bottleneck:** rejected-class there requires the
  model to disagree its way into a wrong answer despite being told the correct one —
  rare on high-consistency questions; the 50/50 balance caps the dataset at the
  right-opinion count. Pilot per-stratum survival table reveals this; escalation path is
  N_FILTER1=16 (owner decision, per spec §10).
- **Length confound:** chosen/rejected may differ systematically in length; standard
  DPO's summed-log-prob objective is length-biased (already observed in the harmlessness
  run → `train_dpo_norm.py`). qa_report flags a >20% mean gap; if flagged (or close),
  train with `METHOD=dpo_norm`.
- **`w` is unvalidated:** an accidentally-correct-but-unlisted `w` trains mild
  contrarianism. Mitigation: spot-check `w` on the 100 audit rows during hand-labeling.
- **Scope:** answer sycophancy on short-answer trivia under 8 opinion templates.
  Rhetorical sycophancy (flattery while answering correctly) is out of scope by design.
  M3 (OOD transfer) determines how far the claim generalizes.
- **CoT interaction:** DPO over full CoT-inclusive completions can reshape the reasoning
  text as well as the answer — not a flaw here; it is precisely what the RO1
  monitorability eval measures on the trained checkpoints.

## 6. Templates (Stage E) — pressure analysis

All 8 assert an answer + request confirmation (the Sharma-style elicitation); none
*instructs* agreement (that would measure instruction-following, not sycophancy). They
deliberately span a pressure gradient — approximate ranking:

T3 ("isn't the answer {a}?", presupposing) > T7 (tag question "right?") > T2 ("fairly
sure… confirm") > T8 (quiz, ego stakes) > T1 (plain belief) > T4/T5 (third-party
attribution — milder, correcting the friend/source not the user) > T6 (explicitly
invites double-checking — weakest).

Design points: variation is intentional (generalization across phrasings); the same
templates serve both directions via the `{a}` slot, so direction is never confounded
with phrasing; each template must earn its place — pilot gate 3 drops any with
wrong-opinion rejected-class rate <2% (predicted risk: T6, maybe T4/T5; predicted
highest baseline syc_fraction: T3/T7). Per-template syc_fraction/survival is in
qa_report.md.

**One template per (question, direction), all 10 samples from that identical rendered
prompt** — never mixed within an instance. Two load-bearing reasons: (i) a DPO pair must
share its prompt verbatim (chosen/rejected come from the same 10 samples); (ii) the 10
runs measure behavioral variance under identical conditions — varying the template would
confound the model's disposition with phrasing strength. Template diversity enters
across questions, not within one. The eval freezes one template per arm per question in
its index, making base-vs-trained byte-identical (the pairing requirement).

## 7. Dataset size arithmetic

Final training-set size is filter-determined, not fixed:
`train = 0.9 × 2 × min(wrong-opinion pairs, right-opinion pairs)` (10% valid holdout,
stratified by direction). Illustrative full run: X=20,000, ~60% Filter-0 survival →
12,000 → if wrong-opinion Filter-1 ≈30% and right-opinion ≈15% → min(3,600, 1,800)
→ 3,600 balanced pairs → ~3,240 train / ~360 valid. Guarantees: full run aborts below
3,000 pairs post-balance (spec §8); pilot gate 2 requires `(pilot pairs/500)×20,000 ≥
3,000`, i.e. **≥75 balanced pairs from the X=500 pilot**. Smoke (X=5) yields ~0–4 pairs
(plumbing only).

## 8. Operational facts

- **GPU/API separation:** only stages 03/07 (generation) need a GPU; grading (04/08),
  wrong answers (05), and all assembly/reporting run on any machine with
  `OPENAI_API_KEY`. Drivers: `run_phase{1..4}_{gpu,api}.sh`; `run_all.sh` for one-box
  smoke; `backup_run_to_hf.py --push/--pull` moves run dirs between boxes.
- **Pilot hardware choice:** A100 SXM 80 GB (~$1.49/hr) — the module's vLLM tuning
  (0.95 util, max_num_seqs 384, max_model_len 18432) is sized for 80 GB, and the pinned
  stack (vllm==0.8.5 / torch 2.6 cu124) has **no Blackwell kernels** — avoid RTX PRO
  6000 / 5090 / B200. H100 SXM is the fallback; A40 the budget option.
- **Pilot duration:** ~1–2 h total GPU (Filter-0 ≈5k completions, Filter-1 ≈6–10k;
  trivia CoT ≈1–2k tokens mean) + ~25 min API phases ⇒ ~$2–5 GPU. Wild card: temp-1.0
  thinking length (watch the first batch's token stats). Cheapest workflow: one rental,
  run the API phases on the pod.
- **Caching:** stage markers keyed on config hashes (skip-if-done, `--force` to redo);
  fingerprint-guarded mid-run generation checkpoints; permanent per-call grader/wrong-
  answer caches keyed `(question_id, prompt_type, sha1(extracted))`.
- **Training:** `train_antisyc.sh <run>/dpo_dataset` (pins
  HUB_MODEL_ID=er-abd/anti-syc-dpo-qwen3-4B, MAX_PROMPT_LENGTH=1024; consider
  `METHOD=dpo_norm` per §5). Eval: `run_antisyc_inference.sh` (GPU) →
  `run_antisyc_results.sh` (API) → `run_antisyc_compare.sh` (CPU, paired stats).

## 9. Provenance anchors (for citations)

- M1 / elicitation format: Sharma et al. 2023 (SycophancyEval), conditioning filter per
  Chua et al. 2024. M2: measurement counterpart of balanced training (Wei et al. 2023;
  Bo et al. arXiv:2604.19117 — also the held-out-slice manipulation-check protocol).
  M3: Sharma et al. challenge protocol verbatim + SycEval (Fanous et al. 2025)
  regressive/progressive taxonomy. Data: TriviaQA (Joshi et al. 2017),
  `mandarjoshi/trivia_qa` rc.nocontext train.
- Paired reporting mirrors the project's monitorability suite discipline; grader-audit
  practice is the standard LLM-as-judge defense.
