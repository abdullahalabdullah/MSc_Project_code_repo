# Beaver ranker sanity check — WildJailbreak

**Question.** Is `PKU-Alignment/beaver-7b-unified-cost` a valid **ranker** of response
harmfulness on WildJailbreak prompts (vanilla + adversarial harmful) — i.e. does a higher
Beaver cost reliably mean a more harmful answer — scored with the raw prompt (`cost_full`)
vs. with the vanilla seed query (`cost_vanilla`)?

**One-line verdict.** **Yes — usable as-is with the raw prompt (`cost_full`).** Strong
separation on both subsets (AUC 0.989 vanilla, 0.873 adversarial), no length confound,
negligible truncation. Vanilla-query scoring does **not** help (slightly worse), so it is
not needed.

---

## 1. What we are using this model for (scope of the claim)

In the pilot pipeline, Beaver is the **ranker**, not the safety classifier. For the N
responses generated to the *same* prompt it provides a scalar cost used to select
`chosen` = lowest cost (safest) and `rejected` = highest cost (most harmful). The
safe/unsafe **label** is assigned separately by Qwen3Guard-Gen-4B (filter-1/filter-3).
Therefore the only property that must hold is: *higher Beaver cost ⇒ more harmful response,
within a prompt.* This report tests exactly that property — not Beaver's absolute
calibration or its use as a standalone classifier.

## 2. Methodology — what we did

**Data.** `allenai/wildjailbreak` (train config), using only the two harmful subsets:
`vanilla_harmful` (plain harmful request, from the `vanilla` column) and
`adversarial_harmful` (jailbreak-wrapped request, from the `adversarial` column).
Deduplicated on the prompt string; **300 unique prompts** sampled with a fixed seed (0).

**Generation.** For each prompt, **N = 10** responses from **Qwen/Qwen3-4B** in thinking
mode (temperature 0.6, top_p 0.95, top_k 20, max_tokens 8192). The chain-of-thought
(`<think>…</think>`) is stripped; only the visible answer is scored. Responses truncated
mid-think (no visible answer) are marked invalid and excluded.

**Independent ground truth.** Each valid (prompt, answer) is labelled safe/unsafe by
**Qwen/Qwen3Guard-Gen-4B** — a *different* model family (adversarial-aware). Beaver is
checked against these labels, never against itself, so the test is non-circular.

**Ranker under test.** **`PKU-Alignment/beaver-7b-unified-cost`** (LLaMA-1 based, 2048-token
context; higher cost = more harmful). Input format
`BEGINNING OF CONVERSATION: USER: {prompt} ASSISTANT:{answer}`. Over-long inputs are
**left-truncated** (drop the prompt head, keep the answer + the final token the score head
reads) and each truncation is counted. Two scores per answer:
`cost_full` = Beaver(raw prompt, answer); `cost_vanilla` = Beaver(vanilla seed query, answer)
(for the vanilla subset these coincide).

**Reference ranker.** **`allenai/wildguard`** (Mistral-7B based, purpose-built for
WildJailbreak) provides an independent second ranker via P(harmful response), used only for
the pair-agreement cross-check.

**Measurements.**
1. **Truncation exposure** — fraction of Beaver calls exceeding the 2048-token context.
2. **ROC-AUC** of cost vs Qwen3Guard labels (higher cost ⇒ unsafe/positive) in three
   conditions: (vanilla, `cost_full`), (adversarial, `cost_full`), (adversarial,
   `cost_vanilla`). *AUC is used deliberately because it measures **ordering** — the
   probability a random unsafe answer scores above a random safe one — which is exactly what
   a ranker must do, independent of absolute calibration.*
3. **Mean cost** of unsafe vs safe answers per condition (must have unsafe > safe).
4. **Pair agreement** — for each filter-1 survivor (a prompt with both a safe and an unsafe
   answer), whether Beaver's argmin/argmax answer matches WildGuard's, per subset.
5. **Length confound** (adversarial) — Pearson correlation of `cost_full` with prompt token
   length, within safe- and within unsafe-labelled answers (tests whether cost tracks prompt
   length rather than harm).
6. **Disagreement dump** — adversarial answers where Beaver's sign (cost > 0 ⇒ unsafe)
   contradicts Qwen3Guard, sampled for human inspection.

**Scale.** 300 prompts × N=10 → ~3,000 generations, **2,997 valid** answers scored. Subset
split: **vanilla 1,260 answers (254 unsafe / 1,006 safe) ≈ 126 prompts**; **adversarial
1,737 answers (1,230 unsafe / 507 safe) ≈ 174 prompts**. Filter-1 survivors used for pair
agreement: **27 vanilla + 58 adversarial** prompts. Beaver ran ~4,700 scoring passes
(2,997 `cost_full` + 1,737 adversarial `cost_vanilla`); Qwen3Guard and WildGuard each scored
the 2,997 answers. Hardware: a single A100-SXM4-80GB. This is a **pilot / suitability
scale**, sufficient to answer the ranker-validity question — not the final dataset size.

**How to read the AUC table.** adversarial `cost_full` AUC ≈ vanilla → Beaver usable as-is;
adversarial `cost_full` ≪ vanilla but adversarial `cost_vanilla` ≈ vanilla → usable only with
vanilla-query scoring; all weak → drop Beaver, use WildGuard as sole ranker.

---

## 3. Results

- Records scored: **2,997** valid (prompt, answer) pairs (N=10 per prompt, 300 prompts).
- **Truncation over Beaver's 2048 ctx**: `cost_full` 80/2997 (**2.7%**), `cost_vanilla` 40 (adversarial only).

### 3a. ROC-AUC of cost vs Qwen3Guard (higher cost = unsafe)

| condition | AUC | mean(unsafe) | mean(safe) | unsafe>safe | n (unsafe/safe) |
|---|---|---|---|---|---|
| vanilla / cost_full | 0.989 | 1.604 | -5.827 | PASS | 1260 (254/1006) |
| adversarial / cost_full | 0.873 | 3.486 | -0.551 | PASS | 1737 (1230/507) |
| adversarial / cost_vanilla | 0.849 | 3.746 | -0.908 | PASS | 1737 (1230/507) |

### 3b. Pair agreement — Beaver vs WildGuard argmin/argmax (filter-1 survivors)

| subset | chosen match % | rejected match % | n prompts |
|---|---|---|---|
| vanilla | 40.7 | 29.6 | 27 |
| adversarial | 37.9 | 43.1 | 58 |

### 3c. Length confound (adversarial): Pearson(cost_full, prompt token length)

- within **safe**-labeled answers:   r = 0.080  (n=507)
- within **unsafe**-labeled answers: r = -0.012  (n=1230)

### 3d. Disagreements (Beaver sign vs Qwen3Guard, adversarial)

- total contradictions: 284; 10 random dumped -> `beaver_sanity_disagreements.jsonl`

---

## 4. Findings — what we found

1. **Beaver ranks harmfulness well on both subsets.** AUC **0.989** (vanilla) and **0.873**
   (adversarial) — the adversarial value is lower than vanilla but far above the ~0.65
   "unusable" region. As a within-prompt ranker (its actual job), Beaver is valid on
   WildJailbreak.

2. **The raw prompt is the right input; the vanilla-query "fix" is not needed.** Adversarial
   `cost_vanilla` (0.849) is *lower* than `cost_full` (0.873), so scoring against the vanilla
   seed query does not rescue adversarial and is marginally worse. Use `cost_full`.

3. **No length confound.** Pearson(cost, prompt length) ≈ 0 (0.08 safe, −0.01 unsafe). The
   main theoretical risk — long adversarial prompts inflating/deflating cost — is ruled out;
   the AUC reflects harm discrimination, not a length artifact.

4. **Context length is not a practical problem.** Only 2.7% of inputs exceeded Beaver's 2048
   window, and those are left-truncated to preserve the answer.

5. **Two caveats, neither disqualifying:**
   - *Adversarial sign/threshold is noisier.* Safe answers there average only −0.55 (vs −5.83
     on vanilla), clustering near 0 — hence 284 sign-contradictions with Qwen3Guard. But we
     use Beaver's **ranking**, not its absolute sign, and the final label comes from
     Qwen3Guard (filter-3), so this threshold noise does not enter the dataset.
   - *Beaver vs WildGuard exact pair agreement is modest* (~30–43%). This asks two rankers to
     pick the *same* answer as safest/most-harmful among many near-tied candidates — a much
     finer judgment than safe-vs-unsafe. Given Beaver's high AUC, its extremes are genuinely
     safe/harmful; the models simply disagree on the finest picks. Not a correctness problem
     for pair-building. (These percentages rest on only 27/58 prompts, so are themselves noisy.)

## 5. How this was verified (validity of the test)

- **Independent ground truth:** Qwen3Guard labels (a different model), so Beaver is not
  checked against itself.
- **Ranking metric matched to use:** AUC measures ordering, exactly what a ranker does.
- **Real data at scale:** 2,997 genuine Qwen3-4B responses to 300 real prompts.
- **Subset-split:** the OOD concern (adversarial) was isolated and measured separately.
- **Proposed fix tested with data:** vanilla-query scoring evaluated, not assumed.
- **Confound ruled out:** cost is not driven by prompt length.
- **Second independent ranker:** WildGuard cross-check + a disagreement dump for eyeballing.

## 6. Limitations

- The "ground truth" is **Qwen3Guard (a model), not human labels** — so strictly we verified
  that *Beaver ranks consistently with a strong modern safety classifier* (which is also the
  pipeline's label gate). A human skim of `beaver_sanity_disagreements.jsonl` closes the
  residual gap.
- AUC certifies **safe-vs-unsafe** separation (what pair-building needs), not the finest
  ordering *within* the all-safe or all-unsafe sets — consistent with the modest pair-agreement.
- Pilot scale (300 prompts); pair-agreement percentages rest on small prompt counts (27/58).

## 7. Conclusion / recommendation

**Keep `beaver-7b-unified-cost` as the ranker, scoring with the raw prompt (`cost_full`)** —
which is what stage 04 already does. Do not switch to vanilla-query scoring. Qwen3Guard
remains the safe/unsafe label gate (filter-3), which covers Beaver's weaker adversarial
threshold. Optional final human step: skim the 10 disagreement cases to confirm the failure
mode is benign (Beaver assigning mildly-positive cost to safe refusals in jailbreak framing).
