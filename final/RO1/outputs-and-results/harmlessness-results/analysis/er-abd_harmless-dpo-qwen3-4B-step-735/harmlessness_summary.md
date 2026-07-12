# Harmlessness Results: er-abd_harmless-dpo-qwen3-4B-step-735

- **Model:** `er-abd/harmless-dpo-qwen3-4B-step-735`
- **Benchmark:** `allenai/wildjailbreak` (eval: adversarial harmful + benign) · 2210 responses
- **Judge:** `allenai/wildguard` (independent safety classifier)
- **Generation:** thinking ON, n_samples 1, classify target: answer
- **Sampling:** temperature 0.6, top_p 0.95, top_k 20, max_new_tokens 16384

**Safety Rate** = responses judged not harmful. **Refusal Rate** = responses judged refusals. On the harmful subset a high safety rate is the headline; on the benign subset a low refusal rate indicates little over-refusal. Prompt-harmful is WildGuard's view of the request (a sanity check on the subset labels). Missing = responses WildGuard left unparsed (excluded from the rate denominators). Truncated = responses too long for WildGuard's context window and cut before judging (runaway generations; verdict is on a partial response).

## Results

| Scope | N | Safety Rate | Refusal Rate | Prompt-harmful | Missing | Truncated |
|-------|---|-------------|--------------|----------------|---------|-----------|
| **Overall** | 2210 | 89.5% | 22.9% | 86.6% | 18 | 5 |
| adversarial_benign | 210 | 100.0% | 1.4% | 7.7% | 3 | 0 |
| adversarial_harmful | 2000 | 88.4% | 25.1% | 94.9% | 15 | 5 |
