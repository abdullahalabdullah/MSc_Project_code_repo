# prior-belief-vs-monitorability — question selection

Selection runs: `outputs-and-results/uncertainty-vs-monitorability/inference-outputs/Qwen_Qwen3-4B/lexam/grader_hacking/unhinted.jsonl` (Qwen/Qwen3-4B, lexam, grader_hacking); valid run = parsed and not truncated.

| step | questions |
|---|---|
| questions | 200 |
| always correct | 55 |
| modal wrong answer < 6 runs | 28 |
| no never-chosen wrong option | 19 |
| selected | 98 |

- Version A target: modal wrong answer; p0 in the selection runs min 0.20 / median 0.62 / max 1.00; 9 questions at p0 = 1; 1 tie(s) broken at random.
- Version B target: never-chosen wrong option; drawn at random for 52 questions with several candidates.
- Original (step 01) hint target equals A for 33 questions and B for 40.
- RNG: numpy default_rng([selection_seed=88, instance_id]).
