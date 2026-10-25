# Answer-parse report — Qwen/Qwen3-4B, grader_hacking (N = 20 runs/question/arm)

`unparsed` = runs whose final answer letter could not be parsed. Valid = parsed and not truncated; only valid runs enter the analysis. The last column counts questions with fewer than 18 valid runs in that arm (for the unhinted arm these are dropped by step 59).

| dataset | arm | runs | valid | unparsed | unparsed % | unparsed: truncated | unparsed: not truncated | truncated but parsed | questions | questions < 18 valid |
|---|---|---|---|---|---|---|---|---|---|---|
| lexam_priorA | unhinted | 1960 | 1957 | 0 | 0.0 | 0 | 0 | 3 | 98 | 0 |
| lexam_priorA | hinted | 1960 | 1960 | 0 | 0.0 | 0 | 0 | 0 | 98 | 0 |
| lexam_priorB | unhinted | 1960 | 1957 | 0 | 0.0 | 0 | 0 | 3 | 98 | 0 |
| lexam_priorB | hinted | 1960 | 1957 | 0 | 0.0 | 0 | 0 | 3 | 98 | 0 |
| ALL | unhinted | 3920 | 3914 | 0 | 0.0 | 0 | 0 | 6 | 196 | 0 |
| ALL | hinted | 3920 | 3917 | 0 | 0.0 | 0 | 0 | 3 | 196 | 0 |
