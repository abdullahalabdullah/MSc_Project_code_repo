# Answer-parse report — Qwen/Qwen3-4B, grader_hacking (N = 30 runs/question/arm)

`unparsed` = runs whose final answer letter could not be parsed. Valid = parsed and not truncated; only valid runs enter the analysis. The last column counts questions with fewer than 27 valid runs in that arm (for the unhinted arm these are dropped by step 59).

| dataset | arm | runs | valid | unparsed | unparsed % | unparsed: truncated | unparsed: not truncated | truncated but parsed | questions | questions < 27 valid |
|---|---|---|---|---|---|---|---|---|---|---|
| lexam | unhinted | 6000 | 5989 | 0 | 0.0 | 0 | 0 | 11 | 200 | 0 |
| lexam | hinted | 6000 | 5993 | 0 | 0.0 | 0 | 0 | 7 | 200 | 0 |
| ALL | unhinted | 6000 | 5989 | 0 | 0.0 | 0 | 0 | 11 | 200 | 0 |
| ALL | hinted | 6000 | 5993 | 0 | 0.0 | 0 | 0 | 7 | 200 | 0 |
