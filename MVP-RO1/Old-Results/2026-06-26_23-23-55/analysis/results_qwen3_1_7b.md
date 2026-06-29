# Results: Qwen/Qwen3-1.7B

**Hint type:** Stanford Professor  
**Datasets:** MMLU, GPQA, WMDP, AGIEval (70 examples each)  
**N = 4** samples per arm (hinted / unhinted)  
**Model:** `Qwen/Qwen3-1.7B`

---

## g-mean² Results

| Dataset | Mean TE (before filtering) | Mean R_effect (before filtering) | n with TE > 0 (post L1) | g-mean² | Unhinted Acc. Mean (±Std) |
| ------- | :------------------------: | :------------------------------: | :---------------------: | :-----: | :-----------------------: |
| mmlu    |           0.0786           |              0.6157              |       13 (18.6%)        | 0.4231  |     0.6702 (±0.4340)     |
| gpqa    |           0.2226           |              0.6905              |       36 (51.4%)        | 0.3843  |     0.3643 (±0.3819)     |
| wmdp    |           0.1924           |              0.5764              |       26 (38.2%)        | 0.5321  |     0.4559 (±0.4165)     |
| agieval |           0.0821           |              0.3438              |       18 (26.1%)        | 0.1235  |     0.5193 (±0.4214)     |

> **Note:** WMDP n=68, AGIEval n=69 (2 and 1 instances excluded due to truncation/no parsed answer).

---

## CoT Token Length

Mean number of output tokens in the chain-of-thought, averaged per instance then across instances.

| Condition                               |  mmlu  |  gpqa  |  wmdp  | agieval | Overall |
| --------------------------------------- | -----: | -----: | -----: | ------: | ------: |
| Unhinted (X=0, all)                     | 1012.5 | 5433.2 | 2564.4 |  3591.6 |  3153.1 |
| Hinted (X=1, all)                       | 1289.9 | 5257.4 | 2627.6 |  3697.7 |  3220.7 |
| Hinted — hint followed (X=1, Y=1)       | 1444.7 | 6403.6 | 2694.8 |  5467.4 |  4395.6 |
| Hinted — hint not followed (X=1, Y=0)  | 1293.4 | 5162.3 | 2617.3 |  3743.5 |  3182.1 |
| Overall (mean across conditions)        | 1260.1 | 5564.1 | 2626.0 |  4125.1 |  3487.9 |

---

## Truncation Summary

| Dataset | Total responses | Truncated | Rate  | Max tokens cap |
| ------- | :-------------: | :-------: | :---: | :------------: |
| mmlu    |       560       |     9     | 1.6%  |     16,384     |
| gpqa    |       560       |    11     | 2.0%  |     16,384     |
| wmdp    |       560       |    34     | 6.1%  |     16,384     |
| agieval |       560       |    17     | 3.0%  |     16,384     |
| **ALL** |    **2,240**    |  **71**   | **3.2%** | **16,384**  |

> High WMDP truncation rate (6.1%) reflects the model entering repetitive loops on certain hazardous-knowledge questions — the same instances truncate consistently across all 4 samples.
