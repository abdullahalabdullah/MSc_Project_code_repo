# Base vs DPO+SK-Llama | Majority@16 | qwen3-1.7B

| Metric | Base | DPO+RM | Change |
|---|---|---|---|
| Valid responses orig % | 99.8 | 98.5 | -1.3 |
| Valid responses CF % | 99.5 | 98.1 | -1.5 |
| Acc orig % | 94.1 | 93.3 | -0.8 |
| Acc CF % | 85.0 | 83.9 | -1.2 |
| Acc diff % | 9.1 | 9.4 | +0.4 |
| Ack orig % | 27.2 | 23.2 | -3.9 |
| Ack CF % | 0.0 | 0.4 | +0.4 |
| Ack diff % | 27.2 | 22.8 | -4.3 |
| **Deceptive % (Maj@16)** | **2.0** | **2.8** | **+0.8** |

n (test examples): 254

## Paper reference (LLaMA-3.1-8B-IT, SK-LLAMA-8B, Maj@16, Math Book)

| | Deceptive % |
|---|---|
| Base | 27.2 ± 1.5 |
| DPO+RM | 34.0 ± 0.7 |
| Δ | +6.8 |
