# Unhinted → hinted median length change (`length_by_arm_3models.png`)

% change = (hinted median − unhinted median) / unhinted median × 100. Hinted = 3 hint types pooled (4800 runs); unhinted = shared arm (1600 runs). Test split, truncated runs included. Average = unweighted mean of the 8 per-dataset % changes.

| Dataset | Base | Harmlessness DPO | Anti-sycophancy DPO |
|---|---:|---:|---:|
| mmlu_pro | +48.7% | +55.8% | +59.9% |
| gpqa_diamond | +8.7% | +8.8% | +8.2% |
| lsat_ar | +33.1% | +37.7% | +43.0% |
| lsat_lr | +22.4% | +49.1% | +77.5% |
| medxpertqa | +2.5% | −0.4% | +1.8% |
| lexam | +15.7% | +12.7% | +14.7% |
| codemmlu | +49.7% | +58.1% | +57.4% |
| wmdp | +12.4% | +21.2% | +26.2% |
| **Average (8 datasets)** | **+24.2%** | **+30.4%** | **+36.1%** |



- Adding a hint has consistently increased the average respons elength across all datastes and all models.
- One observation is that the change in response length after adding the hint for each dataset is relatively similar across all 3 models.
