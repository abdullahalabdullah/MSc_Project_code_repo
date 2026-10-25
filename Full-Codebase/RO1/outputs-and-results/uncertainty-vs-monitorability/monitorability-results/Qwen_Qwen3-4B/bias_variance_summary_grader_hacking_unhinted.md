# Bias-variance decomposition & error-incoherence — Qwen/Qwen3-4B, grader_hacking, unhinted arm (N = 30 runs/question)

Decomposition and error-incoherence as in Hägele et al. (2026), *The Hot Mess of AI* (arXiv:2601.23045; code adapted from github.com/haeggee/hot-mess-of-ai, MIT). Incoherence = Σ variance / Σ error over questions (paper Eq. 2); KL is the paper's headline metric. KL magnitudes scale with −log(1e-12) per wrong run (one-hot samples), so read the KL ratio, not the KL values. `0/1 incoherence` = Σ variance / Σ (bias + variance) with the paper's 0/1 terms.

| scope | questions | error rate (0/1) | questions with ≥1 error | modal answer wrong | KL error-incoherence | Brier error-incoherence | 0/1 incoherence | Brier bias | Brier variance |
|---|---|---|---|---|---|---|---|---|---|
| lexam | 200 | 0.426 | 145 | 0.405 | **0.427** | 0.290 | 0.312 | 0.605 | 0.247 |
| lexam — g-mean² questions only | 142 | 0.457 | 111 | 0.430 | **0.465** | 0.315 | 0.333 | 0.627 | 0.288 |
