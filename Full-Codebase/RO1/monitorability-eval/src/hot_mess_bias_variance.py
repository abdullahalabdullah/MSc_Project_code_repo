"""
Per-question bias-variance decomposition of MCQ errors over test-time sampling,
and the error-incoherence metric, from "The Hot Mess of AI" — used by step 60
(experiment: uncertainty-vs-monitorability).

SOURCE / ACKNOWLEDGEMENT
------------------------
The functions below are ADAPTED from the official code release of

    A. Hägele, A. P. Gema, H. Sleight, E. Perez, J. Sohl-Dickstein (2026).
    "The Hot Mess of AI: How Does Misalignment Scale With Model Intelligence
    and Task Complexity?" arXiv:2601.23045.
    Code: https://github.com/haeggee/hot-mess-of-ai
    (commit 6165eaaaf55946ea996dce7a4d9b07639655a2cd, file scripts/utils.py)

Function-by-function provenance:
    answers_to_probs           <- scripts/utils.py, load_model_data(): the
                                  one-hot construction of per-sample "probs"
                                  for sampled CoT answers (uniform 1/K over the
                                  options when no answer was parsed).
    brier_score                <- scripts/utils.py, brier_score() (verbatim).
    compute_brier_bias_variance<- scripts/utils.py, compute_brier_bias_variance()
                                  (verbatim logic; use_one_hot branch kept).
    kl_divergence              <- scripts/utils.py, kl_divergence() (debug
                                  prints removed).
    compute_kl_bias_variance   <- scripts/utils.py, compute_kl_bias_variance()
                                  (same maths: Laplace smoothing with EPS =
                                  1e-12, normalised geometric-mean prediction,
                                  bias = -log fbar[gold], variance = E[KL] -
                                  bias, clamped at 0; debug prints removed).
    process_question_metrics   <- scripts/utils.py, process_question_metrics()
                                  (the 0/1, KL and Brier parts; jackknife and
                                  "soft"/"unbiased"/"biased" variants dropped;
                                  the hard-coded 4 options generalised to
                                  n_options).
    error_incoherence          <- the paper's Eq. 2, pooled over questions:
                                  sum_i Variance_i / sum_i Error_i.

The paper's headline MCQ metric is the KL-Error-incoherence (Sec. 2.1):
    E_eps[CE(y, f_eps)] = KL(y || fbar) + E_eps[KL(fbar || f_eps)]
    Bias = KL(y || fbar), Variance = E_eps[KL(fbar || f_eps)], Error = Bias + Variance
    fbar[c] ∝ exp(E_eps[log f_eps[c]])

The adapted code is used under the MIT License of the original repository:

    MIT License

    Copyright (c) 2025 Alexander Hägele

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from scipy import special

EPS = 1e-12  # as in scripts/utils.py


def answers_to_probs(answers: List[int], n_options: int) -> np.ndarray:
    """One "probability" vector per sampled answer (adapted from
    load_model_data): one-hot on the chosen option; an unparsed answer
    (index == n_options) becomes uniform over the options."""
    probs = np.zeros((len(answers), n_options))
    for k, a in enumerate(answers):
        if a < n_options:
            probs[k, a] = 1.0
        else:
            probs[k] = 1.0 / n_options
    return probs


def brier_score(probs: np.ndarray, gold_answer: int) -> float:
    """Brier score = sum((p_i - y_i)^2), y one-hot on the gold answer."""
    y = np.zeros_like(probs)
    y[gold_answer] = 1.0
    return np.sum((probs - y) ** 2)


def compute_brier_bias_variance(
    per_q_probs_array: np.ndarray, gold_answer: int, use_one_hot: bool = False
) -> Tuple[float, float]:
    """E[Brier] = bias + variance, bias = Brier(E[pred], y),
    variance = E[||pred - E[pred]||^2]."""
    if use_one_hot:
        one_hot_probs = np.zeros_like(per_q_probs_array)
        one_hot_probs[
            np.arange(len(per_q_probs_array)), np.argmax(per_q_probs_array, axis=1)
        ] = 1
        predictions = one_hot_probs
    else:
        predictions = per_q_probs_array

    brier_scores = [brier_score(pred, gold_answer) for pred in predictions]
    mean_prediction = np.mean(predictions, axis=0)
    brier_bias = brier_score(mean_prediction, gold_answer)
    brier_variance = np.mean(brier_scores) - brier_bias
    brier_variance = max(0.0, brier_variance)

    brier_variance_alt = np.mean(np.sum((predictions - mean_prediction) ** 2, axis=1))
    assert np.isclose(
        brier_variance, brier_variance_alt
    ), f"Brier variance calculations don't match: {brier_variance} != {brier_variance_alt}"
    assert brier_variance >= 0.0, f"Brier variance is negative: {brier_variance}"
    return brier_bias, brier_variance


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) in log space."""
    return np.sum(p * (np.log(p) - np.log(q)))


def compute_kl_bias_variance(
    per_q_probs_array: np.ndarray, gold_answer: int, laplace_smoothing: bool = True
) -> Tuple[float, float]:
    """E[KL(y || f)] = bias + variance, bias = -log fbar[gold] with fbar the
    normalised geometric mean of the predictions, variance = E[KL] - bias."""
    if laplace_smoothing:
        k = per_q_probs_array.shape[1]
        per_q_probs_array = per_q_probs_array + EPS
        per_q_probs_array = per_q_probs_array / (1 + k * EPS)
    kl_divergences = [-np.log(pred[gold_answer]) for pred in per_q_probs_array]

    log_probs = np.log(per_q_probs_array)
    mean_prediction = np.mean(log_probs, axis=0)
    mean_prediction = np.exp(mean_prediction - special.logsumexp(mean_prediction))

    kl_bias = -np.log(mean_prediction[gold_answer])
    kl_variance = np.mean(kl_divergences) - kl_bias

    kl_variance_alt = np.mean(
        [kl_divergence(mean_prediction, pred) for pred in per_q_probs_array]
    )
    assert np.isclose(
        kl_variance, kl_variance_alt, atol=1e-4
    ), f"KL variance calculations don't match: {kl_variance} != {kl_variance_alt}"

    kl_variance = max(0.0, kl_variance)  # clamp, as in the original
    return kl_bias, kl_variance


def process_question_metrics(
    answers: List[int], gold_answer: int, n_options: int
) -> Dict[str, float]:
    """Per-question 0/1, KL and Brier bias/variance (adapted subset of
    process_question_metrics). `answers` are option indices; n_options marks
    an unparsed answer."""
    probs = answers_to_probs(answers, n_options)
    counts = Counter(answers)
    K = len(answers)
    p_per_answer_soft = np.mean(probs, axis=0)

    consensus_answer = counts.most_common(1)[0][0]
    is_consensus_correct = consensus_answer == gold_answer

    # 0/1 loss L(y, y') = 1 - delta(y, y')
    b_i = 1.0 - is_consensus_correct
    error_i = 1 - p_per_answer_soft[gold_answer]
    if consensus_answer < n_options:
        v_i = 1 - p_per_answer_soft[consensus_answer]
        actual_variance = (
            (1 - p_per_answer_soft[consensus_answer])
            if is_consensus_correct
            else -p_per_answer_soft[gold_answer]
        )
    else:
        v_i = 0.0
        actual_variance = error_i - b_i
    assert np.isclose(error_i, b_i + actual_variance), "0/1 error decomposition is incorrect"

    kl_bias_i, kl_variance_i = compute_kl_bias_variance(probs, gold_answer)
    brier_bias_i, brier_variance_i = compute_brier_bias_variance(probs, gold_answer)

    total_error_01 = b_i + v_i
    total_error_kl = kl_bias_i + kl_variance_i
    total_error_brier = brier_bias_i + brier_variance_i
    has_error_kl = total_error_kl > 1e-10

    return {
        "n_samples": K,
        "accuracy": float(p_per_answer_soft[gold_answer]),
        "error_01": float(error_i),
        "is_consensus_correct": bool(is_consensus_correct),
        "bias_01": float(b_i),
        "variance_01": float(v_i),
        "variance_prop_01": float(v_i / total_error_01) if total_error_01 > 1e-10 else 0.0,
        "kl_bias": float(kl_bias_i),
        "kl_variance": float(kl_variance_i),
        "kl_error": float(total_error_kl),
        "has_error_kl": bool(has_error_kl),
        # per-question KL-incoherence; undefined (NaN) without error — the
        # original sets the proportion to 0.0 there, we keep it explicit
        "kl_incoherence": float(kl_variance_i / total_error_kl) if has_error_kl else float("nan"),
        "brier_bias": float(brier_bias_i),
        "brier_variance": float(brier_variance_i),
        "brier_error": float(total_error_brier),
        "brier_incoherence": (float(brier_variance_i / total_error_brier)
                              if total_error_brier > 1e-10 else float("nan")),
    }


def error_incoherence(variances: List[float], errors: List[float]) -> float:
    """Paper Eq. 2: sum_i Variance_i / sum_i Error_i over a set of questions."""
    tot = float(np.sum(errors))
    return float(np.sum(variances)) / tot if tot > 1e-10 else float("nan")
