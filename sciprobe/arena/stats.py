"""Paired, question-level statistics for A/B tests between retrieval arms.

The unit of analysis is the QUESTION. Multiple criteria, trials or judges per
question are correlated and must not be counted as independent samples.
"""
from __future__ import annotations

import math
import random
from math import comb


def binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial test p-value for k successes in n at p = 0.5."""
    if n == 0:
        return 1.0
    pk = comb(n, k)
    return min(1.0, sum(comb(n, i) for i in range(n + 1) if comb(n, i) <= pk) / 2**n)


def sign_test(wins: int, losses: int) -> float:
    """Sign test ignoring ties."""
    return binom_two_sided(wins, wins + losses)


def wilson_ci(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_mean_ci(diffs: list[float], n_boot: int = 5000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    if not diffs:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(n_boot))
    return (means[int(alpha / 2 * n_boot)], means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))])


def min_wins_for_significance(n: int, alpha: float = 0.05) -> int | None:
    for k in range(n // 2, n + 1):
        if binom_two_sided(k, n) < alpha:
            return k
    return None


def questions_needed(win_rate: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Normal-approximation sample size to detect a true win rate vs 0.5."""
    if not 0 < win_rate < 1 or win_rate == 0.5:
        raise ValueError("win_rate must be in (0, 1) and != 0.5")
    za = _z(1 - alpha / 2)
    zb = _z(power)
    return math.ceil(((za * 0.5 + zb * math.sqrt(win_rate * (1 - win_rate))) / abs(win_rate - 0.5)) ** 2)


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def _z(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation)."""
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02, 1.383577518672690e02,
         -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01,
         -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00, -2.549732539343734e00,
         4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    lo, hi = 0.02425, 1 - 0.02425
    if p < lo:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > hi:
        return -_z(1 - p)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
