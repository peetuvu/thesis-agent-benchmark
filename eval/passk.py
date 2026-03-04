from __future__ import annotations

import math
from typing import Optional


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Standard pass@k estimator used in code-generation evaluation.

    n: number of generated samples
    c: number of correct samples (0..n)
    k: how many samples you would "try" (1..n)

    Returns probability at least one of k samples is correct when sampling without replacement:
        pass@k = 1 - (C(n-c, k) / C(n, k))
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not (0 <= c <= n):
        raise ValueError("c must be in [0,n]")
    if not (1 <= k <= n):
        raise ValueError("k must be in [1,n]")

    # If there are fewer than k incorrect samples, success is guaranteed.
    if (n - c) < k:
        return 1.0

    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))
