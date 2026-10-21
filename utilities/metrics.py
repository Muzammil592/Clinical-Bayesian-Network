"""Divergence measures between discrete distributions."""
from __future__ import annotations

import numpy as np


def kl_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """KL(p || q) in nats along the last axis.

    Terms with p = 0 contribute zero; terms with p > 0 and q = 0 make the result infinite.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log(p / q), 0.0)
    return terms.sum(axis=-1)


def weighted_sum(weights: np.ndarray, values: np.ndarray) -> float:
    """Sum of weights * values, treating 0 * inf as 0."""
    weights = np.asarray(weights, dtype=float)
    values = np.asarray(values, dtype=float)
    return float(np.where(weights > 0, weights * values, 0.0).sum())
