"""Divergence measures between discrete distributions."""
from __future__ import annotations

import numpy as np
import pandas as pd


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


def binary_log_loss(labels: np.ndarray, probabilities: np.ndarray, floor: float = 1e-12) -> float:
    """Mean negative log-likelihood of binary labels under predicted probabilities."""
    y = np.asarray(labels, dtype=float)
    p = np.clip(np.asarray(probabilities, dtype=float), floor, 1.0 - floor)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def brier_score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Mean squared error between binary labels and predicted probabilities."""
    return float(((np.asarray(probabilities, dtype=float) - np.asarray(labels, dtype=float)) ** 2).mean())


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve via the rank-sum statistic (ties receive average ranks)."""
    y = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(np.asarray(scores, dtype=float)).rank(method="average").to_numpy()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
