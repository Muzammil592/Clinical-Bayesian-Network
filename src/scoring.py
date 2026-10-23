"""Decomposable scores for ranking candidate Bayesian network structures.

All scores are higher-is-better and decompose as a sum over variables ("families"):

    score(G | D) = sum_i local_score(X_i, Parents_G(X_i) | D)

Supported scores:
    bic    log-likelihood minus 0.5 * log(N) * (number of free parameters)
    k2     Bayesian Dirichlet score with a uniform pseudo-count of one per cell
    bdeu   Bayesian Dirichlet score with an equivalent sample size spread evenly over cells
"""
from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

SCORES = ("bic", "k2", "bdeu")

_lgamma = np.vectorize(math.lgamma, otypes=[float])


class DiscreteDataset:
    """Integer-encoded view of a dataset that provides family count tables."""

    def __init__(self, data: pd.DataFrame, states: Mapping[str, Sequence[str]]) -> None:
        self.states: Dict[str, List[str]] = {v: list(s) for v, s in states.items()}
        self.variables: List[str] = list(self.states)
        self.cardinalities: Dict[str, int] = {v: len(s) for v, s in self.states.items()}
        self.n_samples = len(data)
        self._codes: Dict[str, np.ndarray] = {}
        for var, states_ in self.states.items():
            column = data[var].map({s: i for i, s in enumerate(states_)})
            if column.isna().any():
                raise ValueError(f"Column '{var}' contains states outside {states_}.")
            self._codes[var] = column.to_numpy(dtype=np.int64)

    def family_counts(self, variable: str, parents: Sequence[str]) -> np.ndarray:
        """Counts N_jk with shape (parent configurations, variable states)."""
        r = self.cardinalities[variable]
        if parents:
            shape = tuple(self.cardinalities[p] for p in parents)
            q = int(np.prod(shape))
            configuration = np.ravel_multi_index(tuple(self._codes[p] for p in parents), shape)
        else:
            q = 1
            configuration = np.zeros(self.n_samples, dtype=np.int64)
        flat = configuration * r + self._codes[variable]
        return np.bincount(flat, minlength=q * r).reshape(q, r).astype(float)


def log_likelihood_term(counts: np.ndarray) -> float:
    """Maximised log-likelihood of one family: sum N_jk log(N_jk / N_j)."""
    totals = counts.sum(axis=1, keepdims=True)
    mask = counts > 0
    return float((counts[mask] * np.log((counts / np.where(totals > 0, totals, 1.0))[mask])).sum())


def bic_term(counts: np.ndarray, n_samples: int) -> float:
    q, r = counts.shape
    return log_likelihood_term(counts) - 0.5 * math.log(n_samples) * q * (r - 1)


def k2_term(counts: np.ndarray) -> float:
    q, r = counts.shape
    totals = counts.sum(axis=1)
    return float(q * math.lgamma(r) - _lgamma(totals + r).sum() + _lgamma(counts + 1.0).sum())


def bdeu_term(counts: np.ndarray, equivalent_sample_size: float) -> float:
    q, r = counts.shape
    alpha_j = equivalent_sample_size / q
    alpha_jk = equivalent_sample_size / (q * r)
    totals = counts.sum(axis=1)
    return float(
        q * math.lgamma(alpha_j)
        - _lgamma(alpha_j + totals).sum()
        + _lgamma(alpha_jk + counts).sum()
        - counts.size * math.lgamma(alpha_jk)
    )


class DecomposableScore:
    """Cached local and total structure scores for a dataset."""

    def __init__(self, dataset: DiscreteDataset, kind: str = "bic", equivalent_sample_size: float = 10.0) -> None:
        if kind not in SCORES:
            raise ValueError(f"score must be one of {SCORES}.")
        if equivalent_sample_size <= 0:
            raise ValueError("equivalent_sample_size must be positive.")
        self.dataset = dataset
        self.kind = kind
        self.equivalent_sample_size = float(equivalent_sample_size)
        self._cache: Dict[Tuple[str, Tuple[str, ...]], float] = {}

    def local(self, variable: str, parents: Sequence[str]) -> float:
        key = (variable, tuple(sorted(parents)))
        if key not in self._cache:
            counts = self.dataset.family_counts(variable, key[1])
            if self.kind == "bic":
                value = bic_term(counts, self.dataset.n_samples)
            elif self.kind == "k2":
                value = k2_term(counts)
            else:
                value = bdeu_term(counts, self.equivalent_sample_size)
            self._cache[key] = value
        return self._cache[key]

    def total(self, structure: Mapping[str, Sequence[str]]) -> float:
        """Score of a full structure given as {variable: parents}."""
        return float(sum(self.local(v, parents) for v, parents in structure.items()))
