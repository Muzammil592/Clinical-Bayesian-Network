"""Discrete factors (potentials) over finite-state variables."""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping

import numpy as np


class DiscreteFactor:
    """A non-negative function over the joint states of a set of discrete variables.

    `values` is an array with one axis per variable, in the order of `variables`.
    """

    def __init__(self, variables: Iterable[str], cardinalities: Iterable[int], values) -> None:
        self.variables: List[str] = list(variables)
        cards = [int(c) for c in cardinalities]
        if len(self.variables) != len(set(self.variables)):
            raise ValueError("Factor variables must be unique.")
        if len(self.variables) != len(cards):
            raise ValueError("variables and cardinalities must have equal length.")
        self.cardinalities: Dict[str, int] = dict(zip(self.variables, cards))
        self.values = np.asarray(values, dtype=float).reshape(cards)

    # ------------------------------------------------------------------ helpers
    def _aligned(self, order: List[str]) -> np.ndarray:
        """Return values broadcastable against a full variable `order`."""
        present = [v for v in order if v in self.cardinalities]
        permutation = [self.variables.index(v) for v in present]
        array = np.transpose(self.values, permutation)
        shape = [self.cardinalities.get(v, 1) for v in order]
        return array.reshape(shape)

    def copy(self) -> "DiscreteFactor":
        return DiscreteFactor(self.variables, [self.cardinalities[v] for v in self.variables], self.values.copy())

    # ------------------------------------------------------------------ operations
    def product(self, other: "DiscreteFactor") -> "DiscreteFactor":
        order = list(self.variables) + [v for v in other.variables if v not in self.cardinalities]
        for v in other.variables:
            if v in self.cardinalities and self.cardinalities[v] != other.cardinalities[v]:
                raise ValueError(f"Cardinality mismatch for variable '{v}'.")
        cards = [self.cardinalities.get(v, other.cardinalities.get(v)) for v in order]
        return DiscreteFactor(order, cards, self._aligned(order) * other._aligned(order))

    def marginalize(self, variables: Iterable[str]) -> "DiscreteFactor":
        """Sum out the given variables."""
        drop = set(variables)
        unknown = drop - set(self.variables)
        if unknown:
            raise KeyError(f"Variables not in factor: {sorted(unknown)}")
        axes = tuple(i for i, v in enumerate(self.variables) if v in drop)
        keep = [v for v in self.variables if v not in drop]
        return DiscreteFactor(keep, [self.cardinalities[v] for v in keep], self.values.sum(axis=axes))

    def reduce(self, evidence: Mapping[str, int]) -> "DiscreteFactor":
        """Fix variables to observed state indices and drop them from the scope."""
        index = tuple(evidence[v] if v in evidence else slice(None) for v in self.variables)
        keep = [v for v in self.variables if v not in evidence]
        return DiscreteFactor(keep, [self.cardinalities[v] for v in keep], self.values[index])

    def normalize(self) -> "DiscreteFactor":
        total = self.values.sum()
        if total <= 0:
            raise ValueError("Cannot normalise a factor whose values sum to zero.")
        return DiscreteFactor(self.variables, [self.cardinalities[v] for v in self.variables], self.values / total)

    def __repr__(self) -> str:
        return f"DiscreteFactor(variables={self.variables}, shape={self.values.shape})"
