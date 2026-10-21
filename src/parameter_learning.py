"""Parameter estimation for discrete Bayesian networks with a known structure.

Two estimators are provided:

* ``MaximumLikelihoodEstimator`` - relative frequencies of the observed counts.
* ``BayesianEstimator`` - posterior-mean estimates under Dirichlet priors
  (``"bdeu"`` equivalent-sample-size prior, or ``"k2"`` uniform pseudo-counts).
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from src.bayesian_network import BayesianNetwork
from src.graph import DirectedGraph


class ParameterEstimator:
    """Base class: validates data, collects sufficient statistics and assembles the network."""

    def __init__(
        self,
        structure: Mapping[str, Sequence[str]],
        states: Mapping[str, Sequence[str]],
        name: str = "learned_network",
    ) -> None:
        self.name = name
        self.structure: Dict[str, List[str]] = {v: list(p) for v, p in structure.items()}
        self.states: Dict[str, List[str]] = {v: list(s) for v, s in states.items()}
        if set(self.structure) != set(self.states):
            raise ValueError("structure and states must describe the same variables.")
        graph = DirectedGraph()
        for var, parents in self.structure.items():
            graph.add_node(var)
            for parent in parents:
                graph.add_edge(parent, var)
        self._order = graph.topological_order()

    @classmethod
    def from_network(cls, network: BayesianNetwork, **kwargs):
        """Reuse the structure and state spaces of an existing network (parameters are ignored)."""
        return cls(network.structure, network.states, **kwargs)

    # ------------------------------------------------------------------ statistics
    def _encode(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        codes: Dict[str, np.ndarray] = {}
        for var, states in self.states.items():
            if var not in data.columns:
                raise KeyError(f"Column '{var}' is missing from the data.")
            column = data[var].map({s: i for i, s in enumerate(states)})
            if column.isna().any():
                raise ValueError(f"Column '{var}' contains states outside {states}.")
            codes[var] = column.to_numpy(dtype=int)
        return codes

    def count_table(self, data: pd.DataFrame, variable: str) -> np.ndarray:
        """Co-occurrence counts with shape (parent cards..., variable card)."""
        codes = self._encode(data)
        parents = self.structure[variable]
        shape = tuple(len(self.states[p]) for p in parents) + (len(self.states[variable]),)
        index = tuple(codes[p] for p in parents) + (codes[variable],)
        flat = np.ravel_multi_index(index, shape)
        return np.bincount(flat, minlength=int(np.prod(shape))).reshape(shape).astype(float)

    # ------------------------------------------------------------------ fitting
    def _estimate(self, counts: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit(self, data: pd.DataFrame) -> BayesianNetwork:
        """Estimate every conditional probability table and return a validated network."""
        network = BayesianNetwork(self.name)
        for var in self._order:
            table = self._estimate(self.count_table(data, var))
            network.add_variable(var, self.states[var], self.structure[var], table)
        network.validate()
        return network


class MaximumLikelihoodEstimator(ParameterEstimator):
    """theta(x | pa) = N(x, pa) / N(pa).

    Parent configurations never observed in the data receive a uniform distribution.
    """

    def _estimate(self, counts: np.ndarray) -> np.ndarray:
        totals = counts.sum(axis=-1, keepdims=True)
        uniform = np.full_like(counts, 1.0 / counts.shape[-1])
        return np.divide(counts, totals, out=uniform, where=totals > 0)


class BayesianEstimator(ParameterEstimator):
    """Posterior mean of each CPT row under a Dirichlet prior.

    theta(x | pa) = (N(x, pa) + alpha) / (N(pa) + K * alpha)

    prior="bdeu": alpha = equivalent_sample_size / (number of parent configurations * K)
    prior="k2":   alpha = pseudo_count for every cell
    """

    def __init__(
        self,
        structure: Mapping[str, Sequence[str]],
        states: Mapping[str, Sequence[str]],
        prior: str = "bdeu",
        equivalent_sample_size: float = 10.0,
        pseudo_count: float = 1.0,
        name: str = "learned_network",
    ) -> None:
        super().__init__(structure, states, name)
        if prior not in {"bdeu", "k2"}:
            raise ValueError("prior must be 'bdeu' or 'k2'.")
        if equivalent_sample_size <= 0 or pseudo_count <= 0:
            raise ValueError("Prior strengths must be positive.")
        self.prior = prior
        self.equivalent_sample_size = float(equivalent_sample_size)
        self.pseudo_count = float(pseudo_count)

    def _alpha(self, counts: np.ndarray) -> float:
        if self.prior == "k2":
            return self.pseudo_count
        return self.equivalent_sample_size / counts.size

    def _estimate(self, counts: np.ndarray) -> np.ndarray:
        smoothed = counts + self._alpha(counts)
        return smoothed / smoothed.sum(axis=-1, keepdims=True)

    def dirichlet_posterior(self, data: pd.DataFrame, variable: str) -> np.ndarray:
        """Posterior Dirichlet concentration parameters for a variable's CPT."""
        counts = self.count_table(data, variable)
        return counts + self._alpha(counts)
