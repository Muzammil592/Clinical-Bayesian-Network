"""Discrete Bayesian network: structure, conditional probability tables, joint and sampling."""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.factor import DiscreteFactor
from src.graph import DirectedGraph


class BayesianNetwork:
    """A discrete Bayesian network.

    Each variable stores a conditional probability table (CPT) with shape
    ``(card(parent_1), ..., card(parent_k), card(variable))`` whose last axis sums to one.
    Parent order is the order given in :meth:`add_variable`.
    """

    def __init__(self, name: str = "bayesian_network") -> None:
        self.name = name
        self.graph = DirectedGraph()
        self.states: Dict[str, List[str]] = {}
        self._tables: Dict[str, np.ndarray] = {}
        self._parent_order: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------ construction
    def add_variable(
        self,
        name: str,
        states: Sequence[str],
        parents: Sequence[str],
        table,
    ) -> None:
        if name in self.states:
            raise ValueError(f"Variable '{name}' already defined.")
        self.graph.add_node(name)
        for parent in parents:
            self.graph.add_edge(parent, name)
        self.states[name] = list(states)
        self._parent_order[name] = list(parents)
        self._tables[name] = np.asarray(table, dtype=float)

    def validate(self, atol: float = 1e-9) -> None:
        """Check table shapes, non-negativity and normalisation."""
        for var, table in self._tables.items():
            parents = self._parent_order[var]
            for p in parents:
                if p not in self.states:
                    raise ValueError(f"Parent '{p}' of '{var}' has no definition.")
            expected = tuple(len(self.states[p]) for p in parents) + (len(self.states[var]),)
            if table.shape != expected:
                raise ValueError(f"CPT for '{var}' has shape {table.shape}, expected {expected}.")
            if (table < 0).any():
                raise ValueError(f"CPT for '{var}' contains negative entries.")
            if not np.allclose(table.sum(axis=-1), 1.0, atol=atol):
                raise ValueError(f"CPT rows for '{var}' do not sum to one.")

    # ------------------------------------------------------------------ accessors
    @property
    def variables(self) -> List[str]:
        return self.graph.topological_order()

    def parents(self, variable: str) -> List[str]:
        return list(self._parent_order[variable])

    @property
    def structure(self) -> Dict[str, List[str]]:
        """Mapping from each variable to its ordered list of parents."""
        return {v: list(ps) for v, ps in self._parent_order.items()}

    def cardinality(self, variable: str) -> int:
        return len(self.states[variable])

    def cpt(self, variable: str) -> np.ndarray:
        return self._tables[variable].copy()

    def cpd_factor(self, variable: str) -> DiscreteFactor:
        scope = self._parent_order[variable] + [variable]
        return DiscreteFactor(scope, [self.cardinality(v) for v in scope], self._tables[variable])

    def state_index(self, variable: str, state: str) -> int:
        try:
            return self.states[variable].index(state)
        except ValueError:
            raise KeyError(f"'{state}' is not a state of '{variable}': {self.states[variable]}") from None

    def number_of_parameters(self) -> int:
        """Number of free parameters of the model."""
        return int(sum(t.size - t.size // t.shape[-1] for t in self._tables.values()))

    # ------------------------------------------------------------------ probability
    def joint_probability(self, assignment: Mapping[str, str]) -> float:
        """P(X_1 = x_1, ..., X_n = x_n) for a complete assignment of state names."""
        missing = set(self.states) - set(assignment)
        if missing:
            raise ValueError(f"Assignment is missing variables: {sorted(missing)}")
        prob = 1.0
        for var in self.states:
            idx = tuple(self.state_index(p, assignment[p]) for p in self._parent_order[var])
            idx += (self.state_index(var, assignment[var]),)
            prob *= self._tables[var][idx]
        return float(prob)

    def row_log_likelihood(self, data: pd.DataFrame, floor: Optional[float] = None) -> np.ndarray:
        """Log-probability of each row of a dataset of state names.

        If `floor` is given, probabilities are clipped from below at that value so that
        rows impossible under the model receive a large finite penalty instead of -inf.
        """
        codes = self._encode(data)
        total = np.zeros(len(data))
        for var in self.states:
            index = tuple(codes[p] for p in self._parent_order[var]) + (codes[var],)
            probs = self._tables[var][index]
            if floor is not None:
                probs = np.maximum(probs, floor)
            with np.errstate(divide="ignore"):
                total += np.log(probs)
        return total

    def log_likelihood(self, data: pd.DataFrame, floor: Optional[float] = None) -> float:
        """Total log-likelihood of a dataset of state names under the model."""
        return float(self.row_log_likelihood(data, floor).sum())

    def joint_factor(self) -> DiscreteFactor:
        """Full joint distribution as a single factor (exponential in size; small networks only)."""
        factor: Optional[DiscreteFactor] = None
        for var in self.variables:
            cpd = self.cpd_factor(var)
            factor = cpd if factor is None else factor.product(cpd)
        return factor

    def marginal_table(self, variables: Sequence[str]) -> np.ndarray:
        """Joint marginal over `variables` as an array with axes in the given order."""
        variables = list(variables)
        joint = self.joint_factor()
        others = [v for v in joint.variables if v not in variables]
        reduced = joint.marginalize(others) if others else joint
        return reduced.as_array(variables)

    def marginal(self, variable: str, evidence: Optional[Mapping[str, str]] = None) -> pd.Series:
        """Exact P(variable | evidence) by brute-force enumeration over the joint."""
        evidence = dict(evidence or {})
        joint = self.joint_factor()
        reduced = joint.reduce({v: self.state_index(v, s) for v, s in evidence.items()})
        others = [v for v in reduced.variables if v != variable]
        result = reduced.marginalize(others).normalize()
        return pd.Series(result.values, index=self.states[variable], name=variable)

    # ------------------------------------------------------------------ sampling
    def sample(self, n_samples: int, seed: Optional[int] = None) -> pd.DataFrame:
        """Draw i.i.d. samples by ancestral sampling; returns state names."""
        rng = np.random.default_rng(seed)
        codes: Dict[str, np.ndarray] = {}
        for var in self.variables:
            table = self._tables[var]
            parents = self._parent_order[var]
            if parents:
                probs = table[tuple(codes[p] for p in parents)]
            else:
                probs = np.broadcast_to(table, (n_samples, table.shape[-1]))
            draws = rng.random(n_samples)[:, None]
            codes[var] = np.minimum((draws > np.cumsum(probs, axis=1)).sum(axis=1), table.shape[-1] - 1)
        return pd.DataFrame(
            {var: np.asarray(self.states[var])[codes[var]] for var in self.variables}
        )

    # ------------------------------------------------------------------ internals
    def _encode(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        codes: Dict[str, np.ndarray] = {}
        for var in self.states:
            mapping = {s: i for i, s in enumerate(self.states[var])}
            column = data[var].map(mapping)
            if column.isna().any():
                raise ValueError(f"Column '{var}' contains states outside {self.states[var]}.")
            codes[var] = column.to_numpy(dtype=int)
        return codes

    def __repr__(self) -> str:
        return f"BayesianNetwork(name='{self.name}', variables={len(self.states)}, edges={len(self.graph.edges)})"
