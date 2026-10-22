"""Exact inference in discrete Bayesian networks by variable elimination.

Features
--------
* sum-product elimination with greedy ordering heuristics (min-fill, min-neighbours, min-weight)
* pruning of barren nodes: variables that are neither queried, observed, nor ancestors of
  either cannot influence the answer and are dropped before any computation
* posterior marginals, joint posteriors, marginal MAP assignments and evidence probabilities
"""
from __future__ import annotations

import itertools
from functools import reduce
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

from src.bayesian_network import BayesianNetwork
from src.factor import DiscreteFactor

HEURISTICS = ("min_fill", "min_neighbors", "min_weight")

Targets = Union[str, Sequence[str]]


class ZeroProbabilityEvidenceError(ValueError):
    """Raised when the observed evidence has probability zero under the model."""


class VariableElimination:
    """Exact inference engine bound to a :class:`BayesianNetwork`."""

    def __init__(self, network: BayesianNetwork, heuristic: str = "min_fill", prune: bool = True) -> None:
        if heuristic not in HEURISTICS:
            raise ValueError(f"heuristic must be one of {HEURISTICS}.")
        network.validate()
        self.network = network
        self.heuristic = heuristic
        self.prune = prune

    # ------------------------------------------------------------------ validation
    def _targets(self, targets: Targets) -> List[str]:
        names = [targets] if isinstance(targets, str) else list(targets)
        if len(names) != len(set(names)):
            raise ValueError("Query variables must be unique.")
        for name in names:
            if name not in self.network.states:
                raise KeyError(f"Unknown variable '{name}'.")
        return names

    def _evidence(self, evidence: Optional[Mapping[str, str]]) -> Dict[str, int]:
        indices: Dict[str, int] = {}
        for var, state in (evidence or {}).items():
            if var not in self.network.states:
                raise KeyError(f"Unknown evidence variable '{var}'.")
            indices[var] = self.network.state_index(var, state)
        return indices

    # ------------------------------------------------------------------ planning
    def _relevant_variables(self, targets: Sequence[str], evidence: Mapping[str, int]) -> Set[str]:
        if not self.prune:
            return set(self.network.states)
        keep = set(targets) | set(evidence)
        for var in list(keep):
            keep |= self.network.graph.ancestors(var)
        return keep

    def _order_by_heuristic(self, scopes: List[Set[str]], eliminate: Set[str]) -> Tuple[List[str], int]:
        """Greedy elimination order and the size of the widest intermediate factor."""
        adjacency: Dict[str, Set[str]] = {}
        for scope in scopes:
            for var in scope:
                adjacency.setdefault(var, set()).update(scope - {var})
        remaining = [v for v in self.network.variables if v in eliminate]

        def cost(var: str) -> float:
            neighbours = adjacency[var]
            if self.heuristic == "min_neighbors":
                return len(neighbours)
            if self.heuristic == "min_weight":
                return float(np.prod([self.network.cardinality(n) for n in neighbours])) if neighbours else 1.0
            return sum(1 for a, b in itertools.combinations(neighbours, 2) if b not in adjacency[a])

        order: List[str] = []
        widest = max((len(s) for s in scopes), default=0)
        while remaining:
            chosen = min(remaining, key=lambda v: (cost(v), remaining.index(v)))
            neighbours = adjacency[chosen]
            widest = max(widest, len(neighbours) + 1)
            for a in neighbours:
                adjacency[a].discard(chosen)
                adjacency[a].update(neighbours - {a})
            del adjacency[chosen]
            remaining.remove(chosen)
            order.append(chosen)
        return order, widest

    def plan(self, targets: Targets, evidence: Optional[Mapping[str, str]] = None) -> Tuple[List[str], int]:
        """Elimination order chosen by the heuristic and the widest factor scope it produces."""
        target_list, evidence_idx = self._targets(targets), self._evidence(evidence)
        relevant = self._relevant_variables(target_list, evidence_idx)
        factors = self._initial_factors(relevant, evidence_idx)
        eliminate = relevant - set(target_list) - set(evidence_idx)
        return self._order_by_heuristic([set(f.variables) for f in factors], eliminate)

    # ------------------------------------------------------------------ core algorithm
    def _initial_factors(self, relevant: Set[str], evidence: Mapping[str, int]) -> List[DiscreteFactor]:
        return [
            self.network.cpd_factor(v).reduce(evidence) for v in self.network.variables if v in relevant
        ]

    def _run(
        self,
        targets: List[str],
        evidence: Dict[str, int],
        order: Optional[Sequence[str]] = None,
    ) -> DiscreteFactor:
        overlap = set(targets) & set(evidence)
        if overlap:
            raise ValueError(f"Variables cannot be both queried and observed: {sorted(overlap)}")
        relevant = self._relevant_variables(targets, evidence)
        factors = self._initial_factors(relevant, evidence)
        if not factors:
            return DiscreteFactor([], [], 1.0)
        eliminate = relevant - set(targets) - set(evidence)
        if order is None:
            order, _ = self._order_by_heuristic([set(f.variables) for f in factors], eliminate)
        elif set(order) != eliminate or len(order) != len(eliminate):
            raise ValueError(f"order must be a permutation of {sorted(eliminate)}.")

        for var in order:
            involved = [f for f in factors if var in f.variables]
            others = [f for f in factors if var not in f.variables]
            combined = reduce(lambda a, b: a.product(b), involved)
            factors = others + [combined.marginalize([var])]

        result = reduce(lambda a, b: a.product(b), factors)
        cards = [self.network.cardinality(t) for t in targets]
        return DiscreteFactor(targets, cards, result.as_array(targets))

    # ------------------------------------------------------------------ public queries
    def evidence_probability(self, evidence: Mapping[str, str]) -> float:
        """P(evidence)."""
        return float(self._run([], self._evidence(evidence)).values)

    def query(
        self,
        targets: Targets,
        evidence: Optional[Mapping[str, str]] = None,
        order: Optional[Sequence[str]] = None,
    ) -> DiscreteFactor:
        """Normalised P(targets | evidence) as a factor over the targets (in the given order)."""
        target_list = self._targets(targets)
        if not target_list:
            raise ValueError("At least one query variable is required.")
        factor = self._run(target_list, self._evidence(evidence), order)
        if factor.values.sum() <= 0:
            raise ZeroProbabilityEvidenceError(f"Evidence {dict(evidence or {})} has probability zero.")
        return factor.normalize()

    def posterior(self, variable: str, evidence: Optional[Mapping[str, str]] = None) -> pd.Series:
        """P(variable | evidence) indexed by state name."""
        factor = self.query(variable, evidence)
        return pd.Series(factor.values, index=self.network.states[variable], name=variable)

    def joint_posterior(self, targets: Targets, evidence: Optional[Mapping[str, str]] = None) -> pd.Series:
        """P(targets | evidence) over all joint target states."""
        target_list = self._targets(targets)
        factor = self.query(target_list, evidence)
        spaces = [self.network.states[t] for t in target_list]
        index = (
            pd.Index(spaces[0], name=target_list[0])
            if len(spaces) == 1
            else pd.MultiIndex.from_product(spaces, names=target_list)
        )
        return pd.Series(factor.values.ravel(), index=index, name="probability")

    def map_assignment(
        self, targets: Targets, evidence: Optional[Mapping[str, str]] = None
    ) -> Tuple[Dict[str, str], float]:
        """Most probable joint assignment of the targets given the evidence, and its probability."""
        target_list = self._targets(targets)
        factor = self.query(target_list, evidence)
        flat = int(np.argmax(factor.values))
        positions = np.unravel_index(flat, factor.values.shape)
        assignment = {t: self.network.states[t][int(i)] for t, i in zip(target_list, positions)}
        return assignment, float(factor.values.ravel()[flat])
