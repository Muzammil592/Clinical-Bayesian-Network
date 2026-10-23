"""Score-based structure learning by hill climbing with a tabu list.

The search starts from an empty graph and repeatedly applies the single best edge
addition, deletion or reversal. With ``patience > 0`` it may keep moving through a
bounded number of non-improving steps (a tabu list forbids immediately undoing a move),
which lets it escape the local optima that trap purely greedy search. The best graph
seen is returned. Optional random restarts perturb that graph and search again.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from src.graph import DirectedGraph
from src.scoring import DecomposableScore

Move = Tuple[float, str, str, str]  # (score change, operation, u, v)


@dataclass
class SearchResult:
    graph: DirectedGraph
    score: float
    trace: List[Dict] = field(default_factory=list)
    improving_restarts: int = 0

    @property
    def structure(self) -> Dict[str, List[str]]:
        return {v: self.graph.parents(v) for v in self.graph.nodes}


class HillClimbSearch:
    """Greedy search over DAGs for a decomposable score."""

    def __init__(
        self,
        score: DecomposableScore,
        max_indegree: Optional[int] = None,
        max_iterations: int = 1_000,
        restarts: int = 0,
        perturbation_steps: int = 4,
        tabu_length: int = 10,
        patience: int = 10,
        seed: int = 0,
        epsilon: float = 1e-9,
    ) -> None:
        self.score = score
        self.max_indegree = max_indegree if max_indegree is not None else len(score.dataset.variables)
        self.max_iterations = max_iterations
        self.restarts = restarts
        self.perturbation_steps = perturbation_steps
        self.tabu_length = tabu_length
        self.patience = patience
        self.seed = seed
        self.epsilon = epsilon

    # ------------------------------------------------------------------ moves
    def _moves(self, graph: DirectedGraph) -> Iterator[Move]:
        local = self.score.local
        for u in graph.nodes:
            for v in graph.nodes:
                if u == v:
                    continue
                parents_v = graph.parents(v)
                if graph.has_edge(u, v):
                    without = [p for p in parents_v if p != u]
                    yield local(v, without) - local(v, parents_v), "delete", u, v
                    parents_u = graph.parents(u)
                    if len(parents_u) < self.max_indegree:
                        trial = graph.copy()
                        trial.remove_edge(u, v)
                        if not trial.would_create_cycle(v, u):
                            delta = (
                                local(v, without) + local(u, parents_u + [v])
                                - local(v, parents_v) - local(u, parents_u)
                            )
                            yield delta, "reverse", u, v
                elif len(parents_v) < self.max_indegree and not graph.would_create_cycle(u, v):
                    yield local(v, parents_v + [u]) - local(v, parents_v), "add", u, v

    @staticmethod
    def _apply(graph: DirectedGraph, operation: str, u: str, v: str) -> None:
        if operation == "add":
            graph.add_edge(u, v)
        elif operation == "delete":
            graph.remove_edge(u, v)
        else:
            graph.remove_edge(u, v)
            graph.add_edge(v, u)

    # ------------------------------------------------------------------ search
    @staticmethod
    def _inverse(operation: str, u: str, v: str) -> Tuple[str, str, str]:
        """The move that would undo (operation, u, v); it is forbidden for a while afterwards."""
        if operation == "add":
            return "delete", u, v
        if operation == "delete":
            return "add", u, v
        return "reverse", v, u

    def _climb(self, graph: DirectedGraph, record: bool = False) -> Tuple[DirectedGraph, List[Dict]]:
        current = self._total(graph)
        best_graph, best_score, best_step = graph.copy(), current, 0
        tabu: deque = deque(maxlen=max(self.tabu_length, 1))
        stall = 0
        trace: List[Dict] = []
        for step in range(1, self.max_iterations + 1):
            candidates = [m for m in self._moves(graph) if m[1:] not in tabu]
            best = max(candidates, key=lambda m: m[0], default=None)
            if best is None or stall > self.patience:
                break
            delta, operation, u, v = best
            if delta <= self.epsilon and self.patience == 0:
                break
            self._apply(graph, operation, u, v)
            current += delta
            tabu.append(self._inverse(operation, u, v))
            if record:
                trace.append({"step": step, "operation": operation, "edge": f"{u} -> {v}", "score_change": delta})
            if current > best_score + self.epsilon:
                best_graph, best_score, best_step, stall = graph.copy(), current, step, 0
            else:
                stall += 1
        return best_graph, trace[:best_step]

    def _perturb(self, graph: DirectedGraph, rng: np.random.Generator) -> DirectedGraph:
        for _ in range(self.perturbation_steps):
            moves = list(self._moves(graph))
            if not moves:
                break
            _, operation, u, v = moves[int(rng.integers(len(moves)))]
            self._apply(graph, operation, u, v)
        return graph

    def _total(self, graph: DirectedGraph) -> float:
        return self.score.total({v: graph.parents(v) for v in graph.nodes})

    def fit(self, start: Optional[DirectedGraph] = None) -> SearchResult:
        variables = self.score.dataset.variables
        graph = start.copy() if start is not None else DirectedGraph.from_edges(variables)
        best_graph, trace = self._climb(graph, record=True)
        best_score = self._total(best_graph)
        rng = np.random.default_rng(self.seed)
        improvements = 0
        for _ in range(self.restarts):
            candidate, _ = self._climb(self._perturb(best_graph.copy(), rng))
            candidate_score = self._total(candidate)
            if candidate_score > best_score + self.epsilon:
                best_graph, best_score = candidate, candidate_score
                improvements += 1
        return SearchResult(best_graph, best_score, trace, improvements)
