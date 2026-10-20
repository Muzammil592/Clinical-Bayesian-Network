"""Directed acyclic graph with the structural queries needed for Bayesian networks."""
from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Set, Tuple, Union

NodeSet = Union[str, Iterable[str]]


def _as_set(nodes: NodeSet) -> Set[str]:
    return {nodes} if isinstance(nodes, str) else set(nodes)


class DirectedGraph:
    """A directed acyclic graph. Edges that would create a cycle are rejected."""

    def __init__(self) -> None:
        self._parents: Dict[str, List[str]] = {}
        self._children: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------ building
    def add_node(self, node: str) -> None:
        self._parents.setdefault(node, [])
        self._children.setdefault(node, [])

    def add_edge(self, parent: str, child: str) -> None:
        if parent == child:
            raise ValueError(f"Self-loop on '{parent}' is not allowed.")
        self.add_node(parent)
        self.add_node(child)
        if parent in self._parents[child]:
            return
        if parent in self.descendants(child):
            raise ValueError(f"Edge {parent} -> {child} would create a cycle.")
        self._parents[child].append(parent)
        self._children[parent].append(child)

    # ------------------------------------------------------------------ accessors
    @property
    def nodes(self) -> List[str]:
        return list(self._parents)

    @property
    def edges(self) -> List[Tuple[str, str]]:
        return [(p, c) for c, ps in self._parents.items() for p in ps]

    def parents(self, node: str) -> List[str]:
        return list(self._parents[node])

    def children(self, node: str) -> List[str]:
        return list(self._children[node])

    def _reach(self, node: str, neighbours: Dict[str, List[str]]) -> Set[str]:
        seen: Set[str] = set()
        queue = deque(neighbours[node])
        while queue:
            current = queue.popleft()
            if current not in seen:
                seen.add(current)
                queue.extend(neighbours[current])
        return seen

    def ancestors(self, node: str) -> Set[str]:
        return self._reach(node, self._parents)

    def descendants(self, node: str) -> Set[str]:
        return self._reach(node, self._children)

    def roots(self) -> List[str]:
        return [n for n in self.nodes if not self._parents[n]]

    def topological_order(self) -> List[str]:
        """Kahn's algorithm; ties broken by insertion order for reproducibility."""
        in_degree = {n: len(ps) for n, ps in self._parents.items()}
        queue = deque(n for n in self.nodes if in_degree[n] == 0)
        order: List[str] = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for child in self._children[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        if len(order) != len(self._parents):
            raise ValueError("Graph contains a cycle.")
        return order

    def markov_blanket(self, node: str) -> Set[str]:
        """Parents, children and the children's other parents."""
        blanket = set(self._parents[node]) | set(self._children[node])
        for child in self._children[node]:
            blanket |= set(self._parents[child])
        blanket.discard(node)
        return blanket

    # ------------------------------------------------------------------ independence
    def moral_graph(self, nodes: Iterable[str] = None) -> Dict[str, Set[str]]:
        """Undirected moral graph restricted to `nodes` (all nodes by default).

        `nodes` must be ancestrally closed so that every parent is present.
        """
        keep = set(self.nodes if nodes is None else nodes)
        adjacency: Dict[str, Set[str]] = {n: set() for n in keep}
        for child in keep:
            parents = [p for p in self._parents[child] if p in keep]
            for p in parents:
                adjacency[p].add(child)
                adjacency[child].add(p)
            for i, a in enumerate(parents):
                for b in parents[i + 1:]:
                    adjacency[a].add(b)
                    adjacency[b].add(a)
        return adjacency

    def d_separated(self, x: NodeSet, y: NodeSet, given: NodeSet = ()) -> bool:
        """Test whether x and y are d-separated by `given`.

        Uses the moralised ancestral graph criterion: restrict to the ancestors of
        x, y and given, moralise, delete the conditioning nodes, and check whether
        x and y are disconnected.
        """
        xs, ys, zs = _as_set(x), _as_set(y), _as_set(given)
        if xs & ys or xs & zs or ys & zs:
            raise ValueError("x, y and given must be pairwise disjoint.")
        relevant = xs | ys | zs
        for node in list(relevant):
            relevant |= self.ancestors(node)
        adjacency = self.moral_graph(relevant)
        for z in zs:
            for neighbour in adjacency[z]:
                adjacency[neighbour].discard(z)
            adjacency[z] = set()
        seen = set(xs)
        queue = deque(xs)
        while queue:
            current = queue.popleft()
            if current in ys:
                return False
            for neighbour in adjacency[current]:
                if neighbour not in seen and neighbour not in zs:
                    seen.add(neighbour)
                    queue.append(neighbour)
        return True
