"""Metrics for comparing learned and reference network structures.

Score-based learning can only identify a structure up to Markov equivalence (DAGs with the
same skeleton and v-structures). Comparisons are therefore made both on the raw DAGs and on
their equivalence-class representatives (CPDAGs).
"""
from __future__ import annotations

import itertools
from typing import Dict, FrozenSet, Set, Tuple

import pandas as pd

from src.graph import DirectedGraph

Edge = Tuple[str, str]


def edge_set(graph: DirectedGraph) -> Set[Edge]:
    return set(graph.edges)


def skeleton(graph: DirectedGraph) -> Set[FrozenSet[str]]:
    return {frozenset(e) for e in graph.edges}


def v_structures(graph: DirectedGraph) -> Set[Tuple[str, str, str]]:
    """Unshielded colliders (a, c, b): a -> c <- b with a, b non-adjacent and a < b."""
    adjacent = skeleton(graph)
    found = set()
    for child in graph.nodes:
        for a, b in itertools.combinations(sorted(graph.parents(child)), 2):
            if frozenset((a, b)) not in adjacent:
                found.add((a, child, b))
    return found


def markov_equivalent(first: DirectedGraph, second: DirectedGraph) -> bool:
    return skeleton(first) == skeleton(second) and v_structures(first) == v_structures(second)


def cpdag(graph: DirectedGraph) -> Tuple[Set[Edge], Set[FrozenSet[str]]]:
    """Completed partially directed graph of a DAG: (directed edges, undirected edges).

    Edges in v-structures are oriented first; Meek's rules 1-3 then propagate orientations.
    """
    nodes = graph.nodes
    directed: Set[Edge] = set()
    undirected: Set[FrozenSet[str]] = skeleton(graph)
    adjacency = skeleton(graph)

    def adjacent(a: str, b: str) -> bool:
        return frozenset((a, b)) in adjacency

    def orient(u: str, v: str) -> None:
        undirected.discard(frozenset((u, v)))
        directed.add((u, v))

    for a, c, b in v_structures(graph):
        orient(a, c)
        orient(b, c)

    changed = True
    while changed:
        changed = False
        for u, v in itertools.permutations(nodes, 2):
            if frozenset((u, v)) not in undirected:
                continue
            # Rule 1: w -> u, u - v, w not adjacent to v  =>  u -> v
            rule1 = any((w, u) in directed and not adjacent(w, v) for w in nodes if w not in (u, v))
            # Rule 2: u -> w -> v and u - v  =>  u -> v
            rule2 = any((u, w) in directed and (w, v) in directed for w in nodes if w not in (u, v))
            # Rule 3: u - c -> v, u - d -> v, c and d not adjacent, u - v  =>  u -> v
            rule3 = any(
                frozenset((u, c)) in undirected and frozenset((u, d)) in undirected
                and (c, v) in directed and (d, v) in directed and not adjacent(c, d)
                for c, d in itertools.combinations([w for w in nodes if w not in (u, v)], 2)
            )
            if rule1 or rule2 or rule3:
                orient(u, v)
                changed = True
    return directed, undirected


def _pair_states(graph: DirectedGraph) -> Dict[FrozenSet[str], str]:
    directed, undirected = cpdag(graph)
    states = {pair: "undirected" for pair in undirected}
    for u, v in directed:
        states[frozenset((u, v))] = f"{u}->{v}"
    return states


def cpdag_shd(first: DirectedGraph, second: DirectedGraph) -> int:
    """Structural Hamming distance between equivalence classes (one point per differing node pair)."""
    a, b = _pair_states(first), _pair_states(second)
    return sum(a.get(pair) != b.get(pair) for pair in set(a) | set(b))


def structure_comparison(reference: DirectedGraph, learned: DirectedGraph) -> Dict[str, float]:
    """Edge-level and equivalence-class-level agreement between two DAGs."""
    ref_edges, new_edges = edge_set(reference), edge_set(learned)
    ref_skeleton, new_skeleton = skeleton(reference), skeleton(learned)
    correct = ref_edges & new_edges
    reversed_ = {(u, v) for u, v in new_edges if (v, u) in ref_edges}
    extra = {(u, v) for u, v in new_edges if frozenset((u, v)) not in ref_skeleton}
    missing = {(u, v) for u, v in ref_edges if frozenset((u, v)) not in new_skeleton}
    shared = len(ref_skeleton & new_skeleton)
    precision = shared / len(new_skeleton) if new_skeleton else 1.0
    recall = shared / len(ref_skeleton) if ref_skeleton else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "correct_edges": len(correct),
        "reversed_edges": len(reversed_),
        "extra_edges": len(extra),
        "missing_edges": len(missing),
        "directed_shd": len(reversed_) + len(extra) + len(missing),
        "cpdag_shd": cpdag_shd(reference, learned),
        "skeleton_precision": precision,
        "skeleton_recall": recall,
        "skeleton_f1": f1,
        "markov_equivalent": markov_equivalent(reference, learned),
    }


def edge_comparison_table(reference: DirectedGraph, learned: DirectedGraph) -> pd.DataFrame:
    """One row per edge of either graph, labelled correct / reversed / extra / missing."""
    ref_edges, new_edges = edge_set(reference), edge_set(learned)
    directed, _ = cpdag(reference)
    rows = []
    for u, v in sorted(new_edges):
        if (u, v) in ref_edges:
            status = "correct"
        elif (v, u) in ref_edges:
            status = "reversed"
        else:
            status = "extra"
        compelled = (u, v) in directed or (v, u) in directed
        rows.append({"edge": f"{u} -> {v}", "status": status, "orientation_identifiable": compelled if status != "extra" else None})
    learned_skeleton = skeleton(learned)
    for u, v in sorted(ref_edges):
        if frozenset((u, v)) not in learned_skeleton:
            rows.append({"edge": f"{u} -> {v}", "status": "missing", "orientation_identifiable": (u, v) in directed})
    return pd.DataFrame(rows)
