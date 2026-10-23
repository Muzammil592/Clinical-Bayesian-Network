"""Self-contained verification of graph, factor and network implementations.

Usage:
    python -m src.sanity_checks
"""
from __future__ import annotations

import itertools

import numpy as np

from src.clinical_network import build_clinical_network
from src.evaluation import exact_joint_kl, network_kl_divergence
from src.factor import DiscreteFactor
from src.graph import DirectedGraph
from src.graph_metrics import cpdag, cpdag_shd, markov_equivalent, structure_comparison
from src.scoring import DecomposableScore, DiscreteDataset
from src.structure_learning import HillClimbSearch
from src.inference import HEURISTICS, VariableElimination, ZeroProbabilityEvidenceError
from src.parameter_learning import BayesianEstimator, MaximumLikelihoodEstimator
from utilities.logging_utils import get_logger

logger = get_logger("sanity_checks")


def check_factor_operations() -> None:
    a = DiscreteFactor(["A", "B"], [2, 2], [[0.1, 0.9], [0.4, 0.6]])
    b = DiscreteFactor(["B", "C"], [2, 3], [[0.2, 0.3, 0.5], [0.6, 0.3, 0.1]])
    product = a.product(b)
    assert product.variables == ["A", "B", "C"]
    assert np.isclose(product.values[0, 1, 2], 0.9 * 0.1)
    assert np.allclose(product.marginalize(["C"]).values, a.values * b.values.sum(axis=1))
    assert np.allclose(a.reduce({"A": 1}).values, [0.4, 0.6])
    logger.info("Factor operations: OK")


def check_network(network) -> None:
    network.validate()
    total = sum(
        network.joint_probability(dict(zip(network.variables, states)))
        for states in itertools.product(*(network.states[v] for v in network.variables))
    )
    assert np.isclose(total, 1.0), total
    assert np.isclose(network.joint_factor().values.sum(), 1.0)
    logger.info("Joint distribution sums to one: OK")


def check_independencies(network) -> None:
    g = network.graph
    expectations = [
        (("Smoking", "Pollution", []), True),
        (("Smoking", "Pollution", ["LungDisease"]), False),
        (("Smoking", "Pollution", ["Cough"]), False),
        (("Age", "LungDisease", ["Smoking"]), True),
        (("Age", "LungDisease", []), False),
        (("Cough", "XRay", ["LungDisease"]), True),
        (("Cough", "Dyspnea", ["LungDisease"]), False),
        (("Cough", "Dyspnea", ["LungDisease", "Bronchitis"]), True),
    ]
    for (x, y, z), expected in expectations:
        assert g.d_separated(x, y, z) == expected, (x, y, z)
    assert g.markov_blanket("LungDisease") == {"Smoking", "Pollution", "Cough", "Dyspnea", "XRay", "Bronchitis"}
    logger.info("d-separation and Markov blanket queries: OK")


def check_sampling(network) -> None:
    data = network.sample(100_000, seed=0)
    for var in network.variables:
        exact = network.marginal(var)
        empirical = data[var].value_counts(normalize=True).reindex(exact.index).fillna(0.0)
        assert np.abs(empirical - exact).max() < 0.01, var
    logger.info("Sampled marginals match exact marginals: OK")
    exact_posterior = network.marginal("LungDisease", {"Cough": "yes", "XRay": "abnormal"})
    assert 0.0 < exact_posterior["present"] < 1.0
    logger.info("P(LungDisease=present | Cough=yes, XRay=abnormal) = %.4f", exact_posterior["present"])


def check_parameter_learning(network) -> None:
    data = network.sample(200_000, seed=1)
    mle = MaximumLikelihoodEstimator.from_network(network)
    bayes = BayesianEstimator.from_network(network, prior="bdeu", equivalent_sample_size=10.0)

    counts = mle.count_table(data, "Cough")
    assert counts.shape == (2, 2, 2) and counts.sum() == len(data)

    learned_mle, learned_bayes = mle.fit(data), bayes.fit(data)
    assert network_kl_divergence(network, learned_mle) < 1e-3
    assert network_kl_divergence(network, learned_bayes) < 1e-3
    assert np.isclose(network_kl_divergence(network, network), 0.0)
    logger.info("Estimators recover reference parameters from large samples: OK")

    small = network.sample(60, seed=2)
    small_mle, small_bayes = mle.fit(small), bayes.fit(small)
    assert (small_bayes.cpt("LungDisease") > 0).all()
    assert small_mle.log_likelihood(small) >= small_bayes.log_likelihood(small) - 1e-9
    kl_chain = network_kl_divergence(network, small_bayes)
    assert np.isclose(kl_chain, exact_joint_kl(network, small_bayes))
    logger.info("Chain-rule KL matches brute-force joint KL (%.4f): OK", kl_chain)

    skewed = small[small["XRay"] == "normal"]
    assert (mle.fit(skewed).cpt("XRay") == 0).any() and (bayes.fit(skewed).cpt("XRay") > 0).all()
    logger.info("Smoothing removes zero probabilities: OK")


def check_inference(network) -> None:
    engine = VariableElimination(network)
    cases = [
        ("LungDisease", {"Cough": "yes", "XRay": "abnormal"}),
        ("Bronchitis", {"Cough": "yes", "Dyspnea": "yes", "XRay": "abnormal"}),
        ("Smoking", {"LungDisease": "present", "Pollution": "high"}),
        ("Age", {"Dyspnea": "yes"}),
        ("XRay", {}),
    ]
    for variable, evidence in cases:
        assert np.allclose(engine.posterior(variable, evidence), network.marginal(variable, evidence)), (variable, evidence)
    logger.info("Variable elimination matches enumeration: OK")

    evidence = {"Cough": "yes", "XRay": "abnormal"}
    reference = network.marginal("LungDisease", evidence)
    for heuristic in HEURISTICS:
        for prune in (True, False):
            posterior = VariableElimination(network, heuristic, prune).posterior("LungDisease", evidence)
            assert np.allclose(posterior, reference), (heuristic, prune)
    custom = engine.query("LungDisease", evidence, order=["Bronchitis", "Smoking", "Pollution", "Age"])
    assert np.allclose(custom.values, reference.values)
    logger.info("Heuristics, pruning and custom orders agree: OK")

    expected = network.joint_factor().reduce(
        {v: network.state_index(v, s) for v, s in evidence.items()}
    ).values.sum()
    assert np.isclose(engine.evidence_probability(evidence), expected)
    assert np.isclose(engine.evidence_probability({}), 1.0)
    logger.info("Evidence probability: OK")

    joint = engine.joint_posterior(["LungDisease", "Bronchitis"], evidence)
    assert np.isclose(joint.sum(), 1.0)
    assert np.allclose(joint.groupby(level="LungDisease").sum(), reference)
    assignment, _ = engine.map_assignment(["LungDisease", "Bronchitis"], evidence)
    assert assignment == {"LungDisease": "present", "Bronchitis": "absent"}
    logger.info("Joint posterior and MAP assignment: OK")

    # Graph independence statements must hold numerically (d-separation <-> inference).
    assert network.graph.d_separated("Age", "LungDisease", ["Smoking"])
    assert np.allclose(
        engine.posterior("Age", {"Smoking": "yes", "LungDisease": "present"}),
        engine.posterior("Age", {"Smoking": "yes"}),
    )
    assert not np.allclose(
        engine.posterior("Pollution", {"LungDisease": "present"}),
        engine.posterior("Pollution", {"LungDisease": "present", "Smoking": "yes"}),
    )
    logger.info("Inference respects d-separation: OK")

    impossible = MaximumLikelihoodEstimator.from_network(network).fit(network.sample(60, seed=2).query("XRay == 'normal'"))
    try:
        VariableElimination(impossible).posterior("LungDisease", {"XRay": "abnormal"})
    except ZeroProbabilityEvidenceError:
        logger.info("Impossible evidence is rejected: OK")
    else:
        raise AssertionError("Zero-probability evidence was not detected.")


def check_graph_editing() -> None:
    graph = DirectedGraph.from_edges(["A", "B", "C"], [("A", "B"), ("B", "C")])
    clone = graph.copy()
    clone.remove_edge("A", "B")
    assert graph.has_edge("A", "B") and not clone.has_edge("A", "B")
    assert graph.would_create_cycle("C", "A") and not graph.would_create_cycle("A", "C")
    logger.info("Graph editing and cycle checks: OK")


def check_equivalence_classes(network) -> None:
    chain = DirectedGraph.from_edges("ABC", [("A", "B"), ("B", "C")])
    reverse = DirectedGraph.from_edges("ABC", [("C", "B"), ("B", "A")])
    fork = DirectedGraph.from_edges("ABC", [("B", "A"), ("B", "C")])
    collider = DirectedGraph.from_edges("ABC", [("A", "B"), ("C", "B")])
    assert markov_equivalent(chain, reverse) and markov_equivalent(chain, fork)
    assert not markov_equivalent(chain, collider)
    assert cpdag_shd(chain, reverse) == 0 and cpdag_shd(chain, collider) == 2
    directed, undirected = cpdag(network.graph)
    assert ("LungDisease", "XRay") in directed and ("Pollution", "LungDisease") in directed
    assert undirected == {frozenset({"Age", "Smoking"}), frozenset({"Bronchitis", "Smoking"})}
    assert cpdag_shd(network.graph, network.graph) == 0
    logger.info("CPDAG construction and Markov equivalence: OK")


def check_scores(network) -> None:
    data = network.sample(20_000, seed=3)
    dataset = DiscreteDataset(data, network.states)
    empty = {v: [] for v in network.variables}
    for kind in ("bic", "k2", "bdeu"):
        score = DecomposableScore(dataset, kind)
        assert score.total(network.structure) > score.total(empty)
        forward = dict(empty, Smoking=["Age"])
        backward = dict(empty, Age=["Smoking"])
        if kind != "k2":  # K2 is not score-equivalent; BIC and BDeu are
            assert np.isclose(score.total(forward), score.total(backward)), kind
    assert dataset.family_counts("Cough", ["LungDisease", "Bronchitis"]).sum() == len(data)
    logger.info("Scores prefer the true structure and BIC/BDeu are score-equivalent: OK")


def check_structure_learning(network) -> None:
    data = network.sample(50_000, seed=5)
    score = DecomposableScore(DiscreteDataset(data, network.states), "bic")
    result = HillClimbSearch(score, max_indegree=3, restarts=10, perturbation_steps=10, seed=0).fit()
    assert result.score >= score.total({v: [] for v in network.variables})
    assert result.score >= score.total(network.structure) - 1e-6
    comparison = structure_comparison(network.graph, result.graph)
    assert comparison["skeleton_f1"] == 1.0, comparison
    assert max(len(result.graph.parents(v)) for v in result.graph.nodes) <= 3
    result.graph.topological_order()  # raises if the learned graph is cyclic
    logger.info("Hill climbing recovers the true skeleton from 50k samples (CPDAG SHD %d): OK", comparison["cpdag_shd"])


def main() -> None:
    network = build_clinical_network()
    logger.info("Parameters in model: %d", network.number_of_parameters())
    check_factor_operations()
    check_network(network)
    check_independencies(network)
    check_sampling(network)
    check_parameter_learning(network)
    check_inference(network)
    check_graph_editing()
    check_equivalence_classes(network)
    check_scores(network)
    check_structure_learning(network)
    logger.info("All checks passed.")


if __name__ == "__main__":
    main()
