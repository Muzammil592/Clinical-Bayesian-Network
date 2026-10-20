"""Self-contained verification of graph, factor and network implementations.

Usage:
    python -m src.sanity_checks
"""
from __future__ import annotations

import itertools

import numpy as np

from src.clinical_network import build_clinical_network
from src.factor import DiscreteFactor
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


def main() -> None:
    network = build_clinical_network()
    logger.info("Parameters in model: %d", network.number_of_parameters())
    check_factor_operations()
    check_network(network)
    check_independencies(network)
    check_sampling(network)
    logger.info("All checks passed.")


if __name__ == "__main__":
    main()
