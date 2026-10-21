"""Metrics for comparing learned networks against a reference network."""
from __future__ import annotations

import itertools
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd

from src.bayesian_network import BayesianNetwork
from utilities.metrics import kl_divergence, weighted_sum


def parent_weights(reference: BayesianNetwork) -> Dict[str, np.ndarray]:
    """P_reference(parents) for every variable, arranged to match the CPT parent axes."""
    return {v: reference.marginal_table(reference.parents(v)) for v in reference.variables}


def expected_kl_by_variable(
    reference: BayesianNetwork,
    learned: BayesianNetwork,
    weights: Optional[Mapping[str, np.ndarray]] = None,
) -> pd.Series:
    """E_{pa ~ P}[ KL(P(X | pa) || Q(X | pa)) ] for each variable X.

    For networks that share a structure these terms add up to KL(P || Q) (chain rule).
    """
    weights = weights or parent_weights(reference)
    result = {}
    for var in reference.variables:
        kl = kl_divergence(reference.cpt(var), learned.cpt(var))
        result[var] = weighted_sum(weights[var], kl)
    return pd.Series(result, name="expected_kl")


def network_kl_divergence(
    reference: BayesianNetwork,
    learned: BayesianNetwork,
    weights: Optional[Mapping[str, np.ndarray]] = None,
) -> float:
    """KL(P_reference || P_learned) computed from the factorised form."""
    return float(expected_kl_by_variable(reference, learned, weights).sum())


def exact_joint_kl(reference: BayesianNetwork, learned: BayesianNetwork) -> float:
    """KL(P || Q) by direct summation over the full joint (verification for small networks)."""
    p_factor = reference.joint_factor()
    order = p_factor.variables
    p = p_factor.values.ravel()
    q = learned.joint_factor().as_array(order).ravel()
    return float(kl_divergence(p, q))


def held_out_log_likelihood(network: BayesianNetwork, data: pd.DataFrame, floor: float = 1e-12) -> float:
    """Mean per-row log-likelihood (nats) with probabilities floored for finiteness."""
    return float(network.row_log_likelihood(data, floor).mean())


def zero_probability_rows(network: BayesianNetwork, data: pd.DataFrame) -> int:
    """Number of rows the model assigns probability exactly zero."""
    return int(np.isneginf(network.row_log_likelihood(data)).sum())


def compare_cpts(reference: BayesianNetwork, learned: BayesianNetwork, variable: str) -> pd.DataFrame:
    """Long-format table of reference vs learned conditional probabilities for one variable."""
    parents = reference.parents(variable)
    ref, est = reference.cpt(variable), learned.cpt(variable)
    rows = []
    parent_states = [reference.states[p] for p in parents]
    for config in itertools.product(*[range(len(s)) for s in parent_states]):
        label = ", ".join(f"{p}={parent_states[i][c]}" for i, (p, c) in enumerate(zip(parents, config))) or "(none)"
        for k, state in enumerate(reference.states[variable]):
            rows.append(
                {
                    "variable": variable,
                    "parent_configuration": label,
                    "state": state,
                    "reference": ref[config + (k,)],
                    "learned": est[config + (k,)],
                }
            )
    table = pd.DataFrame(rows)
    table["abs_error"] = (table["reference"] - table["learned"]).abs()
    return table
