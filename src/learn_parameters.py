"""Fit conditional probability tables from data and compare with the reference network.

Usage:
    python -m src.learn_parameters [--variable NAME] [--ess FLOAT]
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.clinical_network import build_clinical_network
from src.evaluation import (
    compare_cpts,
    expected_kl_by_variable,
    held_out_log_likelihood,
    network_kl_divergence,
    zero_probability_rows,
)
from src.parameter_learning import BayesianEstimator, MaximumLikelihoodEstimator
from utilities.config import DATA_DIR, PARAMETER_COMPARISON_FILENAME, DatasetConfig, LearningConfig
from utilities.data_utils import split_train_test
from utilities.io_utils import load_dataframe, save_dataframe
from utilities.logging_utils import get_logger

logger = get_logger("learn_parameters")


def parse_args() -> argparse.Namespace:
    cfg = LearningConfig()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variable", default="LungDisease", help="Variable whose CPT is printed.")
    parser.add_argument("--ess", type=float, default=cfg.equivalent_sample_size, help="BDeu equivalent sample size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = LearningConfig()
    reference = build_clinical_network()
    data = load_dataframe(DatasetConfig().path)
    train, test = split_train_test(data, cfg.test_fraction, cfg.split_seed)
    logger.info("Train rows: %d | test rows: %d", len(train), len(test))

    estimators = {
        "maximum_likelihood": MaximumLikelihoodEstimator.from_network(reference),
        f"bayesian_bdeu_ess{args.ess:g}": BayesianEstimator.from_network(
            reference, prior="bdeu", equivalent_sample_size=args.ess
        ),
        "bayesian_k2": BayesianEstimator.from_network(reference, prior="k2", pseudo_count=cfg.k2_pseudo_count),
    }
    fitted = {name: est.fit(train) for name, est in estimators.items()}

    summary = pd.DataFrame(
        {
            name: {
                "kl_to_reference": network_kl_divergence(reference, net),
                "heldout_loglik_per_row": held_out_log_likelihood(net, test, cfg.likelihood_floor),
                "zero_probability_test_rows": zero_probability_rows(net, test),
            }
            for name, net in fitted.items()
        }
    ).T
    summary.loc["reference_network"] = [0.0, held_out_log_likelihood(reference, test, cfg.likelihood_floor), 0]
    print("\nModel comparison on held-out data")
    print(summary.to_string(float_format=lambda x: f"{x:.5f}"))

    per_variable = pd.DataFrame({name: expected_kl_by_variable(reference, net) for name, net in fitted.items()})
    print("\nExpected KL divergence per variable")
    print(per_variable.to_string(float_format=lambda x: f"{x:.5f}"))

    best = fitted[f"bayesian_bdeu_ess{args.ess:g}"]
    comparison = compare_cpts(reference, best, args.variable)
    print(f"\nReference vs learned CPT for {args.variable} (Bayesian, BDeu)")
    print(comparison.round(4).to_string(index=False))

    tables = pd.concat(
        [compare_cpts(reference, best, v) for v in reference.variables], ignore_index=True
    )
    path = save_dataframe(tables, DATA_DIR / PARAMETER_COMPARISON_FILENAME)
    logger.info("Saved full CPT comparison to %s", path)


if __name__ == "__main__":
    main()
