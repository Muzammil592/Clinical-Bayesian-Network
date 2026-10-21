"""Study how estimation error shrinks with training-set size for each estimator.

For every sample size, several random training subsets are drawn; each estimator is
fitted and scored by KL divergence to the reference network and by held-out likelihood.

Usage:
    python -m src.sample_size_study
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.clinical_network import build_clinical_network
from src.evaluation import held_out_log_likelihood, network_kl_divergence, parent_weights
from src.parameter_learning import BayesianEstimator, MaximumLikelihoodEstimator
from utilities.config import DATA_DIR, SAMPLE_SIZE_STUDY_FILENAME, DatasetConfig, LearningConfig
from utilities.data_utils import split_train_test
from utilities.io_utils import load_dataframe, save_dataframe
from utilities.logging_utils import get_logger

logger = get_logger("sample_size_study")


def main() -> None:
    cfg = LearningConfig()
    reference = build_clinical_network()
    weights = parent_weights(reference)
    train, test = split_train_test(load_dataframe(DatasetConfig().path), cfg.test_fraction, cfg.split_seed)

    estimators = {
        "maximum_likelihood": MaximumLikelihoodEstimator.from_network(reference),
        "bayesian_bdeu": BayesianEstimator.from_network(
            reference, prior="bdeu", equivalent_sample_size=cfg.equivalent_sample_size
        ),
        "bayesian_k2": BayesianEstimator.from_network(reference, prior="k2", pseudo_count=cfg.k2_pseudo_count),
    }

    rng = np.random.default_rng(cfg.split_seed)
    records = []
    for n in cfg.sample_sizes:
        n = min(n, len(train))
        for repeat in range(cfg.n_repeats):
            subset = train.iloc[rng.choice(len(train), size=n, replace=False)]
            for name, estimator in estimators.items():
                learned = estimator.fit(subset)
                records.append(
                    {
                        "sample_size": n,
                        "method": name,
                        "kl": network_kl_divergence(reference, learned, weights),
                        "heldout_loglik": held_out_log_likelihood(learned, test, cfg.likelihood_floor),
                    }
                )
        logger.info("Finished sample size %d", n)

    raw = pd.DataFrame(records)
    summary = (
        raw.groupby(["sample_size", "method"])
        .agg(
            median_kl=("kl", "median"),
            share_infinite_kl=("kl", lambda s: float(np.isinf(s).mean())),
            mean_heldout_loglik=("heldout_loglik", "mean"),
        )
        .reset_index()
    )
    path = save_dataframe(summary, DATA_DIR / SAMPLE_SIZE_STUDY_FILENAME)

    print("\nMedian KL divergence to reference network")
    print(summary.pivot(index="sample_size", columns="method", values="median_kl").to_string(float_format=lambda x: f"{x:.5f}"))
    print("\nShare of fits with infinite KL (zero-probability events)")
    print(summary.pivot(index="sample_size", columns="method", values="share_infinite_kl").to_string(float_format=lambda x: f"{x:.2f}"))
    print("\nMean held-out log-likelihood per row")
    print(summary.pivot(index="sample_size", columns="method", values="mean_heldout_loglik").to_string(float_format=lambda x: f"{x:.4f}"))
    logger.info("Saved study results to %s", path)


if __name__ == "__main__":
    main()
