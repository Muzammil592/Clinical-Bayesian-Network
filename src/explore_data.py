"""Exploratory analysis of the observational dataset.

Compares empirical marginals with the exact marginals of the reference network and
computes pairwise mutual information between variables.

Usage:
    python -m src.explore_data
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.clinical_network import build_clinical_network
from utilities.config import DATA_DIR, MUTUAL_INFORMATION_FILENAME, DatasetConfig
from utilities.information import mutual_information_matrix
from utilities.io_utils import load_dataframe, save_dataframe
from utilities.logging_utils import get_logger

logger = get_logger("explore_data")


def main() -> None:
    data = load_dataframe(DatasetConfig().path)
    network = build_clinical_network()

    logger.info("Dataset shape: %s", data.shape)
    logger.info("Missing values: %d", int(data.isna().sum().sum()))

    rows = []
    for var in network.variables:
        empirical = data[var].value_counts(normalize=True)
        exact = network.marginal(var)
        for state in network.states[var]:
            rows.append(
                {
                    "variable": var,
                    "state": state,
                    "empirical": empirical.get(state, 0.0),
                    "exact": exact[state],
                }
            )
    comparison = pd.DataFrame(rows)
    comparison["abs_error"] = (comparison["empirical"] - comparison["exact"]).abs()
    print("\nMarginal distributions: empirical vs exact")
    print(comparison.round(4).to_string(index=False))

    mi = mutual_information_matrix(data[network.variables])
    save_dataframe(mi, DATA_DIR / MUTUAL_INFORMATION_FILENAME, index=True)

    upper = np.triu(np.ones(mi.shape, dtype=bool), k=1)
    pairs = mi.where(upper).stack().sort_values(ascending=False)

    print("\nTop 10 variable pairs by mutual information (nats)")
    print(pairs.head(10).round(4).to_string())


if __name__ == "__main__":
    main()
