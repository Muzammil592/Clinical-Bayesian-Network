"""Draw observational data from the reference clinical network and store it as CSV.

Usage:
    python -m src.generate_dataset [--n-samples N] [--seed S] [--output PATH]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.clinical_network import build_clinical_network
from utilities.config import DatasetConfig
from utilities.io_utils import save_dataframe
from utilities.logging_utils import get_logger

logger = get_logger("generate_dataset")


def parse_args() -> argparse.Namespace:
    defaults = DatasetConfig()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-samples", type=int, default=defaults.n_samples)
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument("--output", type=Path, default=defaults.path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    network = build_clinical_network()
    logger.info("Built %s", network)
    data = network.sample(args.n_samples, seed=args.seed)
    path = save_dataframe(data, args.output)
    logger.info("Saved %d rows x %d columns to %s", *data.shape, path)


if __name__ == "__main__":
    main()
