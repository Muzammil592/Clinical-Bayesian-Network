"""Dataset splitting helpers."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def split_train_test(
    data: pd.DataFrame, test_fraction: float = 0.2, seed: int = 0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Shuffle rows reproducibly and split into train and test sets."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must lie strictly between 0 and 1.")
    order = np.random.default_rng(seed).permutation(len(data))
    n_test = int(round(len(data) * test_fraction))
    shuffled = data.iloc[order].reset_index(drop=True)
    return shuffled.iloc[n_test:].reset_index(drop=True), shuffled.iloc[:n_test].reset_index(drop=True)
