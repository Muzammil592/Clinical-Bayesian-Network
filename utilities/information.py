"""Information-theoretic measures for discrete variables."""
from __future__ import annotations

import numpy as np
import pandas as pd


def entropy(probabilities: np.ndarray) -> float:
    """Shannon entropy in nats."""
    p = np.asarray(probabilities, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def mutual_information(df: pd.DataFrame, a: str, b: str) -> float:
    """Empirical mutual information I(a; b) in nats."""
    joint = pd.crosstab(df[a], df[b]).to_numpy(dtype=float).copy()
    joint = joint / joint.sum()
    pa = joint.sum(axis=1, keepdims=True)
    pb = joint.sum(axis=0, keepdims=True)
    mask = joint > 0
    return float((joint[mask] * np.log(joint[mask] / (pa @ pb)[mask])).sum())


def mutual_information_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    out = pd.DataFrame(0.0, index=cols, columns=cols)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            out.loc[a, b] = out.loc[b, a] = mutual_information(df, a, b)
    return out
