"""File input/output helpers."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_directory(path: Path) -> Path:
    """Create a directory (and parents) if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_dataframe(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    ensure_directory(path.parent)
    df.to_csv(path, index=index)
    return path


def load_dataframe(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Generate it first with: python -m src.generate_dataset"
        )
    return pd.read_csv(path, **kwargs)
