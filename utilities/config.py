"""Central project configuration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class DatasetConfig:
    """Settings used when drawing observational data from the reference network."""

    n_samples: int = 20_000
    random_seed: int = 42
    filename: str = "clinical_observations.csv"

    @property
    def path(self) -> Path:
        return DATA_DIR / self.filename


MUTUAL_INFORMATION_FILENAME = "pairwise_mutual_information.csv"
