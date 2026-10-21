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


@dataclass(frozen=True)
class LearningConfig:
    """Settings for parameter estimation experiments."""

    test_fraction: float = 0.2
    split_seed: int = 7
    equivalent_sample_size: float = 10.0
    k2_pseudo_count: float = 1.0
    sample_sizes: tuple = (50, 100, 200, 500, 1_000, 2_000, 5_000, 16_000)
    n_repeats: int = 20
    likelihood_floor: float = 1e-12


MUTUAL_INFORMATION_FILENAME = "pairwise_mutual_information.csv"
PARAMETER_COMPARISON_FILENAME = "learned_parameter_comparison.csv"
SAMPLE_SIZE_STUDY_FILENAME = "sample_size_study.csv"
