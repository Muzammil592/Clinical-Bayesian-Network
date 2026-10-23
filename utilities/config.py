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


@dataclass(frozen=True)
class InferenceConfig:
    """Settings for diagnostic inference experiments."""

    target: str = "LungDisease"
    positive_state: str = "present"
    observed: tuple = ("Age", "Pollution", "Smoking", "Cough", "Dyspnea", "XRay")
    training_sizes: tuple = (100, 1_000, 16_000)
    heuristic: str = "min_fill"


@dataclass(frozen=True)
class StructureConfig:
    """Settings for score-based structure learning."""

    score: str = "bic"
    equivalent_sample_size: float = 10.0
    max_indegree: int = 3
    restarts: int = 20
    perturbation_steps: int = 10
    tabu_length: int = 10
    patience: int = 10
    search_seed: int = 11
    compared_scores: tuple = ("bic", "k2", "bdeu")
    study_scores: tuple = ("bic", "bdeu")
    study_sample_sizes: tuple = (200, 500, 1_000, 2_000, 5_000, 16_000)
    study_repeats: int = 10


MUTUAL_INFORMATION_FILENAME = "pairwise_mutual_information.csv"
PARAMETER_COMPARISON_FILENAME = "learned_parameter_comparison.csv"
SAMPLE_SIZE_STUDY_FILENAME = "sample_size_study.csv"
INFERENCE_QUERIES_FILENAME = "inference_queries.csv"
DIAGNOSTIC_EVALUATION_FILENAME = "diagnostic_evaluation.csv"
LEARNED_EDGES_FILENAME = "learned_structure_edges.csv"
STRUCTURE_STUDY_FILENAME = "structure_study.csv"
PIPELINE_SUMMARY_FILENAME = "pipeline_summary.csv"
