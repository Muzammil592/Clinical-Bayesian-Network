"""Study how structure recovery improves with training-set size.

For each sample size and score, several random training subsets are drawn, a structure is
learned, and it is compared with the reference graph.

Usage:
    python -m src.structure_study
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.clinical_network import build_clinical_network
from src.graph_metrics import structure_comparison
from src.learn_structure import build_search
from src.scoring import DecomposableScore, DiscreteDataset
from utilities.config import (
    DATA_DIR,
    STRUCTURE_STUDY_FILENAME,
    DatasetConfig,
    LearningConfig,
    StructureConfig,
)
from utilities.data_utils import split_train_test
from utilities.io_utils import load_dataframe, save_dataframe
from utilities.logging_utils import get_logger

logger = get_logger("structure_study")


def main() -> None:
    cfg, lcfg = StructureConfig(), LearningConfig()
    reference = build_clinical_network()
    train, _ = split_train_test(load_dataframe(DatasetConfig().path), lcfg.test_fraction, lcfg.split_seed)
    rng = np.random.default_rng(cfg.search_seed)

    records = []
    for n in cfg.study_sample_sizes:
        n = min(n, len(train))
        for _ in range(cfg.study_repeats):
            subset = train.iloc[rng.choice(len(train), size=n, replace=False)]
            dataset = DiscreteDataset(subset, reference.states)
            for kind in cfg.study_scores:
                score = DecomposableScore(dataset, kind, cfg.equivalent_sample_size)
                graph = build_search(score, cfg).fit().graph
                records.append({"sample_size": n, "score": kind, **structure_comparison(reference.graph, graph)})
        logger.info("Finished sample size %d", n)

    raw = pd.DataFrame(records)
    summary = (
        raw.groupby(["sample_size", "score"])
        .agg(
            median_cpdag_shd=("cpdag_shd", "median"),
            mean_skeleton_f1=("skeleton_f1", "mean"),
            mean_extra_edges=("extra_edges", "mean"),
            mean_missing_edges=("missing_edges", "mean"),
            share_markov_equivalent=("markov_equivalent", "mean"),
        )
        .reset_index()
    )
    save_dataframe(summary, DATA_DIR / STRUCTURE_STUDY_FILENAME)
    for column, fmt in (
        ("median_cpdag_shd", "{:.1f}"),
        ("mean_skeleton_f1", "{:.3f}"),
        ("mean_missing_edges", "{:.2f}"),
        ("mean_extra_edges", "{:.2f}"),
        ("share_markov_equivalent", "{:.2f}"),
    ):
        print(f"\n{column}")
        print(summary.pivot(index="sample_size", columns="score", values=column).to_string(float_format=fmt.format))


if __name__ == "__main__":
    main()
