"""Learn the network structure from data and compare it with the reference graph.

Usage:
    python -m src.learn_structure
"""
from __future__ import annotations

import pandas as pd

from src.clinical_network import build_clinical_network
from src.evaluation import held_out_log_likelihood
from src.graph_metrics import cpdag, edge_comparison_table, structure_comparison
from src.parameter_learning import BayesianEstimator
from src.scoring import DecomposableScore, DiscreteDataset
from src.structure_learning import HillClimbSearch
from utilities.config import (
    DATA_DIR,
    LEARNED_EDGES_FILENAME,
    DatasetConfig,
    LearningConfig,
    StructureConfig,
)
from utilities.data_utils import split_train_test
from utilities.io_utils import load_dataframe, save_dataframe
from utilities.logging_utils import get_logger

logger = get_logger("learn_structure")


def build_search(score: DecomposableScore, cfg: StructureConfig) -> HillClimbSearch:
    return HillClimbSearch(
        score,
        max_indegree=cfg.max_indegree,
        restarts=cfg.restarts,
        perturbation_steps=cfg.perturbation_steps,
        tabu_length=cfg.tabu_length,
        patience=cfg.patience,
        seed=cfg.search_seed,
    )


def main() -> None:
    cfg, lcfg = StructureConfig(), LearningConfig()
    reference = build_clinical_network()
    data = load_dataframe(DatasetConfig().path)
    train, test = split_train_test(data, lcfg.test_fraction, lcfg.split_seed)
    dataset = DiscreteDataset(train, reference.states)
    logger.info("Learning from %d training rows", len(train))

    directed, undirected = cpdag(reference.graph)
    print("\nReference equivalence class")
    print("  compelled:", ", ".join(f"{u}->{v}" for u, v in sorted(directed)))
    print("  reversible:", ", ".join(" - ".join(sorted(e)) for e in sorted(undirected, key=sorted)))

    rows, learned_graphs = [], {}
    for kind in cfg.compared_scores:
        score = DecomposableScore(dataset, kind, cfg.equivalent_sample_size)
        result = build_search(score, cfg).fit()
        learned_graphs[kind] = result.graph
        estimator = BayesianEstimator(
            result.structure, reference.states, "bdeu", cfg.equivalent_sample_size
        )
        network = estimator.fit(train)
        rows.append(
            {
                "score": kind,
                "learned_score": result.score,
                "reference_graph_score": score.total(reference.structure),
                "edges": len(result.graph.edges),
                **structure_comparison(reference.graph, result.graph),
                "heldout_loglik": held_out_log_likelihood(network, test, lcfg.likelihood_floor),
            }
        )
    reference_net = BayesianEstimator.from_network(
        reference, prior="bdeu", equivalent_sample_size=cfg.equivalent_sample_size
    ).fit(train)
    summary = pd.DataFrame(rows).set_index("score")
    print("\nStructure learning results")
    print(summary.T.to_string(float_format=lambda x: f"{x:.4f}"))
    logger.info(
        "Held-out log-likelihood with the reference structure: %.4f",
        held_out_log_likelihood(reference_net, test, lcfg.likelihood_floor),
    )

    primary = learned_graphs[cfg.score]
    table = edge_comparison_table(reference.graph, primary)
    print(f"\nEdge-level comparison ({cfg.score.upper()})")
    print(table.to_string(index=False))
    save_dataframe(table, DATA_DIR / LEARNED_EDGES_FILENAME)

    trace = build_search(DecomposableScore(dataset, cfg.score, cfg.equivalent_sample_size), cfg).fit().trace
    print(f"\nSearch trace ({cfg.score.upper()})")
    print(pd.DataFrame(trace).to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
