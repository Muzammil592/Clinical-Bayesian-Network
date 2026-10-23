"""End-to-end workflow: data -> structure learning -> parameter learning -> inference -> evaluation.

Compares three models on held-out patients:
    reference      true structure and true parameters (upper bound)
    reference_structure   true structure, parameters learned from data
    learned_structure     structure and parameters both learned from data

Usage:
    python -m src.run_pipeline
"""
from __future__ import annotations

import pandas as pd

from src.clinical_network import build_clinical_network
from src.evaluation import exact_joint_kl, held_out_log_likelihood
from src.graph_metrics import structure_comparison
from src.inference import VariableElimination
from src.learn_structure import build_search
from src.parameter_learning import BayesianEstimator
from src.run_inference_queries import diagnostic_scores
from src.scoring import DecomposableScore, DiscreteDataset
from utilities.config import (
    DATA_DIR,
    PIPELINE_SUMMARY_FILENAME,
    DatasetConfig,
    InferenceConfig,
    LearningConfig,
    StructureConfig,
)
from utilities.data_utils import split_train_test
from utilities.io_utils import load_dataframe, save_dataframe
from utilities.logging_utils import get_logger
from utilities.metrics import binary_log_loss, roc_auc

logger = get_logger("run_pipeline")


def main() -> None:
    scfg, lcfg, icfg = StructureConfig(), LearningConfig(), InferenceConfig()
    reference = build_clinical_network()

    data_cfg = DatasetConfig()
    if data_cfg.path.exists():
        data = load_dataframe(data_cfg.path)
    else:
        data = reference.sample(data_cfg.n_samples, seed=data_cfg.random_seed)
        save_dataframe(data, data_cfg.path)
    train, test = split_train_test(data, lcfg.test_fraction, lcfg.split_seed)
    logger.info("Train rows: %d | test rows: %d", len(train), len(test))

    score = DecomposableScore(DiscreteDataset(train, reference.states), scfg.score, scfg.equivalent_sample_size)
    result = build_search(score, scfg).fit()
    structure_metrics = structure_comparison(reference.graph, result.graph)
    logger.info("Learned %d edges | skeleton F1 %.3f | CPDAG SHD %d",
                len(result.graph.edges), structure_metrics["skeleton_f1"], structure_metrics["cpdag_shd"])

    def fit(structure):
        return BayesianEstimator(structure, reference.states, "bdeu", scfg.equivalent_sample_size).fit(train)

    models = {
        "reference": reference,
        "reference_structure": fit(reference.structure),
        "learned_structure": fit(result.structure),
    }

    labels = (test[icfg.target] == icfg.positive_state).to_numpy()
    oracle = diagnostic_scores(VariableElimination(reference, icfg.heuristic), test, icfg)
    rows = []
    for name, model in models.items():
        scores = oracle if name == "reference" else diagnostic_scores(VariableElimination(model, icfg.heuristic), test, icfg)
        rows.append(
            {
                "model": name,
                "edges": len(model.graph.edges),
                "parameters": model.number_of_parameters(),
                "kl_to_reference": exact_joint_kl(reference, model),
                "heldout_loglik": held_out_log_likelihood(model, test, lcfg.likelihood_floor),
                "diagnosis_auc": roc_auc(labels, scores),
                "diagnosis_log_loss": binary_log_loss(labels, scores),
                "mean_abs_gap_to_reference": float((scores - oracle).abs().mean()),
            }
        )
    summary = pd.DataFrame(rows)
    save_dataframe(summary, DATA_DIR / PIPELINE_SUMMARY_FILENAME)

    print("\nStructure recovery")
    print(pd.Series(structure_metrics).to_string())
    print("\nModel comparison on held-out patients")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    engine = VariableElimination(models["learned_structure"])
    evidence = {"Cough": "yes", "XRay": "abnormal", "Smoking": "yes"}
    posterior = engine.posterior(icfg.target, evidence)[icfg.positive_state]
    exact = VariableElimination(reference).posterior(icfg.target, evidence)[icfg.positive_state]
    print(f"\nP({icfg.target}={icfg.positive_state} | {evidence}) learned: {posterior:.4f} | reference: {exact:.4f}")


if __name__ == "__main__":
    main()
