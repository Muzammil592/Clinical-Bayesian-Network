"""Diagnostic reasoning with exact inference.

Sections
--------
1. Clinical posterior queries, verified against brute-force enumeration
2. Explaining away at the LungDisease collider
3. Effect of elimination heuristics and barren-node pruning
4. Diagnostic performance of the reference and learned networks on held-out patients

Usage:
    python -m src.run_inference_queries
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.clinical_network import build_clinical_network
from src.inference import HEURISTICS, VariableElimination, ZeroProbabilityEvidenceError
from src.parameter_learning import BayesianEstimator, MaximumLikelihoodEstimator
from utilities.config import (
    DATA_DIR,
    DIAGNOSTIC_EVALUATION_FILENAME,
    INFERENCE_QUERIES_FILENAME,
    DatasetConfig,
    InferenceConfig,
    LearningConfig,
)
from utilities.data_utils import split_train_test
from utilities.io_utils import load_dataframe, save_dataframe
from utilities.logging_utils import get_logger
from utilities.metrics import binary_log_loss, brier_score, roc_auc

logger = get_logger("run_inference_queries")


def describe(evidence: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in evidence.items()) or "(no evidence)"


def clinical_queries(network, engine) -> pd.DataFrame:
    cases = [
        ("LungDisease", "present", {}),
        ("LungDisease", "present", {"Cough": "yes"}),
        ("LungDisease", "present", {"XRay": "abnormal"}),
        ("LungDisease", "present", {"Cough": "yes", "XRay": "abnormal"}),
        ("LungDisease", "present", {"Cough": "yes", "XRay": "abnormal", "Smoking": "yes"}),
        ("LungDisease", "present", {"Cough": "yes", "XRay": "normal"}),
        ("Bronchitis", "present", {"Cough": "yes", "Dyspnea": "yes"}),
        ("Bronchitis", "present", {"Cough": "yes", "Dyspnea": "yes", "XRay": "abnormal"}),
        ("Smoking", "yes", {"LungDisease": "present"}),
        ("XRay", "abnormal", {"Smoking": "yes", "Pollution": "high"}),
    ]
    rows = []
    for variable, state, evidence in cases:
        exact = engine.posterior(variable, evidence)[state]
        brute = network.marginal(variable, evidence)[state]
        rows.append(
            {
                "query": f"P({variable}={state} | {describe(evidence)})",
                "variable_elimination": exact,
                "enumeration": brute,
                "abs_difference": abs(exact - brute),
            }
        )
    return pd.DataFrame(rows)


def explaining_away(engine) -> pd.DataFrame:
    rows = [
        ("P(Pollution=high)", engine.posterior("Pollution")["high"]),
        ("P(Pollution=high | LungDisease=present)", engine.posterior("Pollution", {"LungDisease": "present"})["high"]),
        (
            "P(Pollution=high | LungDisease=present, Smoking=yes)",
            engine.posterior("Pollution", {"LungDisease": "present", "Smoking": "yes"})["high"],
        ),
        ("P(Smoking=yes)", engine.posterior("Smoking")["yes"]),
        ("P(Smoking=yes | LungDisease=present)", engine.posterior("Smoking", {"LungDisease": "present"})["yes"]),
        (
            "P(Smoking=yes | LungDisease=present, Pollution=high)",
            engine.posterior("Smoking", {"LungDisease": "present", "Pollution": "high"})["yes"],
        ),
    ]
    return pd.DataFrame(rows, columns=["query", "probability"])


def elimination_comparison(network) -> pd.DataFrame:
    evidence = {"Cough": "yes", "XRay": "abnormal"}
    rows = []
    for prune in (True, False):
        for heuristic in HEURISTICS:
            order, width = VariableElimination(network, heuristic, prune).plan("LungDisease", evidence)
            rows.append(
                {
                    "pruning": prune,
                    "heuristic": heuristic,
                    "variables_eliminated": len(order),
                    "max_factor_scope": width,
                    "order": " > ".join(order),
                }
            )
    return pd.DataFrame(rows)


def diagnostic_scores(engine, test: pd.DataFrame, cfg: InferenceConfig) -> pd.Series:
    """P(target = positive | observed variables) for every test row (NaN if evidence is impossible)."""
    cache = {}
    unique = test[list(cfg.observed)].drop_duplicates()
    for _, row in unique.iterrows():
        evidence = row.to_dict()
        try:
            cache[tuple(row)] = engine.posterior(cfg.target, evidence)[cfg.positive_state]
        except ZeroProbabilityEvidenceError:
            cache[tuple(row)] = np.nan
    return pd.Series([cache[tuple(r)] for r in test[list(cfg.observed)].itertuples(index=False)])


def diagnostic_evaluation(reference, train, test, cfg: InferenceConfig, lcfg: LearningConfig) -> pd.DataFrame:
    labels = (test[cfg.target] == cfg.positive_state).to_numpy()
    oracle = diagnostic_scores(VariableElimination(reference, cfg.heuristic), test, cfg)

    models = {"reference_network": reference}
    for n in cfg.training_sizes:
        subset = train.iloc[:n]
        models[f"maximum_likelihood_n{n}"] = MaximumLikelihoodEstimator.from_network(reference).fit(subset)
        models[f"bayesian_bdeu_n{n}"] = BayesianEstimator.from_network(
            reference, prior="bdeu", equivalent_sample_size=lcfg.equivalent_sample_size
        ).fit(subset)

    rows = []
    prevalence = float((train[cfg.target] == cfg.positive_state).mean())
    baseline = np.full(len(test), prevalence)
    rows.append(
        {
            "model": "prevalence_baseline",
            "auc": 0.5,
            "log_loss": binary_log_loss(labels, baseline),
            "brier": brier_score(labels, baseline),
            "mean_abs_gap_to_reference": float(np.abs(baseline - oracle).mean()),
            "unanswerable_rows": 0,
        }
    )
    for name, model in models.items():
        scores = oracle if name == "reference_network" else diagnostic_scores(
            VariableElimination(model, cfg.heuristic), test, cfg
        )
        valid = scores.notna().to_numpy()
        rows.append(
            {
                "model": name,
                "auc": roc_auc(labels[valid], scores[valid]),
                "log_loss": binary_log_loss(labels[valid], scores[valid]),
                "brier": brier_score(labels[valid], scores[valid]),
                "mean_abs_gap_to_reference": float(np.abs(scores[valid] - oracle[valid]).mean()),
                "unanswerable_rows": int((~valid).sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    cfg, lcfg = InferenceConfig(), LearningConfig()
    network = build_clinical_network()
    engine = VariableElimination(network, cfg.heuristic)

    queries = clinical_queries(network, engine)
    print("\nPosterior queries (variable elimination vs enumeration)")
    print(queries.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    logger.info("Largest discrepancy with enumeration: %.2e", queries["abs_difference"].max())

    print("\nExplaining away")
    print(explaining_away(engine).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nElimination plans for P(LungDisease | Cough=yes, XRay=abnormal)")
    print(elimination_comparison(network).to_string(index=False))

    data = load_dataframe(DatasetConfig().path)
    train, test = split_train_test(data, lcfg.test_fraction, lcfg.split_seed)
    evaluation = diagnostic_evaluation(network, train, test, cfg, lcfg)
    print(f"\nDiagnosing {cfg.target} from {', '.join(cfg.observed)} on {len(test)} held-out patients")
    print(evaluation.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    save_dataframe(queries, DATA_DIR / INFERENCE_QUERIES_FILENAME)
    path = save_dataframe(evaluation, DATA_DIR / DIAGNOSTIC_EVALUATION_FILENAME)
    logger.info("Saved results to %s", DATA_DIR)


if __name__ == "__main__":
    main()
