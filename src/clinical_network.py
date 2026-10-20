"""Reference clinical network used to generate observational data.

Structure
---------
Age -> Smoking
Smoking, Pollution -> LungDisease
Smoking -> Bronchitis
LungDisease, Bronchitis -> Cough
LungDisease, Bronchitis -> Dyspnea
LungDisease -> XRay
"""
from __future__ import annotations

from src.bayesian_network import BayesianNetwork


def build_clinical_network() -> BayesianNetwork:
    network = BayesianNetwork(name="clinical_respiratory_network")

    network.add_variable("Age", ["young", "middle_aged", "senior"], [], [0.30, 0.45, 0.25])
    network.add_variable("Pollution", ["low", "high"], [], [0.65, 0.35])

    # P(Smoking | Age)
    network.add_variable(
        "Smoking", ["no", "yes"], ["Age"],
        [[0.70, 0.30],
         [0.65, 0.35],
         [0.75, 0.25]],
    )

    # P(LungDisease | Smoking, Pollution)
    network.add_variable(
        "LungDisease", ["absent", "present"], ["Smoking", "Pollution"],
        [[[0.99, 0.01], [0.97, 0.03]],
         [[0.93, 0.07], [0.85, 0.15]]],
    )

    # P(Bronchitis | Smoking)
    network.add_variable(
        "Bronchitis", ["absent", "present"], ["Smoking"],
        [[0.90, 0.10],
         [0.60, 0.40]],
    )

    # P(Cough | LungDisease, Bronchitis)
    network.add_variable(
        "Cough", ["no", "yes"], ["LungDisease", "Bronchitis"],
        [[[0.90, 0.10], [0.35, 0.65]],
         [[0.25, 0.75], [0.05, 0.95]]],
    )

    # P(Dyspnea | LungDisease, Bronchitis)
    network.add_variable(
        "Dyspnea", ["no", "yes"], ["LungDisease", "Bronchitis"],
        [[[0.90, 0.10], [0.30, 0.70]],
         [[0.20, 0.80], [0.05, 0.95]]],
    )

    # P(XRay | LungDisease)
    network.add_variable(
        "XRay", ["normal", "abnormal"], ["LungDisease"],
        [[0.95, 0.05],
         [0.10, 0.90]],
    )

    network.validate()
    return network
