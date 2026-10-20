# Clinical Bayesian Network

A from-scratch implementation of discrete probabilistic graphical models (Bayesian networks)
applied to respiratory-disease diagnosis. The project builds a reference clinical network,
samples synthetic patient records from it, and provides the core data structures needed for
inference and learning.

## Reference network

```
Age ──► Smoking ──► LungDisease ◄── Pollution
            │           │   │  └────────► XRay
            ▼           ▼   ▼
        Bronchitis ──► Cough, Dyspnea
```

| Variable    | States                        | Parents                |
|-------------|-------------------------------|------------------------|
| Age         | young, middle_aged, senior    | –                      |
| Pollution   | low, high                     | –                      |
| Smoking     | no, yes                       | Age                    |
| LungDisease | absent, present               | Smoking, Pollution     |
| Bronchitis  | absent, present               | Smoking                |
| Cough       | no, yes                       | LungDisease, Bronchitis|
| Dyspnea     | no, yes                       | LungDisease, Bronchitis|
| XRay        | normal, abnormal              | LungDisease            |

## Project layout

```
data/        Generated datasets and reports (git-ignored)
src/         Core library and runnable scripts
utilities/   Configuration, logging, I/O and information-theoretic helpers
```

| Module                     | Purpose                                                        |
|----------------------------|----------------------------------------------------------------|
| `src/graph.py`             | DAG, topological order, Markov blanket, d-separation           |
| `src/factor.py`            | Discrete factors: product, marginalisation, reduction          |
| `src/bayesian_network.py`  | CPT storage, joint probability, likelihood, ancestral sampling |
| `src/clinical_network.py`  | Reference network definition                                   |
| `src/generate_dataset.py`  | Draws and saves synthetic patient records                      |
| `src/explore_data.py`      | Marginal comparison and pairwise mutual information            |
| `src/sanity_checks.py`     | Verification of all components                                 |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run all commands from the project root.

```bash
python -m src.sanity_checks      # verify the implementation
python -m src.generate_dataset   # writes data/clinical_observations.csv
python -m src.explore_data       # prints diagnostics, writes data/pairwise_mutual_information.csv
```

Querying the model:

```python
from src.clinical_network import build_clinical_network

network = build_clinical_network()
network.marginal("LungDisease", {"Cough": "yes", "XRay": "abnormal"})
network.graph.d_separated("Smoking", "Pollution", given=["LungDisease"])   # False
```
