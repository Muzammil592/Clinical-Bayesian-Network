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
| `src/parameter_learning.py`| Maximum likelihood and Bayesian (Dirichlet) CPT estimation     |
| `src/evaluation.py`        | KL divergence, held-out likelihood and CPT comparison metrics  |
| `src/learn_parameters.py`  | Fits estimators on training data and reports held-out results  |
| `src/sample_size_study.py` | Estimation error as a function of training-set size            |
| `src/inference.py`         | Variable elimination: posteriors, MAP, evidence probability    |
| `src/run_inference_queries.py` | Clinical queries, explaining away, diagnostic evaluation   |
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
python -m src.learn_parameters   # fits MLE / Bayesian CPTs, writes data/learned_parameter_comparison.csv
python -m src.sample_size_study  # error vs training size, writes data/sample_size_study.csv
python -m src.run_inference_queries  # exact inference demos and diagnostic evaluation
```

Querying the model:

```python
from src.clinical_network import build_clinical_network

network = build_clinical_network()
network.marginal("LungDisease", {"Cough": "yes", "XRay": "abnormal"})
network.graph.d_separated("Smoking", "Pollution", given=["LungDisease"])   # False
```

Learning parameters for a known structure:

```python
from src.parameter_learning import BayesianEstimator
from src.evaluation import network_kl_divergence

estimator = BayesianEstimator.from_network(network, prior="bdeu", equivalent_sample_size=10)
learned = estimator.fit(training_data)
network_kl_divergence(network, learned)
```

Exact inference:

```python
from src.inference import VariableElimination

engine = VariableElimination(network, heuristic="min_fill")
engine.posterior("LungDisease", {"Cough": "yes", "XRay": "abnormal"})
engine.map_assignment(["LungDisease", "Bronchitis"], {"Cough": "yes", "XRay": "abnormal"})
engine.evidence_probability({"Cough": "yes"})
```

## Project status

**Completed**
- Directed acyclic graph with cycle rejection, topological ordering, Markov blanket and d-separation queries
- Discrete factor algebra: product, marginalisation, evidence reduction, normalisation
- Bayesian network class with validated CPTs, joint probability, log-likelihood and exact enumeration-based marginals
- Reference clinical network (8 variables, 9 edges, 22 free parameters)
- Ancestral sampling and synthetic patient dataset generation
- Exploratory analysis: empirical vs exact marginals and pairwise mutual information
- Self-verification suite covering factors, normalisation, independence statements, sampling accuracy and estimators
- Parameter learning: maximum likelihood and Bayesian estimation (BDeu and K2 Dirichlet priors)
- Evaluation: KL divergence via the chain rule, held-out log-likelihood, zero-probability detection
- Sample-size study comparing estimators across training-set sizes
- Exact inference by variable elimination with ordering heuristics and barren-node pruning
- Posterior, joint posterior, marginal MAP and evidence-probability queries
- Diagnostic evaluation (AUC, log-loss, Brier score) of reference and learned networks

**Planned**
- Structure learning with score-based search (BIC) and comparison against the true graph
- Model evaluation: held-out likelihood, structural Hamming distance, posterior-query accuracy
