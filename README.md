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
| `src/scoring.py`           | BIC, K2 and BDeu decomposable structure scores                 |
| `src/structure_learning.py`| Hill-climbing / tabu search over DAGs with random restarts     |
| `src/graph_metrics.py`     | CPDAGs, Markov equivalence, skeleton metrics, structural Hamming distance |
| `src/learn_structure.py`   | Learns structures with each score and compares to the reference|
| `src/structure_study.py`   | Structure recovery as a function of training-set size          |
| `src/run_pipeline.py`      | End-to-end workflow and final model comparison                 |
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
python -m src.learn_structure    # structure learning, edge-level comparison, search trace
python -m src.structure_study    # structure recovery vs training size
python -m src.run_pipeline       # end-to-end run with final model comparison
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

Structure learning:

```python
from src.scoring import DiscreteDataset, DecomposableScore
from src.structure_learning import HillClimbSearch
from src.graph_metrics import structure_comparison

score = DecomposableScore(DiscreteDataset(training_data, network.states), "bic")
result = HillClimbSearch(score, max_indegree=3, restarts=20, perturbation_steps=10).fit()
structure_comparison(network.graph, result.graph)
```

Exact inference:

```python
from src.inference import VariableElimination

engine = VariableElimination(network, heuristic="min_fill")
engine.posterior("LungDisease", {"Cough": "yes", "XRay": "abnormal"})
engine.map_assignment(["LungDisease", "Bronchitis"], {"Cough": "yes", "XRay": "abnormal"})
engine.evidence_probability({"Cough": "yes"})
```

## Results summary

All figures come from the held-out evaluation in `python -m src.run_pipeline` (16,000 training rows, 4,000 test rows).

| Model                | KL to reference | Held-out log-likelihood | Diagnosis AUC |
|----------------------|-----------------|-------------------------|---------------|
| Reference            | 0               | -3.8953                 | 0.9855        |
| True structure, learned parameters | 0.0005 | -3.8956             | 0.9854        |
| Learned structure and parameters   | 0.0007 | -3.8958             | 0.9852        |

- Parameter learning: Bayesian estimation avoids the zero-probability failures of maximum likelihood on small samples.
- Structure learning: the skeleton is recovered exactly from about 16,000 samples. Edge orientation is identifiable only up to Markov equivalence, so equivalence-class (CPDAG) distance is reported alongside raw edge errors.
- Greedy hill climbing can stall in local optima; the tabu list and random restarts recover better-scoring graphs.

## Project status

**Completed**
- Directed acyclic graph with cycle rejection, edge editing, topological ordering, Markov blanket and d-separation queries
- Discrete factor algebra: product, marginalisation, evidence reduction, normalisation
- Bayesian network class with validated CPTs, joint probability, log-likelihood and ancestral sampling
- Reference clinical network (8 variables, 9 edges, 22 free parameters) and synthetic patient data generation
- Parameter learning: maximum likelihood and Bayesian estimation (BDeu and K2 Dirichlet priors)
- Exact inference by variable elimination with ordering heuristics and barren-node pruning
- Posterior, joint posterior, marginal MAP and evidence-probability queries
- Decomposable structure scores (BIC, K2, BDeu) and hill-climbing search with tabu list and restarts
- Equivalence-class aware structure evaluation (CPDAG, structural Hamming distance, skeleton precision and recall)
- Evaluation: KL divergence, held-out likelihood, AUC, log-loss, Brier score
- Sample-size studies for parameter and structure learning
- End-to-end pipeline and a self-verification suite covering every component

**Possible extensions**
- Hidden variables and expectation-maximisation
- Approximate inference (likelihood weighting, Gibbs sampling) for larger networks
- Constraint-based structure learning (PC algorithm) and hybrid methods
- Continuous and mixed variables
