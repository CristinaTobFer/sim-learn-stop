# A Sim-Learnheuristic Algorithm for the Stochastic and Context-Aware Team Orienteering Problem

A Sim-Learnheuristic framework for solving the Stochastic and Context-Aware Team Orienteering Problem by integrating metaheuristic optimization, Monte Carlo simulation, and machine learning.

This repository contains the implementation of a Sim-Learnheuristic framework for the Stochastic and Context-Aware Team Orienteering Problem.

The proposed approach extends traditional simheuristics by integrating machine learning and clustering techniques to model travel times under different contextual conditions, such as traffic congestion and weather. Historical data are used both to predict context-dependent travel times and to generate customized probability distributions for the simulation process, enabling a more realistic evaluation of candidate solutions.

The methodology is compared against a deterministic metaheuristic and a standard simheuristic on extended benchmark instances.

## Problem

The Team Orienteering Problem (TOP) consists of designing up to \(m\) routes, each starting and ending at designated depots, that collect as much reward as possible without exceeding a time budget \(T_{\max}\).

This work addresses a **stochastic and context-aware** variant in which travel times are random and depend on operational conditions (traffic, weather, incidents, etc.). Candidate routes must therefore remain robust when the realized travel times differ from their deterministic estimates.

Travel times are conditioned on seven contextual covariates:

| Covariate         | Levels                                      |
| ----------------- | ------------------------------------------- |
| `time_of_day`     | `peak`, `valley`, `night`                   |
| `day_of_week`     | `workday`, `weekend`                        |
| `weather`         | `dry`, `rain`, `snow_fog`                   |
| `accidents`       | `yes`, `no`                                 |
| `special_events`  | `yes`, `no`                                 |
| `season`          | `normal`, `holiday`                         |
| `roadworks`       | `yes`, `no`                                 |

## Methodology

All three algorithms share the same constructive core: a GRASP procedure with restricted candidate list (parameter \(\alpha\)) followed by a swap-based local search. They differ in how travel times are estimated and how solutions are scored.

### Deterministic metaheuristic

Uses the element-wise mean of the historical travel-time matrices. A solution is scored with this single deterministic matrix.

### Simheuristic

Uses the same mean matrix to construct routes, but scores each candidate with Monte Carlo simulation. For every origin–destination pair, a lognormal distribution is fitted to the **full** historical sample (100 simulated matrices).

### Sim-Learnheuristic

1. **Prediction.** A `HistGradientBoostingRegressor` is trained per origin–destination pair on the encoded covariates, with a `log1p` target transform, to obtain a context-specific deterministic matrix used during construction.
2. **Clustering.** Historical contexts are clustered with HDBSCAN (Manhattan distance). The current scenario is assigned to the nearest cluster (or treated as noise).
3. **Simulation.** Per-arc lognormal distributions are fitted inside the assigned cluster (bootstrap resampling if fewer than 30 observations). The candidate solution is scored over 100 sampled matrices.
4. **Validation.** After optimization, all algorithms are re-evaluated on 1,000 independent matrices sampled from the same context-specific distributions.

## Repository structure

```text
sim-learn-stop/
├── main.py                         # Experiment runner (three algorithms × instances × scenarios)
├── Instance.py                     # Instance I/O, clustering, ML prediction, simulation
├── Algorithms/
│   ├── Algorithm.py                # Shared GRASP + local search
│   ├── MetaheuristicAlgorithm.py
│   ├── SimHeuristicAlgorithm.py
│   └── SimLearnHeuristicAlgorithm.py
├── InstanceGenerator/              # Extension of Chao TOP instances with historical matrices
│   ├── main.py
│   ├── GeneratingInstance.py
│   └── ExtendInstance.py
└── tuning/                         # irace configuration for hyperparameter search
    ├── scenario.txt
    ├── parameters.txt
    ├── main.py                     # Target runner
    └── Instances/
```

Benchmark TOP files and the generated historical matrices are expected under `original_instances/` (Chao sets). They are not shipped in this repository; the published dataset is available on Zenodo.

## Data

The extended instances (original Chao TOP files plus historical travel-time matrices and covariates) are published as:

> Tobar Fernández, C. (2026). *Benchmark TOP Instances Augmented with Historical Travel Times and Covariates* (v1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.18925720

Download [`original_instances.zip`](https://zenodo.org/records/18925720) and extract it at the repository root so that `original_instances/` sits next to `main.py`. The archive is about 4.9 GB (CC BY 4.0).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18925720.svg)](https://doi.org/10.5281/zenodo.18925720)

## Requirements

- Python 3.10+
- [irace](https://cran.r-project.org/package=irace) (optional, only for retuning)

```bash
pip install numpy scipy scikit-learn hdbscan joblib
```

## Generating extended instances

The recommended way to obtain the data is to download the Zenodo archive above. To regenerate the historical matrices from scratch, place the original Chao TOP files (`.txt`) under `original_instances/Set_*/` and run:

```bash
python InstanceGenerator/main.py
```

For each selected instance the generator creates 10,000 historical travel-time matrices. Each matrix is sampled from a lognormal model whose mean is the Euclidean time inflated by the active covariates. Files are written next to the instance, e.g. `original_instances/Set_21_234/p2.3.g/`.

The generator currently extends the following instances:

| Set           | Instances              |
| ------------- | ---------------------- |
| `Set_21_234`  | `p2.3.g`, `p2.3.k`     |
| `Set_32_234`  | `p1.3.k`, `p1.4.b`     |
| `Set_33_234`  | `p3.2.i`, `p3.2.j`     |
| `Set_64_234`  | `p6.4.a`, `p6.4.n`     |
| `Set_66_234`  | `p5.2.t`, `p5.3.f`     |
| `Set_100_234` | `p4.3.r`, `p4.3.i`     |
| `Set_102_234` | `p7.2.j`, `p7.3.j`     |

## Running the experiments

From the repository root:

```bash
python main.py
```

The script solves 12 instances under five operational scenarios of increasing congestion, comparing the three algorithms. Default settings (tuned with irace):

| Parameter             | Value   |
| --------------------- | ------- |
| `alpha` (GRASP)       | 0.9964  |
| `max_depth`           | 7       |
| `learning_rate`       | 0.168   |
| `max_iter`            | 264     |
| `l2_regularization`   | 0.9441  |
| Time limit            | 300 s   |
| Seed                  | 54      |

Scenarios evaluated in `main.py`:

1. Night, weekend, dry
2. Night, workday, rain
3. Peak, workday, dry
4. Peak, workday, rain
5. Peak, workday, rain, roadworks

Intermediate results are written to `metrics/metrics_<timestamp>.csv` and a final table to `final_metrics_<timestamp>.csv` (semicolon-separated, comma as decimal separator). Columns:

- `Instance`, `Scenario`, `Algorithm`
- `Score Simulation` — score used during the search
- `Score Validation` — mean score over 1,000 hold-out matrices
- `Number of Iterations`

## Hyperparameter tuning

The `tuning/` folder contains an [irace](https://mlopez-ibanez.github.io/irace/) scenario that searches over \(\alpha\) and the gradient-boosting hyperparameters. From that directory, with irace installed:

```r
library(irace)
irace(scenarioFile = "scenario.txt")
```

The elite configuration reported after 60 experiments is the one used in `main.py`.

## Citation

If you use this code or the instances, please cite the accompanying paper and the dataset:

> *A Sim-Learnheuristic Algorithm for the Stochastic and Context-Aware Team Orienteering Problem* (forthcoming).

> Tobar Fernández, C. (2026). *Benchmark TOP Instances Augmented with Historical Travel Times and Covariates* (v1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.18925720
