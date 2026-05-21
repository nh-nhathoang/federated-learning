# Comparing Geographic and Correlation-Based Graphs in Personalized Federated Wind Speed Prediction

This project studies personalized federated learning (FL) for short-term wind speed prediction using weather observations from the Finnish Meteorological Institute (FMI).

The project compares two graph construction strategies in graph-based federated learning:

- System A: geographic k-nearest-neighbor graph
- System B: correlation-based graph using Pearson correlation

The implementation uses synchronous gradient-based message passing with generalized total variation minimization (GTVMin).

---

## Problem Description

Each FMI weather station is treated as one federated learning node.

The prediction task is:

- Predict wind speed 3 hours ahead

using the following weather variables:

- Wind speed
- Maximum temperature
- Minimum temperature
- Relative humidity
- Air pressure

Each station trains a personalized linear regression model locally while exchanging model parameters with neighboring stations through a graph structure.

---

## Dataset

Source:
- Finnish Meteorological Institute (FMI) Open Data

Time range:
- January 2025 – December 2025

Number of stations:
- 12 stations in southern Finland

Train / validation / test split:
- 60% train
- 20% validation
- 20% test

Chronological splitting is used because this is a time-series prediction problem.

---

## Federated Learning Systems

### System A — Geographic Graph

- Stations connect to their k nearest geographic neighbors
- Edge weights:
  
\[
A_{i,i'} = \exp(-d_{i,i'}/\sigma)
\]

where:
- \(d_{i,i'}\) is geographic distance
- \(\sigma\) controls exponential decay

---

### System B — Correlation Graph

- Stations connect only when Pearson correlation exceeds a threshold

\[
A_{i,i'} = \max(0, \rho_{i,i'})
\]

where:
- \(\rho_{i,i'}\) is Pearson correlation between wind-speed time series

---

## Method

The project uses generalized total variation minimization (GTVMin):

\[
\sum_i L_i(w^{(i)})
+
\alpha
\sum_{(i,i') \in E}
A_{i,i'}
\|w^{(i)} - w^{(i')}\|_2^2
\]

where:
- the first term minimizes local prediction error
- the second term regularizes neighboring stations to learn similar parameters

Training is performed using synchronous gradient-based updates.

---

## Repository Structure

```text
.
├── fl_wind.py
├── requirements.txt
├── data/
├── results/
└── figures/
```

---

## Installation

Clone repository:

```bash
git clone https://github.com/nh-nhathoang/federated-learning.git
cd federated-learning
```

Install dependencies:

```bash
pip3 install -r requirements.txt
```

---

## Running the Project

Place:
- station Excel files
- `stations.csv`

inside the `data/` folder.

Run:

```bash
python3 fl_wind.py --data_dir data
```

Results will be saved to:

```text
results/
```

including:
- station-level MSE
- learned parameters
- graph visualizations
- summary tables

---

## Dependencies

Main packages:

- numpy
- pandas
- scikit-learn
- scipy
- matplotlib
- networkx
- openpyxl

See `requirements.txt` for exact versions.

---

## Results Summary

The experiments show:

- Local-only training achieves the lowest final test MSE
- Geographic and correlation-based FL systems achieve similar performance
- Different graph constructions produce different collaboration structures
- Federated learning does not clearly outperform local-only training in this experimental setting

The results also suggest that geographically nearby stations do not always exhibit highly similar wind-speed behavior.

## AI Assistance

This project used AI-based tools to assist with:
- debugging Python implementation issues,
- improving report structure and wording,
- generating visualization ideas