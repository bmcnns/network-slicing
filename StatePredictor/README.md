# Digital Twin

## 1. Project Goal

This project implements the core components of a **Digital Twin (DT)** for a wireless network control system.

A Digital Twin is a learned surrogate of the real environment, designed to:
- Approximate environment dynamics
- Reduce expensive interactions with the real simulator
- Enable faster and cheaper agent training

---

## 2. Digital Twin Overview

The Digital Twin in this project is composed of two components:


- **State Predictor (Ver1 Implemented)**  
  Learns system dynamics and predicts future states from historical observations.

- **Reward Predictor (Not yet implemented)**  
  Will approximate the environment reward function and is **reserved for future work**.

At the current stage, **only the State Predictor is implemented and evaluated**.

---

## 3. State Predictor Description

The State Predictor models the system transition dynamics:

ŝ(t+1) = fθ( s(t−K+1), ..., s(t) )


- **Input**: a window of historical states

s(t−K+1:t) ∈ R^(K × 3)


- **Output**: predicted next state

ŝ(t+1) ∈ R^3


The predictor is implemented using an **LSTM-based neural network**.

---

## 4. Prediction Modes

The State Predictor supports two evaluation modes:

- **t+1 prediction (single-step)**  
  Uses ground-truth states as input at each step and evaluates local prediction accuracy.

- **t+n prediction (multi-step free rollout)**  
  Feeds predicted states back into the model to evaluate long-term dynamics consistency.

---

## 5. Directory Structure

```

StatePredictor/
│
├── data_builder.py
├── eval_rollout.py
├── model.py
├── train.py
├── plotting.py
├── main.py
│
├── data/
│   ├── states.npy
│   └── meta.json
│
├── model_components/
│   ├── lstm_dt.pt
│   └── standardizer.json
│
├── log/
│   └── train_log.jsonl
│
└── eval_outputs/

````

---

## 6. File Responsibilities

### `data_builder.py`
- Constructs datasets from either:
  - Trigonometric signals (`sin`, `cos`, `both`)
  - Simulator-generated data
- Produces continuous time-series data `(T, 3)`
- Splits data into **train / validation / test** sets (contiguous split)
- Builds sliding-window datasets for training
- Fits a **Standardizer** (mean / std) on training data
- Saves:
  - `data/states.npy`
  - `data/meta.json`
  - `model_components/standardizer.json`

> **Note:** The standardizer is part of the model and must always be used together with the trained LSTM.

---

### `model.py`
- Defines the **LSTMStatePredictor**
- Input shape: `(B, K, 3)`
- Output shape: `(B, 3)`
- Contains only model architecture (no data processing)

---

### `train.py`
- Handles model training
- Uses window-based datasets
- Default loss function: **MSE**
- Logs metrics for **train / validation / test**
- Logs are saved in **JSONL** format for post-processing

---

### `plotting.py`
- Centralized visualization module
- Automatically detects and plots all available metrics:
  - MSE
  - MAE
  - RSE
  - R²
- Supports:
  - Metric curves (overall + per-dimension)
  - True vs. prediction scatter plots
  - Rollout trajectory visualization
  - **One-click plotting of all figures**

---

### `eval_rollout.py`
- Performs model evaluation:
  - t+1 (single-step) rollout
  - t+n (multi-step free rollout)
- Saves rollout results as:
  - `.npz`
  - `.csv`
- Generates rollout comparison plots

---

### `main.py`
**Recommended entry point.**

From a single script, it performs:
1. Dataset construction
2. Standardizer fitting and saving
3. Model training
4. Model & standardizer saving
5. t+1 and t+n rollout evaluation
6. Automatic plotting of all results

---

## 7. Output Artifacts

- `model_components/`
  - `lstm_dt.pt`: trained LSTM model
  - `standardizer.json`: fitted normalization parameters
- `log/`
  - `train_log.jsonl`: per-epoch metrics
- `eval_outputs/`
  - rollout details and trajectories

---

## 8. How to Run

1. Open `main.py`
2. Configure:
   - Data mode (`trig` or `simulator`)
   - Number of training epochs
3. Run:

```bash
python main.py
````

Paths of run

                 MODE

                   │
        ┌──────────┼──────────┐
        │          │          │
      build      train       eval
        │          │          │
        │          │          │
        │          ▼          │
        │     rebuild=False
        │    
        │            ▼
        │        load states
        │            │
        │            ▼
        │          train
        │
        ▼
  dataset ready

---

## 9. Current Status & Future Work

* **State Predictor**: ✓ implemented
* **Reward Predictor**: ✗ reserved

Future extensions include:

* Reward prediction
* Agent training using the Digital Twin
* Multi-policy simulator data collection


