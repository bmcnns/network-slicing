# Digital Twin (DT) Module README

## 1. Overview

This module implements a **Digital Twin (DT)** for the network slicing simulator.

The DT consists of two learned components:

* **State Predictor (LSTM)**: models system dynamics
* **Reward Predictor (DNN)**: estimates reward given state/action/hidden features

The goal is to approximate the simulator such that:

* It can be used as a **fast surrogate environment**
* It supports **rollout (T+1, T+N)** and **online step() interaction (Gym-style)**

---

## 2. Project Structure

Key files:

```
main.py
    End-to-end pipeline example (collect → train → rollout → plot)

pipeline.py
    Modular orchestration for training and evaluation

env.py
    Digital Twin Gym environment (IMPORTANT)

DT_test.py
    Example: how to use DT as an environment

dataset.py
    Data collection from simulator (PPO / random etc.)

dataloader.py
    Sequence chunk construction for training

models.py
    - LSTM state predictor
    - DNN reward predictor

trainer.py
    Training loops

rollout.py
    T+1 and T+N rollout implementations

plotting.py
    Visualization (loss curves, rollout scatter, etc.)

standardizer.py
    Z-score normalization utilities
```

---

## 3. Pipeline Usage (main.py)

`main.py` demonstrates the **fully modular pipeline**:

1. Collect transitions from simulator
2. Fit standardizers
3. Normalize dataset
4. Train models
5. Run rollout (T+1, T+N)
6. Plot results

Each stage is independent and reusable.

---

## 4. Using DT as an Environment (DT_test.py)

`env.py` exposes a Gym-style environment:

```python
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

Important:

* `step()` is designed to be **equivalent to one step of T+N rollout**
* Internally uses a **sequence buffer (length = seq_len)**
* Fully autoregressive

---

## 5. Configuration (VERY IMPORTANT)

Everything is controlled via:

```python
env.set_config({...})
```

You typically **do NOT need to manually load models or standardizers**.

Expected behavior:

* Paths are automatically created
* Models and standardizers are automatically saved/loaded
* Versioning is handled internally

⚠️ However, you MUST ensure:

* Correct paths exist for:

  * `norm_npz_path`
  * `state_standardizer.pkl`
  * `reward_standardizer.pkl`
  * `action_standardizer.pkl`

If something behaves incorrectly, **first check paths**.

---

## 6. Standardizer (Critical Concept)

The system uses **Z-score normalization**:

x_norm = (x - mean) / (std + eps)

Why it matters:

* Models are trained in **normalized space**
* `env.step()` also operates internally in **model space**
* Outputs are converted back to raw space only when returned

Implications:

* Incorrect standardizer = completely broken predictions
* Mixing raw / normalized inputs = explosion

---

## 7. Artifacts & Versioning (IMPORTANT)

All outputs are automatically saved under:

```
artifacts/
```

Structure:

```
artifacts/
└── versions/
    └── <version_id>/
        ├── data/
        ├── models/
        ├── logs/
        └── plots/
```

Key points:

* Each run creates a new **version folder**
* `version_id` = **run start timestamp**

  * format: `YYYYMMDD_HHMMSS`
* Everything related to that run is stored inside this folder:

  * collected data
  * trained models
  * standardizers
  * rollout results
  * plots

`DT_test.py` specifically:

* All outputs are written into `artifacts/versions/`
* Each test run is isolated and reproducible

---

## 8. Current Limitations (IMPORTANT)

### 1. Data Collection Distribution

The DT is sensitive to the **distribution of collected training data**.

Observed behavior:

* When the dataset is collected mainly with **PPO actions**, the trained DT works well under PPO-like interaction, but generalization to very different actions can be weak.
* When the dataset is collected using a **random policy**, the trained DT shows much better robustness.
* A DT trained on random-policy data can still interact normally with a **PPO policy agent**.

Interpretation:

* The main issue is not simply “policy sensitivity” of the model itself.
* The more likely cause is **insufficient state-action coverage** in the training dataset.
* PPO-collected data appears to be narrower and more policy-specific, while random collection provides broader coverage of the action space.

---

### 2. Action Distribution Coverage

Current findings suggest that **data coverage matters more than the specific policy label**.

Problem:

* PPO-based collection may explore only a limited region of the state-action space.
* This can make the learned DT accurate mainly near the PPO-induced trajectory distribution.

Consequence:

* Generalization to unseen or weakly covered actions may degrade.
* Broader exploration during collection leads to a more robust DT.

Current status:

* Random-policy collection currently produces better overall behavior than expected.
* More systematic comparison is still needed:

  * PPO → PPO
  * PPO → random
  * random → random
  * random → PPO

---

### 3. Simulator Dependency

Current setup assumes:

> **Simulator = Config 1**

If you want to replace the simulator:

Replace the simulator code in `dataset.py`.

No other module should require modification, as the rest of the pipeline is already modularized.

---


## 9. Key Design Decisions

* Fully modular pipeline
* Clear separation:

  * data → training → rollout → environment
* DT environment = **online version of rollout**
* Internal state always maintained in **model space**
* Automatic versioning and artifact tracking

---

## 10. Quick Summary

* DT works and is stable
* Pipeline is modular and ready
* Environment interface is usable
* All outputs are versioned automatically
* **Main weakness: policy sensitivity due to data coverage**

---
