# main.py
from __future__ import annotations

import json
import os
import logging
from typing import Tuple, Dict, Any

import numpy as np
import torch
from stable_baselines3 import PPO

from StatePredictor.eval_rollout import (
    rollout_t_plus_1,
    rollout_t_plus_n,
    save_rollout_result,
    save_rollout_csv,
    plot_rollout_overlay,
    plot_rollout_3d,
    plot_rollout_4d_color,
)
from StatePredictor.plotting import load_jsonl, plot_all
from StatePredictor.data_builder import (
    DatasetSplits,
    Standardizer,
    SimulatorDataBuilder,
    TrigonometricDataBuilder,
    ChaosDataBuilder,
    CSVDataBuilder,
)
from StatePredictor.train import train_from_splits, load_trained_model

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("main")


# =========================================================
# Global Config
# =========================================================
MODE = "eval"   # "build" | "train" | "eval"
REBUILD_DATA = False

DATA_DIR = "data/random_policy_1000steps"
LOG_PATH = "log/train_log.jsonl"
MODEL_PATH = "./model_components/lstm_dt.pt"
SCALER_PATH = "./model_components/standardizer.json"
RUN_CONFIG_PATH = "./model_components/run_config.json"
EVAL_OUT_DIR = "./eval_outputs"

CSV_PATH = "data/agent.csv"
SEED = 0

# Data config
NUM_STEPS = 200
MODE_TRIG = "sin"  # "sin" | "cos" | "both"
WINDOW_SIZE = 1
STRIDE = 1
NOISE_STD = 1
SPLIT = (0.7, 0.15, 0.15)
STANDARDIZE = True

# Train config
EPOCHS = 200
BATCH_SIZE = 128
LR = 1e-3
MAX_PAIRS = 16

# Model config
HIDDEN_DIM = 64
NUM_LAYERS = 1
DROPOUT = 0.0

# Eval config
ROLLOUT_LENGTH = 10000
ROLLOUT_START_T = 0


# =========================================================
# Config / Builders
# =========================================================
def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    return device


def make_run_cfg(input_dim: int, device: torch.device) -> Dict[str, Any]:
    return {
        "seed": SEED,
        "device": str(device),

        "num_steps": NUM_STEPS,
        "mode": MODE_TRIG,
        "dt": 0.01,
        "noise_std": NOISE_STD,
        "split": SPLIT,
        "standardize": STANDARDIZE,

        "window_size": WINDOW_SIZE,
        "stride": STRIDE,

        "input_dim": input_dim,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,

        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "loss": "MSE",

        "model_path": MODEL_PATH,
        "log_path": LOG_PATH,
        "data_dir": DATA_DIR,
    }


def make_builder():
    """
    Choose ONE builder here.
    """

    # -------- Simulator ----------
    builder = SimulatorDataBuilder(
        num_steps=NUM_STEPS,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        split=SPLIT,
        seed=SEED,
        standardize=STANDARDIZE,
        policy="random",
        meta={"env": "NetworkSlicingGymContinuous"},
        agent=None,  # PPO.load("../Simulator/models/PPO-Network-Slicing")
    )
    return builder

    # -------- Trig ----------
    # return TrigonometricDataBuilder(
    #     num_steps=NUM_STEPS,
    #     mode=MODE_TRIG,
    #     dt=0.01,
    #     window_size=WINDOW_SIZE,
    #     stride=STRIDE,
    #     noise_std=NOISE_STD,
    #     split=SPLIT,
    #     seed=SEED,
    #     standardize=STANDARDIZE,
    # )

    # -------- Chaos ----------
    # return ChaosDataBuilder(
    #     num_steps=20000,
    #     dt=0.01,
    #     window_size=WINDOW_SIZE,
    #     stride=STRIDE,
    #     split=SPLIT,
    #     standardize=STANDARDIZE,
    #     noise_std=NOISE_STD,
    #     seed=SEED,
    # )

    # -------- CSV ----------
    # return CSVDataBuilder(
    #     csv_path=CSV_PATH,
    #     window_size=WINDOW_SIZE,
    #     stride=STRIDE,
    #     split=SPLIT,
    #     seed=SEED,
    #     standardize=STANDARDIZE,
    # )


def build_data() -> DatasetSplits:
    logger.info("Building fresh dataset...")
    builder = make_builder()
    splits: DatasetSplits = builder.build()
    splits.save(DATA_DIR)
    logger.info(f"Dataset saved to: {DATA_DIR}")
    logger.info(f"Raw states shape: {splits.raw.states.shape} (T,D)")
    return splits


def load_data() -> DatasetSplits:
    logger.info(f"Loading cached dataset from: {DATA_DIR}")
    splits = DatasetSplits.load(
        DATA_DIR,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        split=SPLIT,
        standardize=STANDARDIZE,
    )
    logger.info(f"Loaded states shape: {splits.raw.states.shape} (T,D)")
    return splits


def prepare_data(rebuild: bool = False) -> DatasetSplits:
    if rebuild:
        return build_data()

    states_path = os.path.join(DATA_DIR, "states.npy")
    meta_path = os.path.join(DATA_DIR, "meta.json")
    if os.path.exists(states_path) and os.path.exists(meta_path):
        return load_data()

    logger.info("Cached dataset not found. Building fresh dataset instead...")
    return build_data()


def get_test_states_phys(states_phys: np.ndarray, split: Tuple[float, float, float]) -> np.ndarray:
    T = states_phys.shape[0]
    n_train = int(split[0] * T)
    n_val = int(split[1] * T)
    test_states_phys = states_phys[n_train + n_val:]

    logger.info(f"train size: {n_train}")
    logger.info(f"val size:   {n_val}")
    logger.info(f"test size:  {test_states_phys.shape[0]}")
    return test_states_phys


# =========================================================
# Build / Train / Eval entry points
# =========================================================
def build_main() -> None:
    _ = build_data()
    logger.info("Build-only done.")


def train_main(rebuild_data: bool = False) -> None:
    torch.manual_seed(SEED)
    device = get_device()

    splits = prepare_data(rebuild=rebuild_data)

    sample_x, sample_y = splits.train[0]
    input_dim = int(sample_x.shape[-1])
    logger.info(f"Input dim D = {input_dim}")

    extra_run_cfg = make_run_cfg(input_dim=input_dim, device=device)

    train_from_splits(
        splits=splits,
        device=device,
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        log_path=LOG_PATH,
        model_path=MODEL_PATH,
        run_config_path=RUN_CONFIG_PATH,
        max_pairs=MAX_PAIRS,
        extra_run_cfg=extra_run_cfg,
    )


def eval_main() -> None:
    device = get_device()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    states_path = os.path.join(DATA_DIR, "states.npy")
    if not os.path.exists(states_path):
        raise FileNotFoundError(f"Cached states not found: {states_path}")

    model, cfg = load_trained_model(
        model_path=MODEL_PATH,
        device=device,
    )

    scaler = Standardizer.load(SCALER_PATH)

    states_phys = np.load(states_path).astype(np.float32, copy=False)
    test_states_phys = get_test_states_phys(states_phys, SPLIT)

    K = WINDOW_SIZE
    L = ROLLOUT_LENGTH
    start_t = ROLLOUT_START_T

    logger.info("Running t+1 rollout...")
    res_t1 = rollout_t_plus_1(
        model=model,
        scaler=scaler,
        states_phys=test_states_phys,
        window_size=K,
        start_t=start_t,
        length=L,
        device=device,
    )

    logger.info("Running t+n rollout...")
    res_tn = rollout_t_plus_n(
        model=model,
        scaler=scaler,
        states_phys=test_states_phys,
        window_size=K,
        start_t=start_t,
        length=L,
        device=device,
    )

    os.makedirs(EVAL_OUT_DIR, exist_ok=True)
    save_rollout_result(res_t1, out_dir=EVAL_OUT_DIR, prefix="test")
    save_rollout_result(res_tn, out_dir=EVAL_OUT_DIR, prefix="test")

    plot_rollout_overlay(res_t1, title_prefix="TEST", save_plot=False, smooth_window=20)
    plot_rollout_overlay(res_tn, title_prefix="TEST", save_plot=False, smooth_window=20)

    state_dim = test_states_phys.shape[1]

    if state_dim >= 3:
        plot_rollout_3d(res_t1, title_prefix="TEST", save_plot=False)
        plot_rollout_3d(res_tn, title_prefix="TEST", save_plot=False)

    if state_dim == 4:
        plot_rollout_4d_color(res_t1)
        plot_rollout_4d_color(res_tn)

    logger.info("Evaluation done.")


def plot_logs_main() -> None:
    if not os.path.exists(LOG_PATH):
        raise FileNotFoundError(f"Log file not found: {LOG_PATH}")
    records = load_jsonl(LOG_PATH)
    plot_all(records, save=False)


# =========================================================
# Main dispatch
# =========================================================
if __name__ == "__main__":
    if MODE == "build":
        build_main()

    elif MODE == "train":
        train_main(rebuild_data=REBUILD_DATA)

    elif MODE == "eval":
        eval_main()
        plot_logs_main()

    else:
        raise ValueError(f"Unknown MODE: {MODE}")