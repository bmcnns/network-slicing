# main.py
from __future__ import annotations

import json
import os
import logging
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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
from plotting import load_jsonl, plot_all
from data_builder import (
    TrigonometricDataBuilder,
    DatasetSplits,
    Standardizer,
    SimulatorDataBuilder,
    ChaosDataBuilder,
    CSVDataBuilder,
)
from model import LSTMStatePredictor

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("train")

# =========================================================
# Global Config
# =========================================================
MODE = "eval"  # "build" | "train" | "eval"
REBUILD_DATA = True  # only meaningful in "train"

DATA_DIR = "data/random_policy_1000steps"
LOG_PATH = "log/train_log.jsonl"
MODEL_PATH = "./model_components/lstm_dt.pt"
SCALER_PATH = "./model_components/standardizer.json"
EVAL_OUT_DIR = "./eval_outputs"

CSV_PATH = "data/agent.csv"
SEED = 0

# Data config
NUM_STEPS = 20_000
MODE_TRIG = "sin"  # "sin" | "cos" | "both"
WINDOW_SIZE = 32
STRIDE = 1
NOISE_STD = 1
SPLIT = (0.7, 0.15, 0.15)
STANDARDIZE = True

# Train config
EPOCHS = 5000
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
# Metrics
# =========================================================
def mse_per_dim(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return ((y_pred - y_true) ** 2).mean(dim=0)


def mae_per_dim(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return (y_pred - y_true).abs().mean(dim=0)


def r2_per_dim(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    ss_res = ((y_true - y_pred) ** 2).sum(dim=0)
    mean_true = y_true.mean(dim=0, keepdim=True)
    ss_tot = ((y_true - mean_true) ** 2).sum(dim=0)
    return 1.0 - ss_res / (ss_tot + eps)


def to_list(x: torch.Tensor) -> List[float]:
    return x.detach().cpu().float().tolist()


# =========================================================
# Record structure (JSONL)
# =========================================================
@dataclass
class SplitEpochRecord:
    split: str
    epoch: int

    mse: float
    mse_per_dim: List[float]
    mae_per_dim: List[float]
    r2_per_dim: List[float]
    rse: float

    true_mean: List[float]
    pred_mean: List[float]

    sample_true: List[List[float]]
    sample_pred: List[List[float]]


# =========================================================
# Train / Eval helpers
# =========================================================
def run_train_epoch(
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        loss_fn: nn.Module,
        epoch: int,
        log_every: int = 20,
) -> float:
    model.train()
    total = 0.0
    n_batches = 0

    logger.info(f"Train epoch {epoch} start")

    for step, (x_seq, y_true) in enumerate(loader, start=1):
        x_seq = x_seq.to(device)
        y_true = y_true.to(device)

        optimizer.zero_grad(set_to_none=True)
        y_pred = model(x_seq)
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        optimizer.step()

        total += float(loss.detach().cpu().item())
        n_batches += 1

        if step % log_every == 0:
            logger.info(
                f"Train epoch {epoch} | step {step}/{len(loader)} | "
                f"batch_mse={total / n_batches:.6f}"
            )

    avg = total / max(n_batches, 1)
    logger.info(f"Train epoch {epoch} done | avg_mse={avg:.6f}")
    return avg


@torch.no_grad()
def eval_split(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        split: str,
        epoch: int,
        max_pairs: int = 16,
) -> SplitEpochRecord:
    model.eval()
    preds = []
    trues = []

    for x_seq, y_true in loader:
        x_seq = x_seq.to(device)
        y_true = y_true.to(device)
        y_pred = model(x_seq)

        preds.append(y_pred)
        trues.append(y_true)

    y_pred_all = torch.cat(preds, dim=0)
    y_true_all = torch.cat(trues, dim=0)

    mse = float(((y_pred_all - y_true_all) ** 2).mean().detach().cpu().item())

    mse_d = mse_per_dim(y_pred_all, y_true_all)
    mae_d = mae_per_dim(y_pred_all, y_true_all)
    r2_d = r2_per_dim(y_pred_all, y_true_all)

    rse_d = 1.0 - r2_d
    rse = float(rse_d.mean().cpu().item())

    true_mean = y_true_all.mean(dim=0)
    pred_mean = y_pred_all.mean(dim=0)

    n = min(max_pairs, y_true_all.shape[0])
    sample_true = y_true_all[:n].detach().cpu().float().tolist()
    sample_pred = y_pred_all[:n].detach().cpu().float().tolist()

    return SplitEpochRecord(
        split=split,
        epoch=epoch,
        mse=mse,
        mse_per_dim=to_list(mse_d),
        mae_per_dim=to_list(mae_d),
        rse=rse,
        r2_per_dim=to_list(r2_d),
        true_mean=to_list(true_mean),
        pred_mean=to_list(pred_mean),
        sample_true=sample_true,
        sample_pred=sample_pred,
    )


def append_jsonl(path: str, record: SplitEpochRecord) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record)) + "\n")


def save_lstm_checkpoint(
        path: str,
        model: torch.nn.Module,
        *,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    ckpt = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_dim": int(input_dim),
            "hidden_dim": int(hidden_dim),
            "num_layers": int(num_layers),
            "dropout": float(dropout),
        },
    }
    torch.save(ckpt, path)
    logger.info(f"[INFO] Saved LSTM checkpoint to: {path}")


def load_csv_4d(csv_path: str) -> np.ndarray:
    states = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != 4:
        raise ValueError(f"Expected (T,4) after header, got {states.shape}")
    return states


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


def make_loaders(splits: DatasetSplits) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(splits.train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(splits.val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(splits.test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


def build_model(input_dim: int, device: torch.device) -> nn.Module:
    # input dim is state dim + action dim
    model = LSTMStatePredictor(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)
    return model


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
    input_dim = splits.raw.states.shape[1]
    logger.info(f"Input dim D = {input_dim}")

    train_loader, val_loader, test_loader = make_loaders(splits)

    model = build_model(input_dim=input_dim, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
    open(LOG_PATH, "w", encoding="utf-8").close()
    logger.info(f"Logging to: {LOG_PATH}")

    for epoch in range(EPOCHS):
        train_batch_mse = run_train_epoch(
            model, train_loader, optimizer, device, loss_fn, epoch
        )

        rec_train = eval_split(model, train_loader, device, "train", epoch, max_pairs=MAX_PAIRS)
        rec_val = eval_split(model, val_loader, device, "val", epoch, max_pairs=MAX_PAIRS)
        rec_test = eval_split(model, test_loader, device, "test", epoch, max_pairs=MAX_PAIRS)

        append_jsonl(LOG_PATH, rec_train)
        append_jsonl(LOG_PATH, rec_val)
        append_jsonl(LOG_PATH, rec_test)

        logger.info(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"train_mse={rec_train.mse:.6f} (batch_avg={train_batch_mse:.6f}) | "
            f"val_mse={rec_val.mse:.6f} | "
            f"test_mse={rec_test.mse:.6f}"
        )

    logger.info("Training done.")

    save_lstm_checkpoint(
        path=MODEL_PATH,
        model=model,
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )

    run_cfg = make_run_cfg(input_dim=input_dim, device=device)
    os.makedirs("./model_components", exist_ok=True)
    with open("./model_components/run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, indent=2, ensure_ascii=False)

    logger.info("Training artifacts saved.")


def eval_main() -> None:
    device = get_device()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    states_path = os.path.join(DATA_DIR, "states.npy")
    if not os.path.exists(states_path):
        raise FileNotFoundError(f"Cached states not found: {states_path}")

    ckpt = torch.load(MODEL_PATH, map_location=device)
    cfg = ckpt["model_config"]

    model = LSTMStatePredictor(**cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

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

    plot_rollout_3d(res_t1, title_prefix="TEST", save_plot=False)
    plot_rollout_3d(res_tn, title_prefix="TEST", save_plot=False)

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
