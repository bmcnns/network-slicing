# main.py
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple
import logging
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from StatePredictor.eval_rollout import rollout_t_plus_1, rollout_t_plus_n, save_rollout_result, save_rollout_csv, \
    plot_rollout_overlay, plot_rollout_3d, plot_rollout_4d_color
from plotting import load_jsonl, plot_all
from data_builder import TrigonometricDataBuilder, DatasetSplits, Standardizer, \
    SimulatorDataBuilder, ChaosDataBuilder, CSVDataBuilder

from model import LSTMStatePredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger("train")


# -----------------------------
# Metrics
# -----------------------------
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


# -----------------------------
# Record structure (JSONL)
# -----------------------------
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


# -----------------------------
# Train / Eval helpers
# -----------------------------
def run_train_epoch(
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        loss_fn: nn.Module,
        epoch: int,
        log_every: int = 20,  # 每 N 个 batch 打一次
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

    y_pred_all = torch.cat(preds, dim=0)  # (N,D)
    y_true_all = torch.cat(trues, dim=0)  # (N,D)

    # overall MSE
    mse = float(((y_pred_all - y_true_all) ** 2).mean().detach().cpu().item())

    mse_d = mse_per_dim(y_pred_all, y_true_all)
    mae_d = mae_per_dim(y_pred_all, y_true_all)
    r2_d = r2_per_dim(y_pred_all, y_true_all)

    rse_d = 1.0 - r2_d  # (D,)
    rse = float(rse_d.mean().cpu().item())  # overall r2 error

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
    """
    CSV 有表头，且只有 4 列纯数值 state。
    行号 = 时间索引，不需要额外 time 列。
    """
    states = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != 4:
        raise ValueError(f"Expected (T,4) after header, got {states.shape}")
    return states

# -----------------------------
# Main
# -----------------------------
def main():
    # ---------- Config ----------
    CSV_PATH = "data/agent.csv"  # 改这里random_policy.csv
    seed = 0
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device:{device}")

    # Data config
    num_steps = 20_000
    mode = "sin"  # "sin" | "cos" | "both" only for trig dataset for testing the model
    window_size = 32
    stride = 1
    noise_std = 1
    split = (0.7, 0.15, 0.15)
    standardize = True

    # Train config
    epochs = 1000
    batch_size = 128
    lr = 1e-3
    max_pairs = 16
    log_path = "log/train_log.jsonl"

    # Model config
    hidden_dim = 64 # 8
    num_layers = 1
    dropout = 0.0

    # ---------- Build data ----------
    # builder = TrigonometricDataBuilder(
    #     num_steps=num_steps,
    #     mode=mode,
    #     dt=0.01,
    #     window_size=window_size,
    #     stride=stride,
    #     noise_std=noise_std,
    #     split=split,
    #     seed=seed,
    #     standardize=standardize,
    # )
    # builder = SimulatorDataBuilder(
    #     num_steps=20_000,
    #     window_size=32,
    #     stride=1,
    #     split=(0.7, 0.15, 0.15),
    #     seed=0,
    #     standardize=True,
    #     policy="random",
    #     meta={"env": "NetworkSlicingEnv"},
    # )
    # builder = ChaosDataBuilder(
    #     num_steps=20000,
    #     dt=0.01,
    #     window_size=32,
    #     stride=1,
    #     split=(0.7, 0.15, 0.15),
    #     standardize=True,
    #     noise_std=noise_std,
    #     seed=0,
    # )
    builder = CSVDataBuilder(
        csv_path=CSV_PATH,
        window_size=window_size,
        stride=stride,
        split=split,
        seed=seed,
        standardize=standardize,
    )

    splits: DatasetSplits = builder.build()
    splits.save("data")
    input_dim = splits.raw.states.shape[1]
    logger.info(f"Raw states shape: {splits.raw.states.shape}  (T,D)")
    logger.info(f"Input dim D = {input_dim}")

    train_loader = DataLoader(splits.train, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(splits.val, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(splits.test, batch_size=batch_size, shuffle=False, num_workers=0)

    # ---------- Build model ----------
    model = LSTMStatePredictor(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Clear log
    open(log_path, "w", encoding="utf-8").close()
    logger.info(f"Logging to:{log_path}")

    # ---------- Train loop ----------

    for epoch in range(epochs):
        train_batch_mse = run_train_epoch(model, train_loader, optimizer, device, loss_fn, epoch)

        rec_train = eval_split(model, train_loader, device, "train", epoch, max_pairs=max_pairs)
        rec_val = eval_split(model, val_loader, device, "val", epoch, max_pairs=max_pairs)
        rec_test = eval_split(model, test_loader, device, "test", epoch, max_pairs=max_pairs)

        append_jsonl(log_path, rec_train)
        append_jsonl(log_path, rec_val)
        append_jsonl(log_path, rec_test)

        logger.info({
            "train_mse": f"{rec_train.mse:.6f}",
            "val_mse": f"{rec_val.mse:.6f}",
            "test_mse": f"{rec_test.mse:.6f}",
        })

        # 控制台也打印一行（可选）
        logger.info(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_mse={rec_train.mse:.6f} (batch_avg={train_batch_mse:.6f}) | "
            f"val_mse={rec_val.mse:.6f} | "
            f"test_mse={rec_test.mse:.6f}"
        )

    logger.info("Done.")

    # save LSTM model
    save_lstm_checkpoint(
        path="./model_components/lstm_dt.pt",
        model=model,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )

    # ---- build test_states_phys from raw continuous states ----
    states_phys = splits.raw.states  # (T, D) 期望是“物理空间”的原始连续序列

    T = states_phys.shape[0]
    n_train = int(split[0] * T)
    n_val = int(split[1] * T)

    train_states_phys = states_phys[:n_train]
    val_states_phys = states_phys[n_train:n_train + n_val]
    test_states_phys = states_phys[n_train + n_val:]

    logger.info(f"train_states_phys: {train_states_phys.shape}")
    logger.info(f"val_states_phys:   {val_states_phys.shape}")
    logger.info(f"test_states_phys:  {test_states_phys.shape}")

    # -----------------------------
    # Collect run config (for reproducibility)
    # -----------------------------
    run_cfg = {
        # ---- experiment ----
        "seed": seed,
        "device": str(device),

        # ---- data ----
        "num_steps": num_steps,
        "mode": mode,
        "dt": 0.01,
        "noise_std": noise_std,
        "split": split,
        "standardize": standardize,

        # ---- windowing ----
        "window_size": window_size,
        "stride": stride,

        # ---- model ----
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,

        # ---- training ----
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "loss": "MSE",

        # ---- artifacts ----
        "model_path": "./model_components/lstm_dt.pt",
        "log_path": log_path,
    }

    return splits, test_states_phys, run_cfg


if __name__ == "__main__":
    splits, test_states_phys, run_cfg = main()

    # ---- training curves ----
    records = load_jsonl(run_cfg["log_path"])
    plot_all(records, save=False)

    # ---- load model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(run_cfg["model_path"], map_location=device)
    cfg = ckpt["model_config"]

    model = LSTMStatePredictor(**cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ---- load standardizer ----
    scaler = Standardizer.load("./model_components/standardizer.json")

    # ---- rollout params ----
    K = run_cfg["window_size"]
    L = 10000
    start_t = 0

    # ---- t+1 / t+n ----
    res_t1 = rollout_t_plus_1(
        model, scaler, test_states_phys,
        window_size=K, start_t=start_t, length=L, device=device
    )
    res_tn = rollout_t_plus_n(
        model, scaler, test_states_phys,
        window_size=K, start_t=start_t, length=L, device=device
    )

    # ---- save + plot ----
    save_rollout_result(res_t1, out_dir="./eval_outputs", prefix="test")
    save_rollout_result(res_tn, out_dir="./eval_outputs", prefix="test")

    plot_rollout_overlay(res_t1, title_prefix="TEST", save_plot=False, smooth_window=20)
    plot_rollout_overlay(res_tn, title_prefix="TEST", save_plot=False, smooth_window=20)

    plot_rollout_3d(res_t1, title_prefix="TEST", save_plot=False)
    plot_rollout_3d(res_tn, title_prefix="TEST", save_plot=False)
    plot_rollout_4d_color(res_t1)
    plot_rollout_4d_color(res_tn)

