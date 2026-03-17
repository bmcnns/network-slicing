from __future__ import annotations

import json
import os
import logging
from dataclasses import asdict, dataclass
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from StatePredictor.data_builder import DatasetSplits
from StatePredictor.model import LSTMStatePredictor

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("train")


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


def make_loaders(
    splits: DatasetSplits,
    *,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(splits.train, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(splits.val, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(splits.test, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


def build_model(
    *,
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    device: torch.device,
) -> nn.Module:
    model = LSTMStatePredictor(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    return model


def train_from_splits(
    *,
    splits: DatasetSplits,
    device: torch.device,
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    log_path: str,
    model_path: str,
    run_config_path: str,
    max_pairs: int = 16,
    extra_run_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    train_loader, val_loader, test_loader = make_loaders(
        splits,
        batch_size=batch_size,
    )

    model = build_model(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        device=device,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    open(log_path, "w", encoding="utf-8").close()
    logger.info(f"Logging to: {log_path}")

    for epoch in range(epochs):
        train_batch_mse = run_train_epoch(
            model, train_loader, optimizer, device, loss_fn, epoch
        )

        rec_train = eval_split(model, train_loader, device, "train", epoch, max_pairs=max_pairs)
        rec_val = eval_split(model, val_loader, device, "val", epoch, max_pairs=max_pairs)
        rec_test = eval_split(model, test_loader, device, "test", epoch, max_pairs=max_pairs)

        append_jsonl(log_path, rec_train)
        append_jsonl(log_path, rec_val)
        append_jsonl(log_path, rec_test)

        logger.info(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_mse={rec_train.mse:.6f} (batch_avg={train_batch_mse:.6f}) | "
            f"val_mse={rec_val.mse:.6f} | "
            f"test_mse={rec_test.mse:.6f}"
        )

    logger.info("Training done.")

    save_lstm_checkpoint(
        path=model_path,
        model=model,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )

    run_cfg = {
        "device": str(device),
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "loss": "MSE",
        "model_path": model_path,
        "log_path": log_path,
    }

    if extra_run_cfg is not None:
        run_cfg.update(extra_run_cfg)

    os.makedirs(os.path.dirname(run_config_path) or ".", exist_ok=True)
    with open(run_config_path, "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, indent=2, ensure_ascii=False)

    logger.info("Training artifacts saved.")

    return model, run_cfg


def load_trained_model(
    *,
    model_path: str,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, Any]]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    ckpt = torch.load(model_path, map_location=device)
    cfg = ckpt["model_config"]

    model = LSTMStatePredictor(**cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, cfg