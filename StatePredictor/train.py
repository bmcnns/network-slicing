# trainer_min.py
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.regression import RelativeSquaredError

# -----------------------------
# 1) Utils: metrics
# -----------------------------

def mse_per_dim(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Per-dimension Mean Squared Error
    Returns: (D,)
    """
    return ((y_pred - y_true) ** 2).mean(dim=0)


def mae_per_dim(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Per-dimension Mean Absolute Error
    Returns: (D,)
    """
    return (y_pred - y_true).abs().mean(dim=0)


def r2_per_dim(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Per-dimension R^2 score.

    R^2 = 1 - SS_res / SS_tot

    Returns: (D,)
    """
    # residual sum of squares
    ss_res = ((y_true - y_pred) ** 2).sum(dim=0)

    # total sum of squares
    mean_true = y_true.mean(dim=0, keepdim=True)
    ss_tot = ((y_true - mean_true) ** 2).sum(dim=0)

    return 1.0 - ss_res / (ss_tot + eps)


def metric_per_dim(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    metric: str = "mse",
) -> torch.Tensor:
    """
    Unified per-dimension metric selector.

    metric:
      - "mse"
      - "mae"
      - "r2"
    """
    metric = metric.lower()

    if metric == "mse":
        return mse_per_dim(y_pred, y_true)
    elif metric == "mae":
        return mae_per_dim(y_pred, y_true)
    elif metric in {"r2", "r_squared", "r2_score"}:
        return r2_per_dim(y_pred, y_true)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def to_list(x: torch.Tensor) -> List[float]:
    return x.detach().cpu().float().tolist()


# -----------------------------
# 2) Records
# -----------------------------

@dataclass
class SplitEpochRecord:
    split: str                 # train/val/test
    epoch: int
    loss_mse: float            # overall MSE (scalar)
    mse_per_dim: List[float]   # (D,)
    mae_per_dim: List[float]   # (D,)

    # true/pred summary (means across all samples)
    true_mean: List[float]     # (D,)
    pred_mean: List[float]     # (D,)

    # A small sample of true/pred pairs for debugging (first N items)
    sample_true: List[List[float]]  # (N, D)
    sample_pred: List[List[float]]  # (N, D)


# -----------------------------
# 3) One pass: train or eval
# -----------------------------

@torch.no_grad()
def _collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_samples_for_pairs: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collect all preds/targets across loader (for metrics + means).
    Returns:
      y_pred_all: (N, D)
      y_true_all: (N, D)
    """
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

    return y_pred_all, y_true_all


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    loss_fn: nn.Module,
) -> float:
    """
    If optimizer is provided -> train mode, else eval-only pass.
    Returns mean loss over batches.
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    n_batches = 0

    for x_seq, y_true in loader:
        x_seq = x_seq.to(device)
        y_true = y_true.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        y_pred = model(x_seq)
        loss = loss_fn(y_pred, y_true)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach().cpu().item())
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate_and_record(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    split: str,
    epoch: int,
    loss_fn: nn.Module,
    max_pairs: int = 32,
) -> SplitEpochRecord:
    """
    Compute:
      - overall mse
      - mse/mae per dim
      - true/pred mean vectors
      - a small sample of true/pred pairs
    """
    y_pred_all, y_true_all = _collect_predictions(model, loader, device)

    loss_mse = float(loss_fn(y_pred_all, y_true_all).detach().cpu().item())
    mse_d = mse_per_dim(y_pred_all, y_true_all)
    mae_d = mae_per_dim(y_pred_all, y_true_all)

    true_mean = y_true_all.mean(dim=0)
    pred_mean = y_pred_all.mean(dim=0)

    n = min(max_pairs, y_true_all.shape[0])
    sample_true = y_true_all[:n].detach().cpu().float().tolist()
    sample_pred = y_pred_all[:n].detach().cpu().float().tolist()

    return SplitEpochRecord(
        split=split,
        epoch=epoch,
        loss_mse=loss_mse,
        mse_per_dim=to_list(mse_d),
        mae_per_dim=to_list(mae_d),
        true_mean=to_list(true_mean),
        pred_mean=to_list(pred_mean),
        sample_true=sample_true,
        sample_pred=sample_pred,
    )


# -----------------------------
# 4) Main training loop
# -----------------------------

def train_simple(
    model: nn.Module,
    train_ds,
    val_ds,
    test_ds,
    *,
    device: str = "cpu",
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    num_workers: int = 0,
    log_path: str = "train_log.jsonl",
    max_pairs: int = 32,
) -> List[SplitEpochRecord]:
    """
    Simple training pipeline:
      - trains on train_ds
      - records train/val/test metrics each epoch
      - writes JSONL log (one record per split per epoch)

    Returns list of records in memory too.
    """
    dev = torch.device(device)
    model.to(dev)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    records: List[SplitEpochRecord] = []

    # overwrite log
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("")

    for epoch in range(1, epochs + 1):
        # 1) train update pass (loss only)
        train_loss = run_one_epoch(model, train_loader, optimizer, dev, loss_fn)

        # 2) detailed records for each split (true/pred + per-dim)
        rec_train = evaluate_and_record(model, train_loader, dev, "train", epoch, loss_fn, max_pairs=max_pairs)
        rec_val = evaluate_and_record(model, val_loader, dev, "val", epoch, loss_fn, max_pairs=max_pairs)
        rec_test = evaluate_and_record(model, test_loader, dev, "test", epoch, loss_fn, max_pairs=max_pairs)

        # (optional) keep the simple train pass loss too
        # Overwrite rec_train.loss_mse? no, keep both if you want
        # For now we keep rec_train.loss_mse (computed from full set).
        # If you need batch-avg training loss: print train_loss.
        print(
            f"Epoch {epoch:03d} | "
            f"train_mse={rec_train.loss_mse:.6f} (batch_avg={train_loss:.6f}) | "
            f"val_mse={rec_val.loss_mse:.6f} | "
            f"test_mse={rec_test.loss_mse:.6f}"
        )

        # 3) write JSONL
        with open(log_path, "a", encoding="utf-8") as f:
            for rec in (rec_train, rec_val, rec_test):
                f.write(json.dumps(asdict(rec)) + "\n")

        records.extend([rec_train, rec_val, rec_test])

    return records
