# plotting.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
# -----------------------------
# Utilities
# -----------------------------

def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Load JSONL log file produced by main/trainer.
    Each line is a dict record.
    """
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _filter_split(records: Sequence[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
    out = [r for r in records if r.get("split") == split]
    out.sort(key=lambda r: int(r.get("epoch", 0)))
    return out


def _get_metric_keys(metric: str) -> Tuple[str, str]:
    m = metric.lower()

    if m == "mse":
        return "mse", "mse_per_dim"
    if m == "mae":
        return "mae", "mae_per_dim"
    if m == "rse":
        return "rse", "r2_per_dim"
    raise ValueError("metric must be 'mse' or 'mae' or 'rse' for loss plots.")


def _default_title(s: str) -> str:
    return s.replace("_", " ").title()


# -----------------------------
# Plot 1: Loss curves
# -----------------------------

def plot_loss_curves(
    records: Sequence[Dict[str, Any]],
    split: str,
    *,
    metric: str = "mse",
    save: bool = False,
    out_dir: str = "./plots",
    filename: Optional[str] = None,
    show: bool = True,
) -> Optional[str]:
    """
    One figure per split: overall loss + per-dim loss lines over epochs.

    Args:
      records: list of dicts (already computed stats)
      split: "train" | "val" | "test"
      metric: "mse" | "mae" | "rse" | "mse_per_dim" | "mae_per_dim" | "rse_per_dim"
      save: whether to save
      out_dir: default ./plots
      filename: override filename
      show: whether to plt.show() when not saving

    Returns:
      Saved filepath if save=True else None
    """
    rs = _filter_split(records, split)
    if not rs:
        raise ValueError(f"No records found for split='{split}'")

    overall_key, per_dim_key = _get_metric_keys(metric)

    epochs = [int(r["epoch"]) for r in rs]

    # overall series
    overall = [float(r[overall_key]) for r in rs]

    # per-dim series: list over epochs of list over dims
    per_dim_all = [r[per_dim_key] for r in rs]
    D = len(per_dim_all[0])
    for row in per_dim_all:
        if len(row) != D:
            raise ValueError(f"Inconsistent {per_dim_key} dimension in records for split={split}")

    # transpose to dim-major
    per_dim_series = [[float(per_dim_all[e][d]) for e in range(len(epochs))] for d in range(D)]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, overall, linewidth=2, label=f"overall_{metric}")
    for d in range(D):
        plt.plot(epochs, per_dim_series[d], label=f"dim_{d}_{metric}")

    plt.xlabel("epoch")
    plt.ylabel(metric)
    plt.title(f"{_default_title(split)} loss ({metric})")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)

    saved_path = None
    if save:
        outp = _ensure_dir(out_dir)
        if filename is None:
            filename = f"loss_{split}_{metric}_{_timestamp()}.png"
        saved_path = str(outp / filename)
        plt.tight_layout()
        plt.savefig(saved_path, dpi=200)
        plt.close()
        return saved_path

    plt.tight_layout()
    if show:
        plt.show()
    return None


# -----------------------------
# Plot 2: True-Pred fit (scatter)
# -----------------------------

def plot_true_pred_fit(
    records: Sequence[Dict[str, Any]],
    split: str,
    *,
    epoch: Optional[int] = None,
    max_points: Optional[int] = None,
    save: bool = False,
    out_dir: str = "./plots",
    filename: Optional[str] = None,
    show: bool = True,
) -> Optional[str]:
    """
    One figure per split: true vs pred scatter for each dim (subplot grid).
    Uses record fields:
      - sample_true: (N,D)
      - sample_pred: (N,D)

    Args:
      epoch: if None, uses the last epoch in that split
      max_points: optionally downsample points to avoid dense scatter
    """
    rs = _filter_split(records, split)
    if not rs:
        raise ValueError(f"No records found for split='{split}'")

    if epoch is None:
        rec = rs[-1]
        epoch = int(rec["epoch"])
    else:
        matches = [r for r in rs if int(r.get("epoch", -1)) == int(epoch)]
        if not matches:
            raise ValueError(f"No record found for split={split}, epoch={epoch}")
        rec = matches[0]

    sample_true = rec.get("sample_true", None)
    sample_pred = rec.get("sample_pred", None)
    if sample_true is None or sample_pred is None:
        raise ValueError("Records do not contain sample_true/sample_pred. Enable sampling in trainer.")

    # Convert to python lists (N,D)
    true_arr = sample_true
    pred_arr = sample_pred
    if len(true_arr) == 0:
        raise ValueError("sample_true is empty; increase max_pairs in trainer/eval logging.")

    N = len(true_arr)
    D = len(true_arr[0])

    # Optional downsample
    if max_points is not None and N > max_points:
        true_arr = true_arr[:max_points]
        pred_arr = pred_arr[:max_points]
        N = len(true_arr)

    # Determine subplot grid (roughly square)

    cols = int(math.ceil(math.sqrt(D)))
    rows = int(math.ceil(D / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    # axes could be scalar if rows=cols=1
    if not isinstance(axes, (list, tuple)) and hasattr(axes, "shape"):
        axes_list = axes.ravel().tolist()
    else:
        axes_list = axes  # type: ignore

    # flatten robustly
    try:
        axes_flat = list(fig.axes)
    except Exception:
        axes_flat = axes_list  # fallback

    # For each dim: scatter + y=x line
    for d in range(D):
        ax = axes_flat[d]
        xs = [float(true_arr[i][d]) for i in range(N)]
        ys = [float(pred_arr[i][d]) for i in range(N)]

        ax.scatter(xs, ys, s=12, alpha=0.6)

        # y=x reference line
        mn = min(min(xs), min(ys))
        mx = max(max(xs), max(ys))
        ax.plot([mn, mx], [mn, mx], linewidth=1)

        ax.set_title(f"{split} true vs pred (dim {d})")
        ax.set_xlabel("true")
        ax.set_ylabel("pred")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for k in range(D, len(axes_flat)):
        axes_flat[k].axis("off")

    fig.suptitle(f"{_default_title(split)} true-pred fit (epoch {epoch}, N={N})", y=1.02)
    fig.tight_layout()

    saved_path = None
    if save:
        outp = _ensure_dir(out_dir)
        if filename is None:
            filename = f"fit_{split}_epoch{epoch}_{_timestamp()}.png"
        saved_path = str(outp / filename)
        fig.savefig(saved_path, dpi=200)
        plt.close(fig)
        return saved_path

    if show:
        plt.show()
    return None


# -----------------------------
# Convenience: plot all requested figures
# -----------------------------
def plot_metric_curves(
    records: Sequence[Dict[str, Any]],
    split: str,
    *,
    metric: str,
    save: bool = False,
    out_dir: str = "./plots",
    filename: Optional[str] = None,
    show: bool = True,
) -> Optional[str]:
    """
    Plot overall + per-dim curves over epochs for any metric that has
    (overall_key, per_dim_key). If overall_key is missing, it will be
    derived as mean(per_dim_key).

    Supported metrics (if present in records):
      - mse, mae, rse, r2
    """
    rs = _filter_split(records, split)
    if not rs:
        raise ValueError(f"No records found for split='{split}'")

    overall_key, per_dim_key = _get_metric_keys(metric)

    # Must have per-dim
    if per_dim_key not in rs[0]:
        raise ValueError(f"Records do not contain '{per_dim_key}' for metric='{metric}'")

    epochs = [int(r["epoch"]) for r in rs]

    per_dim_all = [r[per_dim_key] for r in rs]
    D = len(per_dim_all[0])
    for row in per_dim_all:
        if len(row) != D:
            raise ValueError(f"Inconsistent {per_dim_key} dimension in records for split={split}")

    # overall series: use stored overall if available; else mean(per-dim)
    if overall_key in rs[0]:
        overall = [float(r[overall_key]) for r in rs]
        overall_label = f"overall_{metric}"
    else:
        overall = [float(sum(map(float, per_dim_all[e])) / D) for e in range(len(epochs))]
        overall_label = f"mean_dim_{metric}"

    # transpose per-dim
    per_dim_series = [[float(per_dim_all[e][d]) for e in range(len(epochs))] for d in range(D)]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, overall, linewidth=2, label=overall_label)
    for d in range(D):
        plt.plot(epochs, per_dim_series[d], label=f"dim_{d}_{metric}")

    plt.xlabel("epoch")
    plt.ylabel(metric)
    plt.title(f"{_default_title(split)} metric ({metric})")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)

    if save:
        outp = _ensure_dir(out_dir)
        if filename is None:
            filename = f"metric_{metric}_{split}_{_timestamp()}.png"
        saved_path = str(outp / filename)
        plt.tight_layout()
        plt.savefig(saved_path, dpi=200)
        plt.close()
        return saved_path

    plt.tight_layout()
    if show:
        plt.show()
    return None


def plot_all(
    records: Sequence[Dict[str, Any]],
    *,
    save: bool = False,
    out_dir: str = "./plots",
    show: bool = True,
) -> List[str]:
    """
    Generates, for each split:
      - metric curves for all available metrics (mse/mae/rse/r2 if present)
      - true-pred fit for train/val/test

    Returns list of saved file paths (empty if save=False).
    """
    if not records:
        raise ValueError("records is empty")

    saved: List[str] = []

    # Detect what metrics exist (prefer checking for per-dim keys)
    example = records[0]
    metrics: List[str] = []

    if "mse_per_dim" in example and "mse" in example:
        metrics.append("mse")
    if "mae_per_dim" in example:
        metrics.append("mae")
    if "r2_per_dim" in example:
        metrics.append("rse")


    if not metrics:
        raise ValueError("No known metrics found in records (expected mse/mae/rse/r2 keys).")

    for split in ("train", "val", "test"):
        # 1) metric curves (overall + per-dim) for all metrics
        for metric in metrics:
            p = plot_metric_curves(
                records,
                split,
                metric=metric,
                save=save,
                out_dir=out_dir,
                show=show and not save,
            )
            if p:
                saved.append(p)

        # 2) true-pred fit (scatter) for this split
        p_fit = plot_true_pred_fit(
            records,
            split,
            save=save,
            out_dir=out_dir,
            show=show and not save,
        )
        if p_fit:
            saved.append(p_fit)

    return saved
