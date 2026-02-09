from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt


# -----------------------------
# helpers: io / timestamp
# -----------------------------
def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(p: str) -> Path:
    pp = Path(p)
    pp.mkdir(parents=True, exist_ok=True)
    return pp


# -----------------------------
# helpers: predict one step (window -> next)
# -----------------------------
@torch.no_grad()
def predict_next_state_phys(
    model: torch.nn.Module,
    scaler,
    window_phys: np.ndarray,          # (K, D) physical
    device: torch.device,
) -> np.ndarray:
    """
    Standard usage:
      physical window -> normalize -> model -> inverse -> physical next state
    """
    if window_phys.ndim != 2:
        raise ValueError(f"window_phys must be (K,D), got {window_phys.shape}")

    win_norm = scaler.transform(window_phys.astype(np.float32, copy=False))       # (K,D)
    x = torch.from_numpy(win_norm).unsqueeze(0).to(device)                        # (1,K,D)
    y_next_norm = model(x).squeeze(0).detach().cpu().float().numpy()              # (D,)
    y_next_phys = scaler.inverse_transform(y_next_norm[None, :])[0]               # (D,)
    return y_next_phys.astype(np.float32, copy=False)


# -----------------------------
# t+1: teacher forcing rollout
# -----------------------------
@torch.no_grad()
def rollout_t_plus_1(
    model: torch.nn.Module,
    scaler,
    states_phys: np.ndarray,          # (T, D) physical
    *,
    window_size: int,
    start_t: int,
    length: int,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Teacher forcing:
      at each step k:
        input window = true states_phys[start_t+k : start_t+k+K]
        pred         = next state prediction
        true         = states_phys[start_t+k+K]
    Output aligned to target timeline (start_t+K ... start_t+K+L-1)
    """
    model.eval()
    states_phys = np.asarray(states_phys, dtype=np.float32)
    T, D = states_phys.shape
    K = int(window_size)
    L = int(length)

    max_needed = start_t + (L - 1) + K
    if start_t < 0 or max_needed >= T:
        raise ValueError(f"Invalid range: need max index {max_needed} < T={T}")

    preds = np.zeros((L, D), dtype=np.float32)
    trues = np.zeros((L, D), dtype=np.float32)

    for k in range(L):
        i = start_t + k
        window = states_phys[i : i + K]
        preds[k] = predict_next_state_phys(model, scaler, window, device)
        trues[k] = states_phys[i + K]

    return {
        "mode": "t+1",
        "start_t": start_t + K,
        "window_size": K,
        "length": L,
        "true": trues,
        "pred": preds,
    }


# -----------------------------
# t+n: free rollout (closed-loop)
# -----------------------------
@torch.no_grad()
def rollout_t_plus_n(
    model: torch.nn.Module,
    scaler,
    states_phys: np.ndarray,          # (T, D) physical
    *,
    window_size: int,
    start_t: int,
    length: int,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Free rollout:
      first window is true physical
      then feed predicted physical next state back into the window
    """
    model.eval()
    states_phys = np.asarray(states_phys, dtype=np.float32)
    T, D = states_phys.shape
    K = int(window_size)
    L = int(length)

    max_needed = start_t + (L - 1) + K
    if start_t < 0 or max_needed >= T:
        raise ValueError(f"Invalid range: need max index {max_needed} < T={T}")

    window = states_phys[start_t : start_t + K].copy()  # (K,D)

    preds = np.zeros((L, D), dtype=np.float32)
    trues = np.zeros((L, D), dtype=np.float32)

    for k in range(L):
        preds[k] = predict_next_state_phys(model, scaler, window, device)
        trues[k] = states_phys[start_t + K + k]

        # feed predicted physical state
        window = np.vstack([window[1:], preds[k][None, :]])

    return {
        "mode": "t+n",
        "start_t": start_t + K,
        "window_size": K,
        "length": L,
        "true": trues,
        "pred": preds,
    }


# -----------------------------
# saving results
# -----------------------------
def save_rollout_result(
    result: Dict[str, Any],
    *,
    out_dir: str = "./eval_outputs",
    prefix: str = "test",
) -> str:
    """
    Saves as .npz for exact arrays + minimal metadata.
    """
    outp = _ensure_dir(out_dir)
    fn = f"{prefix}_{result['mode']}_start{result['start_t']}_K{result['window_size']}_L{result['length']}_{_ts()}.npz"
    path = str(outp / fn)

    np.savez_compressed(
        path,
        mode=result["mode"],
        start_t=int(result["start_t"]),
        window_size=int(result["window_size"]),
        length=int(result["length"]),
        true=result["true"],
        pred=result["pred"],
    )
    return path


def save_rollout_csv(
    result: Dict[str, Any],
    *,
    out_dir: str = "./eval_outputs",
    prefix: str = "test",
) -> str:
    """
    Saves a wide CSV:
      t, true_s0..true_s{D-1}, pred_s0..pred_s{D-1}
    """
    outp = _ensure_dir(out_dir)
    true = result["true"]
    pred = result["pred"]
    L, D = true.shape
    t0 = int(result["start_t"])
    t = np.arange(t0, t0 + L, dtype=np.int64)

    header = ["t"] + [f"true_s{d}" for d in range(D)] + [f"pred_s{d}" for d in range(D)]
    data = np.concatenate([t[:, None], true, pred], axis=1)

    fn = f"{prefix}_{result['mode']}_start{t0}_K{result['window_size']}_L{L}_{_ts()}.csv"
    path = str(outp / fn)
    np.savetxt(path, data, delimiter=",", header=",".join(header), comments="")
    return path


# -----------------------------
# plotting (overlay true vs pred over time)
# -----------------------------
def plot_rollout_overlay(
    result: Dict[str, Any],
    *,
    dims: Optional[List[int]] = None,
    max_points: Optional[int] = None,
    title_prefix: str = "TEST",
    save_plot: bool = False,
    plots_dir: str = "./plots",
) -> Optional[str]:
    true = np.asarray(result["true"], dtype=np.float32)
    pred = np.asarray(result["pred"], dtype=np.float32)
    L, D = true.shape
    t0 = int(result["start_t"])
    mode = str(result["mode"])

    if max_points is not None and L > max_points:
        true = true[:max_points]
        pred = pred[:max_points]
        L = true.shape[0]

    if dims is None:
        # default max 6 dim
        dims = list(range(min(D, 6)))

    t = np.arange(t0, t0 + L)

    plt.figure(figsize=(12, 2.8 * len(dims)))
    for i, d in enumerate(dims, start=1):
        ax = plt.subplot(len(dims), 1, i)
        ax.plot(t, true[:, d], linewidth=2, label=f"true dim {d}")
        ax.plot(t, pred[:, d], linestyle="--", label=f"pred dim {d}")
        ax.set_xlabel("time index")
        ax.set_ylabel(f"state[{d}]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    title = f"{title_prefix} {mode} overlay (start={t0}, L={L}, K={result['window_size']})"
    plt.suptitle(title, fontsize=14, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.tight_layout()

    if save_plot:
        outp = _ensure_dir(plots_dir)
        fn = f"overlay_{title_prefix.lower()}_{mode}_start{t0}_K{result['window_size']}_L{L}_{_ts()}.png"
        path = str(outp / fn)
        plt.savefig(path, dpi=200)
        plt.close()
        return path

    plt.show()
    return None
