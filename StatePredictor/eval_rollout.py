from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

    Behavior change:
      If requested length exceeds available range, L is clipped automatically.
    """
    model.eval()
    states_phys = np.asarray(states_phys, dtype=np.float32)
    T, D = states_phys.shape
    K = int(window_size)

    if start_t < 0:
        raise ValueError("start_t must be >= 0")
    if start_t + K >= T:
        raise ValueError(f"start_t+K must be < T. Got start_t={start_t}, K={K}, T={T}")

    # Max L such that (start_t + (L-1) + K) < T  => L <= T - start_t - K
    L_max = T - start_t - K
    if L_max <= 0:
        raise ValueError(f"No room for any prediction: L_max={L_max}. Check start_t/window_size.")

    L_req = int(length)
    L = min(L_req, L_max)

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
        "length_requested": L_req,
        "length_max": int(L_max),
        "clipped": bool(L < L_req),
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
    Free rollout (closed-loop):
      - first window is true physical
      - then feed predicted physical next state back into the window

    Behavior change:
      If requested length exceeds available 'true' range, L is clipped automatically.
      (Because we still return aligned ground-truth `true`.)
    """
    model.eval()
    states_phys = np.asarray(states_phys, dtype=np.float32)
    T, D = states_phys.shape
    K = int(window_size)

    if start_t < 0:
        raise ValueError("start_t must be >= 0")
    if start_t + K >= T:
        raise ValueError(f"start_t+K must be < T. Got start_t={start_t}, K={K}, T={T}")

    # Need true at indices: start_t + K + k, for k=0..L-1
    # => start_t + K + (L-1) < T  => L <= T - start_t - K
    L_max = T - start_t - K
    if L_max <= 0:
        raise ValueError(f"No room for any rollout: L_max={L_max}. Check start_t/window_size.")

    L_req = int(length)
    L = min(L_req, L_max)

    window = states_phys[start_t : start_t + K].copy()  # (K,D)

    preds = np.zeros((L, D), dtype=np.float32)
    trues = np.zeros((L, D), dtype=np.float32)

    for k in range(L):
        preds[k] = predict_next_state_phys(model, scaler, window, device)
        trues[k] = states_phys[start_t + K + k]
        window = np.vstack([window[1:], preds[k][None, :]])

    return {
        "mode": "t+n",
        "start_t": start_t + K,
        "window_size": K,
        "length": L,
        "length_requested": L_req,
        "length_max": int(L_max),
        "clipped": bool(L < L_req),
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
    smooth_window: Optional[int] = None,
    title_prefix: str = "TEST",
    save_plot: bool = False,
    plots_dir: str = "./plots",
) -> Optional[str]:

    def _smooth(x: np.ndarray, k: int) -> np.ndarray:
        if k <= 1:
            return x
        kernel = np.ones(k) / k
        return np.convolve(x, kernel, mode="same")

    pred = np.asarray(result["pred"], dtype=np.float32)
    true_raw = result.get("true", None)

    if true_raw is not None:
        true = np.asarray(true_raw, dtype=np.float32)
        if true.shape != pred.shape:
            raise ValueError("true and pred must have same shape")
    else:
        true = None

    L, D = pred.shape
    t0 = int(result["start_t"])
    mode = str(result["mode"])

    # Optional truncation
    if max_points is not None and L > max_points:
        pred = pred[:max_points]
        if true is not None:
            true = true[:max_points]
        L = pred.shape[0]

    if dims is None:
        dims = list(range(min(D, 6)))

    t = np.arange(t0, t0 + L)

    plt.figure(figsize=(12, 2.8 * len(dims)))

    for i, d in enumerate(dims, start=1):
        ax = plt.subplot(len(dims), 1, i)

        pred_dim = pred[:, d]
        true_dim = true[:, d] if true is not None else None

        # Apply smoothing if requested
        if smooth_window is not None and smooth_window > 1:
            pred_dim = _smooth(pred_dim, smooth_window)
            if true_dim is not None:
                true_dim = _smooth(true_dim, smooth_window)

        # Plot true
        if true_dim is not None:
            ax.plot(
                t,
                true_dim,
                linewidth=2,
                label=f"true dim {d}",
            )

        # Plot pred
        ax.plot(
            t,
            pred_dim,
            linestyle="--",
            label=f"pred dim {d}",
        )

        ax.set_xlabel("time index")
        ax.set_ylabel(f"state[{d}]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    # Title
    smooth_text = f", smooth={smooth_window}" if smooth_window else ""
    if true is None:
        title = f"{title_prefix} {mode} overlay (pred only, start={t0}, L={L}, K={result['window_size']}{smooth_text})"
    else:
        title = f"{title_prefix} {mode} overlay (start={t0}, L={L}, K={result['window_size']}{smooth_text})"

    plt.suptitle(title, fontsize=14, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_plot:
        outp = _ensure_dir(plots_dir)
        fn = f"overlay_{title_prefix.lower()}_{mode}_{_ts()}.png"
        path = str(outp / fn)
        plt.savefig(path, dpi=200)
        plt.close()
        return path

    plt.show()
    return None



def plot_rollout_3d(
    result: Dict[str, Any],
    *,
    save_plot: bool = False,
    plots_dir: str = "./plots",
    title_prefix: str = "TEST",
) -> Optional[str]:
    """
    Plot 3D trajectories:
        - True trajectory
        - Predicted trajectory
    Works only when D >= 3
    """
    true = np.asarray(result["true"], dtype=np.float32)
    pred = np.asarray(result["pred"], dtype=np.float32)

    if true.shape[1] < 3:
        raise ValueError("3D plot requires D >= 3")

    mode = result["mode"]
    t0 = result["start_t"]
    L = result["length"]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        true[:, 0],
        true[:, 1],
        true[:, 2],
        linewidth=1,
        label="True",
    )

    ax.plot(
        pred[:, 0],
        pred[:, 1],
        pred[:, 2],
        linestyle="--",
        linewidth=1,
        label="Pred",
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"{title_prefix} {mode} 3D Trajectory (start={t0}, L={L})")

    ax.legend()
    plt.tight_layout()

    if save_plot:
        outp = _ensure_dir(plots_dir)
        fn = f"trajectory3d_{title_prefix.lower()}_{mode}_{_ts()}.png"
        path = str(outp / fn)
        plt.savefig(path, dpi=200)
        plt.close()
        return path

    plt.show()
    return None

def plot_rollout_4d_color(result):
    true = np.asarray(result["true"], dtype=np.float32)

    if true.shape[1] < 4:
        raise ValueError("4D plot requires D >= 4")

    x, y, z, w = true[:, 0], true[:, 1], true[:, 2], true[:, 3]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(x, y, z, c=w, cmap="viridis", s=5)

    fig.colorbar(sc, ax=ax, label="W (4th dimension)")
    ax.set_title("4D Trajectory (Color Encoded)")
    plt.show()