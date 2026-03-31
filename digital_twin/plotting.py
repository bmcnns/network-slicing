from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from digital_twin.utilities import ensure_dirs, load_json


def load_npz_dict(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path)
    return {k: data[k] for k in data.files}


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if window is None or window <= 1 or x.size < window:
        return x
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(x, kernel, mode="same")


def _fit_line(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size < 2:
        return 0.0, float(y[0]) if y.size == 1 else 0.0

    if np.allclose(x, x[0]):
        return 0.0, float(np.mean(y))

    if np.allclose(y, y[0]):
        return 0.0, float(y[0])

    try:
        a, b = np.polyfit(x, y, deg=1)
        return float(a), float(b)
    except Exception:
        return 0.0, float(np.mean(y))


def _format_rollout_tag(tag: str) -> str:
    tag_lower = tag.lower()
    if "t1" in tag_lower:
        return "T+1 Rollout"
    if "tn" in tag_lower:
        return "T+N Rollout"
    return tag.replace("_", " ").title()


def plot_sequence_training_loss(
    history_path: str,
    save_dir: str = "sequence_plots",
    show: bool = False,
) -> dict:
    ensure_dirs(save_dir)
    history = load_json(history_path)

    out = {}

    state_train = np.asarray(history["state_train_loss"], dtype=np.float32)
    state_val = np.asarray(history["state_val_loss"], dtype=np.float32)
    epochs = np.arange(1, len(state_train) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, state_train, label="Train State MSE")
    plt.plot(epochs, state_val, label="Validation State MSE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("State Prediction Training Loss")
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(save_dir, "sequence_state_training_loss.png")
    plt.savefig(save_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close()
    out["state_loss"] = save_path

    reward_train = np.asarray(history["reward_train_loss"], dtype=np.float32)
    reward_val = np.asarray(history["reward_val_loss"], dtype=np.float32)
    epochs = np.arange(1, len(reward_train) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, reward_train, label="Train Reward MSE")
    plt.plot(epochs, reward_val, label="Validation Reward MSE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Reward Prediction Training Loss")
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(save_dir, "sequence_reward_training_loss.png")
    plt.savefig(save_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close()
    out["reward_loss"] = save_path

    if "state_per_dim" in history and len(history["state_per_dim"]) > 0:
        state_per_dim = np.asarray(history["state_per_dim"], dtype=np.float32)
        epochs = np.arange(1, state_per_dim.shape[0] + 1)
        state_dim = state_per_dim.shape[1]

        plt.figure(figsize=(9, 5))
        for d in range(state_dim):
            plt.plot(epochs, state_per_dim[:, d], label=f"State Dim {d}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("State Prediction Loss by Dimension")
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(save_dir, "sequence_state_per_dim_loss.png")
        plt.savefig(save_path, dpi=200)
        if show:
            plt.show()
        else:
            plt.close()
        out["state_per_dim"] = save_path

    return out


def plot_rollout_scatter_combined(
    rollout_npz_path: str,
    save_dir: str = "sequence_plots",
    save_name: str = "sequence_rollout_scatter_combined.png",
    title_prefix: Optional[str] = None,
    show: bool = False,
) -> str:
    ensure_dirs(save_dir)
    result = load_npz_dict(rollout_npz_path)

    true_next = np.asarray(result["true_next_states"], dtype=np.float32)
    pred_next = np.asarray(result["pred_next_states"], dtype=np.float32)
    true_reward = np.asarray(result["true_rewards"], dtype=np.float32).reshape(-1)
    pred_reward = np.asarray(result["pred_rewards"], dtype=np.float32).reshape(-1)

    state_dim = true_next.shape[1]
    total_plots = state_dim + 1

    ncols = int(np.ceil(np.sqrt(total_plots)))
    nrows = int(np.ceil(total_plots / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.8 * ncols, 4.8 * nrows),
        squeeze=False,
    )
    axes = axes.flatten()

    scatter_kwargs = dict(s=10, alpha=0.7, color="tab:blue")
    line_kwargs = dict(linewidth=1.8, color="tab:orange")

    for d in range(state_dim):
        ax = axes[d]
        x = true_next[:, d]
        y = pred_next[:, d]

        a, b = _fit_line(x, y)
        ax.scatter(x, y, **scatter_kwargs)

        if x.size > 0:
            x_line = np.array([x.min(), x.max()], dtype=np.float32)
            if np.allclose(x_line[0], x_line[1]):
                x_line = np.array([x_line[0] - 1.0, x_line[1] + 1.0], dtype=np.float32)
            y_line = a * x_line + b
            ax.plot(x_line, y_line, **line_kwargs, label=f"Fit: y={a:.3f}x+{b:.3f}")

        ax.set_xlabel(f"True State Dim {d}")
        ax.set_ylabel(f"Pred State Dim {d}")
        ax.set_title(f"State Dimension {d}")
        ax.legend()

    ax = axes[state_dim]
    a, b = _fit_line(true_reward, pred_reward)
    ax.scatter(true_reward, pred_reward, **scatter_kwargs)

    if true_reward.size > 0:
        x_line = np.array([true_reward.min(), true_reward.max()], dtype=np.float32)
        if np.allclose(x_line[0], x_line[1]):
            x_line = np.array([x_line[0] - 1.0, x_line[1] + 1.0], dtype=np.float32)
        y_line = a * x_line + b
        ax.plot(x_line, y_line, **line_kwargs, label=f"Fit: y={a:.3f}x+{b:.3f}")

    ax.set_xlabel("True Reward")
    ax.set_ylabel("Pred Reward")
    ax.set_title("Reward")
    ax.legend()

    for i in range(total_plots, len(axes)):
        axes[i].axis("off")

    if title_prefix is None:
        title_prefix = "Rollout"

    fig.suptitle(f"{title_prefix} Scatter Comparison", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=200)

    if show:
        plt.show()
    else:
        plt.close()

    return save_path


def plot_rollout_curve_state(
    rollout_npz_path: str,
    save_dir: str = "sequence_plots",
    prefix: str = "sequence_rollout_curve_state",
    title_prefix: Optional[str] = None,
    show: bool = False,
    max_points: Optional[int] = None,
    smooth_window: Optional[int] = None,
) -> list[str]:
    ensure_dirs(save_dir)
    result = load_npz_dict(rollout_npz_path)

    true_next = np.asarray(result["true_next_states"], dtype=np.float32)
    pred_next = np.asarray(result["pred_next_states"], dtype=np.float32)

    if max_points is not None:
        true_next = true_next[:max_points]
        pred_next = pred_next[:max_points]

    t = np.arange(true_next.shape[0])
    state_dim = true_next.shape[1]
    saved = []

    for d in range(state_dim):
        true_y = true_next[:, d]
        pred_y = pred_next[:, d]

        if smooth_window is not None and smooth_window > 1:
            true_y = _moving_average(true_y, smooth_window)
            pred_y = _moving_average(pred_y, smooth_window)

        plt.figure(figsize=(10, 4))
        plt.plot(t, true_y, label=f"True State Dim {d}")
        plt.plot(t, pred_y, label=f"Pred State Dim {d}")
        plt.xlabel("Time Step")
        plt.ylabel("Value")

        base_title = f"State Dimension {d} Over Time"
        if title_prefix is not None:
            base_title = f"{title_prefix} - {base_title}"
        if smooth_window is not None and smooth_window > 1:
            base_title += f" (MA={smooth_window})"

        plt.title(base_title)
        plt.legend()
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"{prefix}_dim{d}.png")
        plt.savefig(save_path, dpi=200)
        saved.append(save_path)

        if show:
            plt.show()
        else:
            plt.close()

    return saved


def plot_rollout_curve_reward(
    rollout_npz_path: str,
    save_dir: str = "sequence_plots",
    save_name: str = "sequence_rollout_curve_reward.png",
    title_prefix: Optional[str] = None,
    show: bool = False,
    max_points: Optional[int] = None,
    smooth_window: Optional[int] = None,
) -> str:
    ensure_dirs(save_dir)
    result = load_npz_dict(rollout_npz_path)

    true_reward = np.asarray(result["true_rewards"], dtype=np.float32).reshape(-1)
    pred_reward = np.asarray(result["pred_rewards"], dtype=np.float32).reshape(-1)

    if max_points is not None:
        true_reward = true_reward[:max_points]
        pred_reward = pred_reward[:max_points]

    if smooth_window is not None and smooth_window > 1:
        true_reward = _moving_average(true_reward, smooth_window)
        pred_reward = _moving_average(pred_reward, smooth_window)

    t = np.arange(true_reward.shape[0])

    plt.figure(figsize=(10, 4))
    plt.plot(t, true_reward, label="True Reward")
    plt.plot(t, pred_reward, label="Pred Reward")
    plt.xlabel("Time Step")
    plt.ylabel("Value")

    title = "Reward Over Time"
    if title_prefix is not None:
        title = f"{title_prefix} - {title}"
    if smooth_window is not None and smooth_window > 1:
        title += f" (MA={smooth_window})"

    plt.title(title)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=200)

    if show:
        plt.show()
    else:
        plt.close()

    return save_path


def plot_delta_curve_state(
    rollout_npz_path: str,
    save_dir: str = "sequence_plots",
    prefix: str = "sequence_delta_curve_state",
    title_prefix: Optional[str] = None,
    show: bool = False,
    max_points: Optional[int] = None,
) -> list[str]:
    ensure_dirs(save_dir)
    result = load_npz_dict(rollout_npz_path)

    true_next = np.asarray(result["true_next_states"], dtype=np.float32)
    pred_next = np.asarray(result["pred_next_states"], dtype=np.float32)

    if true_next.shape[0] < 2:
        return []

    true_delta = np.diff(true_next, axis=0)
    pred_delta = np.diff(pred_next, axis=0)

    if max_points is not None:
        true_delta = true_delta[:max_points]
        pred_delta = pred_delta[:max_points]

    t = np.arange(true_delta.shape[0])
    state_dim = true_delta.shape[1]
    saved = []

    for d in range(state_dim):
        plt.figure(figsize=(10, 4))
        plt.plot(t, true_delta[:, d], label=f"True Δ State Dim {d}")
        plt.plot(t, pred_delta[:, d], label=f"Pred Δ State Dim {d}")
        plt.xlabel("Time Step")
        plt.ylabel("Δ Value")

        title = f"Delta State Dimension {d} Over Time"
        if title_prefix is not None:
            title = f"{title_prefix} - {title}"

        plt.title(title)
        plt.legend()
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"{prefix}_dim{d}.png")
        plt.savefig(save_path, dpi=200)
        saved.append(save_path)

        if show:
            plt.show()
        else:
            plt.close()

    return saved


def plot_sliding_rse_state(
    rollout_npz_path: str,
    save_dir: str = "sequence_plots",
    save_name: str = "sequence_sliding_rse_state.png",
    title_prefix: Optional[str] = None,
    show: bool = False,
) -> str:
    ensure_dirs(save_dir)
    result = load_npz_dict(rollout_npz_path)

    sliding = np.asarray(result.get("state_sliding_rse_per_dim", np.empty((0, 0))), dtype=np.float32)
    if sliding.size == 0:
        return ""

    t = np.arange(sliding.shape[0])

    plt.figure(figsize=(10, 5))
    for d in range(sliding.shape[1]):
        plt.plot(t, sliding[:, d], label=f"State Dim {d}")

    plt.xlabel("Window Start Index")
    plt.ylabel("RSE")
    title = "Sliding Window State RSE"
    if title_prefix is not None:
        title = f"{title_prefix} - {title}"
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=200)

    if show:
        plt.show()
    else:
        plt.close()

    return save_path


def plot_sliding_corr_state(
    rollout_npz_path: str,
    save_dir: str = "sequence_plots",
    save_name: str = "sequence_sliding_corr_state.png",
    title_prefix: Optional[str] = None,
    show: bool = False,
) -> str:
    ensure_dirs(save_dir)
    result = load_npz_dict(rollout_npz_path)

    sliding = np.asarray(result.get("state_sliding_corr_per_dim", np.empty((0, 0))), dtype=np.float32)
    if sliding.size == 0:
        return ""

    t = np.arange(sliding.shape[0])

    plt.figure(figsize=(10, 5))
    for d in range(sliding.shape[1]):
        plt.plot(t, sliding[:, d], label=f"State Dim {d}")

    plt.xlabel("Window Start Index")
    plt.ylabel("Correlation")
    title = "Sliding Window State Correlation"
    if title_prefix is not None:
        title = f"{title_prefix} - {title}"
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=200)

    if show:
        plt.show()
    else:
        plt.close()

    return save_path


def plot_sliding_rse_reward(
    rollout_npz_path: str,
    save_dir: str = "sequence_plots",
    save_name: str = "sequence_sliding_rse_reward.png",
    title_prefix: Optional[str] = None,
    show: bool = False,
) -> str:
    ensure_dirs(save_dir)
    result = load_npz_dict(rollout_npz_path)

    sliding = np.asarray(result.get("reward_sliding_rse", np.array([])), dtype=np.float32).reshape(-1)
    if sliding.size == 0:
        return ""

    t = np.arange(sliding.size)

    plt.figure(figsize=(10, 4))
    plt.plot(t, sliding, label="Reward")
    plt.xlabel("Window Start Index")
    plt.ylabel("RSE")
    title = "Sliding Window Reward RSE"
    if title_prefix is not None:
        title = f"{title_prefix} - {title}"
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=200)

    if show:
        plt.show()
    else:
        plt.close()

    return save_path


def plot_sliding_corr_reward(
    rollout_npz_path: str,
    save_dir: str = "sequence_plots",
    save_name: str = "sequence_sliding_corr_reward.png",
    title_prefix: Optional[str] = None,
    show: bool = False,
) -> str:
    ensure_dirs(save_dir)
    result = load_npz_dict(rollout_npz_path)

    sliding = np.asarray(result.get("reward_sliding_corr", np.array([])), dtype=np.float32).reshape(-1)
    if sliding.size == 0:
        return ""

    t = np.arange(sliding.size)

    plt.figure(figsize=(10, 4))
    plt.plot(t, sliding, label="Reward")
    plt.xlabel("Window Start Index")
    plt.ylabel("Correlation")
    title = "Sliding Window Reward Correlation"
    if title_prefix is not None:
        title = f"{title_prefix} - {title}"
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=200)

    if show:
        plt.show()
    else:
        plt.close()

    return save_path


def plot_metrics_summary(
    rollout_npz_path: str,
    save_dir: str = "sequence_plots",
    save_name: str = "sequence_metrics_summary.png",
    title_prefix: Optional[str] = None,
    show: bool = False,
) -> str:
    ensure_dirs(save_dir)
    result = load_npz_dict(rollout_npz_path)

    state_rse = np.asarray(result.get("state_rse_per_dim", np.array([])), dtype=np.float32).reshape(-1)
    state_corr = np.asarray(result.get("state_corr_per_dim", np.array([])), dtype=np.float32).reshape(-1)
    state_delta_rse = np.asarray(result.get("state_delta_rse_per_dim", np.array([])), dtype=np.float32).reshape(-1)
    state_delta_corr = np.asarray(result.get("state_delta_corr_per_dim", np.array([])), dtype=np.float32).reshape(-1)
    state_amp = np.asarray(result.get("state_amp_ratio_per_dim", np.array([])), dtype=np.float32).reshape(-1)
    state_lag = np.asarray(result.get("state_best_lag_per_dim", np.array([])), dtype=np.float32).reshape(-1)

    reward_vals = {
        "RSE": float(result.get("reward_rse", np.nan)),
        "Corr": float(result.get("reward_corr", np.nan)),
        "Δ RSE": float(result.get("reward_delta_rse", np.nan)),
        "Δ Corr": float(result.get("reward_delta_corr", np.nan)),
        "Amp Ratio": float(result.get("reward_amp_ratio", np.nan)),
        "Best Lag": float(result.get("reward_best_lag", np.nan)),
    }

    fig, axes = plt.subplots(2, 1, figsize=(11, 8))

    if state_rse.size > 0:
        x = np.arange(state_rse.size)
        width = 0.13

        axes[0].bar(x - 2.5 * width, state_rse, width=width, label="RSE")
        axes[0].bar(x - 1.5 * width, state_corr, width=width, label="Corr")
        axes[0].bar(x - 0.5 * width, state_delta_rse, width=width, label="Δ RSE")
        axes[0].bar(x + 0.5 * width, state_delta_corr, width=width, label="Δ Corr")
        axes[0].bar(x + 1.5 * width, state_amp, width=width, label="Amp Ratio")
        axes[0].bar(x + 2.5 * width, state_lag, width=width, label="Best Lag")

        axes[0].set_xticks(x)
        axes[0].set_xticklabels([f"Dim {i}" for i in range(state_rse.size)])
        axes[0].set_title("State Metrics by Dimension")
        axes[0].legend()

    reward_names = list(reward_vals.keys())
    reward_data = [reward_vals[k] for k in reward_names]
    axes[1].bar(np.arange(len(reward_names)), reward_data)
    axes[1].set_xticks(np.arange(len(reward_names)))
    axes[1].set_xticklabels(reward_names)
    axes[1].set_title("Reward Metrics")

    title = "Rollout Metrics Summary"
    if title_prefix is not None:
        title = f"{title_prefix} - {title}"
    fig.suptitle(title, fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=200)

    if show:
        plt.show()
    else:
        plt.close()

    return save_path


def plot_all_rollout(
    rollout_npz_path: str,
    save_dir: str = "sequence_plots",
    tag: str = "sequence_rollout",
    show: bool = False,
    max_points: Optional[int] = None,
    smooth_window: Optional[int] = None,
) -> dict:
    ensure_dirs(save_dir)

    title_prefix = _format_rollout_tag(tag)

    out = {
        "scatter_combined": plot_rollout_scatter_combined(
            rollout_npz_path=rollout_npz_path,
            save_dir=save_dir,
            save_name=f"{tag}_scatter_combined.png",
            title_prefix=title_prefix,
            show=show,
        ),
        "curve_state": plot_rollout_curve_state(
            rollout_npz_path=rollout_npz_path,
            save_dir=save_dir,
            prefix=f"{tag}_curve_state",
            title_prefix=title_prefix,
            show=show,
            max_points=max_points,
            smooth_window=smooth_window,
        ),
        "curve_reward": plot_rollout_curve_reward(
            rollout_npz_path=rollout_npz_path,
            save_dir=save_dir,
            save_name=f"{tag}_curve_reward.png",
            title_prefix=title_prefix,
            show=show,
            max_points=max_points,
            smooth_window=smooth_window,
        ),
        "delta_curve_state": plot_delta_curve_state(
            rollout_npz_path=rollout_npz_path,
            save_dir=save_dir,
            prefix=f"{tag}_delta_curve_state",
            title_prefix=title_prefix,
            show=show,
            max_points=max_points,
        ),
        "sliding_rse_state": plot_sliding_rse_state(
            rollout_npz_path=rollout_npz_path,
            save_dir=save_dir,
            save_name=f"{tag}_sliding_rse_state.png",
            title_prefix=title_prefix,
            show=show,
        ),
        "sliding_corr_state": plot_sliding_corr_state(
            rollout_npz_path=rollout_npz_path,
            save_dir=save_dir,
            save_name=f"{tag}_sliding_corr_state.png",
            title_prefix=title_prefix,
            show=show,
        ),
        "sliding_rse_reward": plot_sliding_rse_reward(
            rollout_npz_path=rollout_npz_path,
            save_dir=save_dir,
            save_name=f"{tag}_sliding_rse_reward.png",
            title_prefix=title_prefix,
            show=show,
        ),
        "sliding_corr_reward": plot_sliding_corr_reward(
            rollout_npz_path=rollout_npz_path,
            save_dir=save_dir,
            save_name=f"{tag}_sliding_corr_reward.png",
            title_prefix=title_prefix,
            show=show,
        ),
        "metrics_summary": plot_metrics_summary(
            rollout_npz_path=rollout_npz_path,
            save_dir=save_dir,
            save_name=f"{tag}_metrics_summary.png",
            title_prefix=title_prefix,
            show=show,
        ),
    }
    return out