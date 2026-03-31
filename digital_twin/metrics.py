from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def _to_1d(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _finite_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = _to_1d(x)
    y = _to_1d(y)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def compute_rse(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true, y_pred = _finite_pair(y_true, y_pred)
    if y_true.size == 0:
        return float("nan")

    num = np.sum((y_true - y_pred) ** 2)
    den = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(num / (den + eps))


def compute_corr(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true, y_pred = _finite_pair(y_true, y_pred)
    if y_true.size < 2:
        return float("nan")

    std_true = np.std(y_true)
    std_pred = np.std(y_pred)
    if std_true < eps or std_pred < eps:
        return float("nan")

    return float(np.corrcoef(y_true, y_pred)[0, 1])


def compute_amplitude_ratio(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true, y_pred = _finite_pair(y_true, y_pred)
    if y_true.size == 0:
        return float("nan")

    std_true = np.std(y_true)
    std_pred = np.std(y_pred)
    return float(std_pred / (std_true + eps))


def compute_delta(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        if x.size < 2:
            return np.array([], dtype=np.float64)
        return np.diff(x)
    if x.ndim == 2:
        if x.shape[0] < 2:
            return np.empty((0, x.shape[1]), dtype=np.float64)
        return np.diff(x, axis=0)
    raise ValueError("compute_delta only supports 1D or 2D arrays")


def compute_sliding_metric_1d(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    window: int,
    metric_fn,
) -> np.ndarray:
    y_true, y_pred = _finite_pair(y_true, y_pred)

    if window <= 1 or y_true.size < window:
        return np.array([], dtype=np.float32)

    out = []
    for i in range(y_true.size - window + 1):
        out.append(metric_fn(y_true[i:i + window], y_pred[i:i + window]))
    return np.asarray(out, dtype=np.float32)


def compute_sliding_rse(y_true: np.ndarray, y_pred: np.ndarray, window: int, eps: float = 1e-8) -> np.ndarray:
    return compute_sliding_metric_1d(
        y_true=y_true,
        y_pred=y_pred,
        window=window,
        metric_fn=lambda a, b: compute_rse(a, b, eps=eps),
    )


def compute_sliding_corr(y_true: np.ndarray, y_pred: np.ndarray, window: int, eps: float = 1e-8) -> np.ndarray:
    return compute_sliding_metric_1d(
        y_true=y_true,
        y_pred=y_pred,
        window=window,
        metric_fn=lambda a, b: compute_corr(a, b, eps=eps),
    )


def compute_best_lag_corr(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_lag: int = 50,
    eps: float = 1e-8,
) -> Dict[str, float]:
    y_true, y_pred = _finite_pair(y_true, y_pred)
    if y_true.size < 2:
        return {"best_lag": float("nan"), "best_corr": float("nan")}

    best_lag = 0
    best_corr = -np.inf

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a = y_true[-lag:]
            b = y_pred[: len(y_pred) + lag]
        elif lag > 0:
            a = y_true[:-lag]
            b = y_pred[lag:]
        else:
            a = y_true
            b = y_pred

        if len(a) < 2 or len(b) < 2:
            continue

        corr = compute_corr(a, b, eps=eps)
        if np.isfinite(corr) and corr > best_corr:
            best_corr = corr
            best_lag = lag

    return {"best_lag": float(best_lag), "best_corr": float(best_corr)}


def summarize_vector_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sliding_window: Optional[int] = None,
    lag_max: int = 50,
    eps: float = 1e-8,
) -> Dict[str, np.ndarray | float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    if y_true.ndim == 1:
        rse = compute_rse(y_true, y_pred, eps=eps)
        corr = compute_corr(y_true, y_pred, eps=eps)
        amp_ratio = compute_amplitude_ratio(y_true, y_pred, eps=eps)

        delta_true = compute_delta(y_true)
        delta_pred = compute_delta(y_pred)

        delta_rse = compute_rse(delta_true, delta_pred, eps=eps) if delta_true.size > 0 else float("nan")
        delta_corr = compute_corr(delta_true, delta_pred, eps=eps) if delta_true.size > 1 else float("nan")

        lag_info = compute_best_lag_corr(y_true, y_pred, max_lag=lag_max, eps=eps)

        out: Dict[str, np.ndarray | float] = {
            "rse": float(rse),
            "corr": float(corr),
            "amp_ratio": float(amp_ratio),
            "delta_rse": float(delta_rse),
            "delta_corr": float(delta_corr),
            "best_lag": float(lag_info["best_lag"]),
            "best_lag_corr": float(lag_info["best_corr"]),
        }

        if sliding_window is not None and sliding_window > 1:
            out["sliding_rse"] = compute_sliding_rse(y_true, y_pred, window=sliding_window, eps=eps)
            out["sliding_corr"] = compute_sliding_corr(y_true, y_pred, window=sliding_window, eps=eps)
        else:
            out["sliding_rse"] = np.array([], dtype=np.float32)
            out["sliding_corr"] = np.array([], dtype=np.float32)

        return out

    if y_true.ndim == 2:
        dim = y_true.shape[1]

        rse_per_dim = []
        corr_per_dim = []
        amp_ratio_per_dim = []
        delta_rse_per_dim = []
        delta_corr_per_dim = []
        lag_per_dim = []
        lag_corr_per_dim = []

        sliding_rse_list = []
        sliding_corr_list = []

        for d in range(dim):
            stats = summarize_vector_metrics(
                y_true[:, d],
                y_pred[:, d],
                sliding_window=sliding_window,
                lag_max=lag_max,
                eps=eps,
            )
            rse_per_dim.append(stats["rse"])
            corr_per_dim.append(stats["corr"])
            amp_ratio_per_dim.append(stats["amp_ratio"])
            delta_rse_per_dim.append(stats["delta_rse"])
            delta_corr_per_dim.append(stats["delta_corr"])
            lag_per_dim.append(stats["best_lag"])
            lag_corr_per_dim.append(stats["best_lag_corr"])
            sliding_rse_list.append(stats["sliding_rse"])
            sliding_corr_list.append(stats["sliding_corr"])

        delta_true = compute_delta(y_true)
        delta_pred = compute_delta(y_pred)

        out2: Dict[str, np.ndarray | float] = {
            "rse_mean": float(np.nanmean(rse_per_dim)),
            "rse_flattened": float(compute_rse(y_true.reshape(-1), y_pred.reshape(-1), eps=eps)),
            "rse_per_dim": np.asarray(rse_per_dim, dtype=np.float32),
            "corr_mean": float(np.nanmean(corr_per_dim)),
            "corr_per_dim": np.asarray(corr_per_dim, dtype=np.float32),
            "amp_ratio_mean": float(np.nanmean(amp_ratio_per_dim)),
            "amp_ratio_per_dim": np.asarray(amp_ratio_per_dim, dtype=np.float32),
            "delta_rse_mean": float(np.nanmean(delta_rse_per_dim)),
            "delta_rse_flattened": float(compute_rse(delta_true.reshape(-1), delta_pred.reshape(-1), eps=eps))
            if delta_true.size > 0 else float("nan"),
            "delta_rse_per_dim": np.asarray(delta_rse_per_dim, dtype=np.float32),
            "delta_corr_mean": float(np.nanmean(delta_corr_per_dim)),
            "delta_corr_per_dim": np.asarray(delta_corr_per_dim, dtype=np.float32),
            "best_lag_mean": float(np.nanmean(lag_per_dim)),
            "best_lag_per_dim": np.asarray(lag_per_dim, dtype=np.float32),
            "best_lag_corr_mean": float(np.nanmean(lag_corr_per_dim)),
            "best_lag_corr_per_dim": np.asarray(lag_corr_per_dim, dtype=np.float32),
        }

        if sliding_window is not None and sliding_window > 1:
            out2["sliding_rse_per_dim"] = np.stack(sliding_rse_list, axis=1) if len(sliding_rse_list) > 0 else np.empty((0, dim), dtype=np.float32)
            out2["sliding_corr_per_dim"] = np.stack(sliding_corr_list, axis=1) if len(sliding_corr_list) > 0 else np.empty((0, dim), dtype=np.float32)
        else:
            out2["sliding_rse_per_dim"] = np.empty((0, dim), dtype=np.float32)
            out2["sliding_corr_per_dim"] = np.empty((0, dim), dtype=np.float32)

        return out2

    raise ValueError("Only 1D or 2D arrays are supported")