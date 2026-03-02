# dataset_builder.py
from __future__ import annotations
from __future__ import annotations

import logging
from dataclasses import dataclass
import random
from typing import Callable, Dict, Optional, Tuple, Any, List

import numpy as np
import torch
from torch.utils.data import Dataset
import json
from pathlib import Path

from Simulator.environment import NetworkSlicingEnv
from Simulator.config import Config
from typing import Any, Dict, Optional, Tuple
from scipy.integrate import solve_ivp

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger("data-builder")


# ============================================================
# 1. Common containers
# ============================================================

@dataclass
class TimeSeriesData:
    """
    Raw continuous time-series container.
    states: (T, D)
    """
    states: np.ndarray
    meta: Dict[str, Any]


@dataclass
class DatasetSplits:
    raw: TimeSeriesData
    train: Dataset
    val: Dataset
    test: Dataset

    def save(self, dir_path: str) -> None:
        """
        Save raw continuous time-series + metadata needed to reproduce splits & windows.

        Files written:
          - states.npy
          - meta.json
        """
        p = Path(dir_path)
        p.mkdir(parents=True, exist_ok=True)

        # 1) raw states
        np.save(p / "states.npy", self.raw.states.astype(np.float32, copy=False))

        # 2) meta (json-serializable)
        meta = self.raw.meta.copy()

        # ensure JSON serializable
        def _to_jsonable(x):
            if isinstance(x, (np.ndarray,)):
                return x.tolist()
            if isinstance(x, (np.floating, np.integer)):
                return x.item()
            return x

        meta = {k: _to_jsonable(v) for k, v in meta.items()}

        with open(p / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    @staticmethod
    def load(
            dir_path: str,
            window_size: int,
            stride: int = 1,
            split: Optional[Tuple[float, float, float]] = None,
            standardize: Optional[bool] = None,
    ) -> "DatasetSplits":
        """
        Load raw states + meta, then rebuild train/val/test window datasets.

        Notes:
        - If split/standardize is not provided, tries to read from meta.json.
        - This recreates window datasets deterministically (contiguous split).
        """
        p = Path(dir_path)
        states = np.load(p / "states.npy").astype(np.float32, copy=False)

        with open(p / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        # pick defaults from meta if not supplied
        if split is None:
            split = tuple(meta.get("split", (0.7, 0.15, 0.15)))  # type: ignore
        if standardize is None:
            standardize = bool(meta.get("standardize", True))

        raw = TimeSeriesData(states=states, meta=meta)

        train_s, val_s, test_s = split_contiguous(states, *split)

        train_ds, val_ds, test_ds, scaler = build_datasets(
            train_s,
            val_s,
            test_s,
            window_size=window_size,
            stride=stride,
            standardize=standardize,
        )

        # if we recompute scaler, store it back into meta (optional)
        if scaler is not None:
            raw.meta["train_mean"] = scaler.mean_.tolist()
            raw.meta["train_std"] = scaler.std_.tolist()

        return DatasetSplits(raw=raw, train=train_ds, val=val_ds, test=test_ds)


# ============================================================
# 2. Normalization (fit on train only)
# ============================================================

class Standardizer:
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray):
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Standardizer not fitted")
        return (x - self.mean_) / (self.std_ + self.eps)

    def inverse_transform(self, x):
        if self.mean_ is None:
            raise RuntimeError("Standardizer not fitted")
        return x * (self.std_ + self.eps) + self.mean_

    def to_dict(self):
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Standardizer not fitted")
        return {"eps": self.eps, "mean": self.mean_.tolist(), "std": self.std_.tolist()}

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def load(path: str) -> "Standardizer":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        s = Standardizer(eps=float(d["eps"]))
        s.mean_ = np.array(d["mean"], dtype=np.float32)
        s.std_ = np.array(d["std"], dtype=np.float32)
        return s


# ============================================================
# 3. Sliding-window dataset (core)
# ============================================================

class NextStateWindowDataset(Dataset):
    """
    Continuous sequence → sliding windows

    input : x_seq  (K, D)
    target: y_next (D,)
    """

    def __init__(self, states: np.ndarray, window_size: int, stride: int = 1):
        if states.ndim != 2:
            raise ValueError("states must be (T, D)")
        if window_size <= 0:
            raise ValueError("window_size must be > 0")

        self.states = states.astype(np.float32, copy=False)
        self.window_size = window_size
        self.stride = stride

        T = states.shape[0]
        self.indices = np.arange(
            0, T - window_size - 1, stride, dtype=np.int64
        )

        if len(self.indices) == 0:
            raise ValueError("Time series too short for given window_size")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        i = int(self.indices[idx])
        x_seq = self.states[i: i + self.window_size]
        y_next = self.states[i + self.window_size]
        return (
            torch.from_numpy(x_seq),
            torch.from_numpy(y_next),
        )


# ============================================================
# 4. Time-series split (no leakage)
# ============================================================

def split_contiguous(
        states: np.ndarray,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
):
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1")

    T = states.shape[0]
    t_train = int(T * train_ratio)
    t_val = int(T * val_ratio)

    train = states[:t_train]
    val = states[t_train: t_train + t_val]
    test = states[t_train + t_val:]

    return train, val, test


def build_datasets(
        train_states: np.ndarray,
        val_states: np.ndarray,
        test_states: np.ndarray,
        window_size: int,
        stride: int,
        standardize: bool,
        scaler_save_path: str | None = "model_components/standardizer.json"
):
    scaler = None
    if standardize:
        scaler = Standardizer().fit(train_states)

        if scaler_save_path is not None:
            scaler.save(scaler_save_path)
            logger.info(f"[INFO] Standardizer saved to {scaler_save_path}")

        train_states = scaler.transform(train_states)
        val_states = scaler.transform(val_states)
        test_states = scaler.transform(test_states)

    return (
        NextStateWindowDataset(train_states, window_size, stride),
        NextStateWindowDataset(val_states, window_size, stride),
        NextStateWindowDataset(test_states, window_size, stride),
        scaler,
    )


# ============================================================
# 5. Trigonometric data builder
# ============================================================

class TrigonometricDataBuilder:
    """
    Generate synthetic continuous time-series using sin/cos.
    """

    def __init__(
            self,
            num_steps: int,
            mode: str = "both",  # sin | cos | both
            dt: float = 0.01,
            window_size: int = 32,
            stride: int = 1,
            noise_std: float = 0.0,
            split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
            seed: int = 0,
            standardize: bool = True,
    ):
        self.num_steps = num_steps
        self.mode = mode
        self.dt = dt
        self.window_size = window_size
        self.stride = stride
        self.noise_std = noise_std
        self.split = split
        self.seed = seed
        self.standardize = standardize

        if mode not in {"sin", "cos", "both"}:
            raise ValueError("mode must be sin / cos / both")

    def build(self) -> DatasetSplits:
        rng = np.random.default_rng(self.seed)
        t = np.arange(self.num_steps) * self.dt

        base = []
        for f in (0.7, 1.3, 2.1):
            if self.mode in {"sin", "both"}:
                base.append(np.sin(2 * np.pi * f * t))
            if self.mode in {"cos", "both"}:
                base.append(np.cos(2 * np.pi * f * t))

        states = np.stack(base, axis=1).astype(np.float32)

        if self.noise_std > 0:
            states += rng.normal(0, self.noise_std, states.shape)

        raw = TimeSeriesData(
            states=states,
            meta={
                "source": "trigonometric",
                "mode": self.mode,
                "dt": self.dt,
                "split": list(self.split),
                "standardize": self.standardize,
                "window_size": self.window_size,
                "stride": self.stride,
                "noise_std": self.noise_std,
                "seed": self.seed,
            },
        )

        train_s, val_s, test_s = split_contiguous(
            states, *self.split
        )

        train_ds, val_ds, test_ds, scaler = build_datasets(
            train_s,
            val_s,
            test_s,
            self.window_size,
            self.stride,
            self.standardize,
        )

        if scaler is not None:
            raw.meta["train_mean"] = scaler.mean_.tolist()
            raw.meta["train_std"] = scaler.std_.tolist()

        return DatasetSplits(raw, train_ds, val_ds, test_ds)


# ============================================================
# 6. Simulator data builder
# ============================================================

class SimulatorDataBuilder:
    """
    Build dataset by running the NetworkSlicing simulator with a built-in policy.

    Current policy:
      - "random": Dirichlet random bandwidth allocation (sum = 20 MHz)

    Output:
      - raw.states: (T, D) continuous physical state sequence
      - train/val/test: windowed datasets (optionally standardized)
    """

    def __init__(
            self,
            num_steps: int,
            window_size: int = 32,
            stride: int = 1,
            split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
            seed: int = 0,
            standardize: bool = True,
            policy: str = "random",
            meta: Optional[Dict[str, Any]] = None,
    ):
        self.num_steps = int(num_steps)
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.split = split
        self.seed = int(seed)
        self.standardize = bool(standardize)
        self.policy = policy
        self.meta = meta or {}

        if policy not in {"random"}:
            raise ValueError(f"Unsupported policy: {policy}")

    # -----------------------------
    # Internal: run simulator
    # -----------------------------
    def _run_simulator(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        env = NetworkSlicingEnv(seed=self.seed)

        states = []

        for t in range(self.num_steps):
            if self.policy == "random":
                # Dirichlet -> sum = 20 MHz
                # todo: try different actnions
                action = rng.dirichlet([1.0, 1.0, 1.0]) * 20.0
            else:
                raise RuntimeError("Unreachable")

            state, reward, done, info = env.step(action)
            state = [random.randint(1, 100) for _ in range(3)]
            # Ensure (D,)
            state_arr = np.asarray(state, dtype=np.float32).reshape(-1)
            states.append(state_arr)

            # Optional: handle episodic envs
            if done and hasattr(env, "reset"):
                reset_out = env.reset()
                # reset state is usually ignored to keep continuity

        return np.stack(states, axis=0)  # (T, D)

    # -----------------------------
    # Public: build datasets
    # -----------------------------
    def build(self) -> DatasetSplits:
        states = self._run_simulator()

        if states.ndim != 2:
            raise ValueError("Simulator must produce (T, D) state array")

        raw = TimeSeriesData(
            states=states,
            meta={
                "source": "network_slicing_simulator",
                "policy": self.policy,
                "num_steps": self.num_steps,
                "split": list(self.split),
                "standardize": self.standardize,
                "window_size": self.window_size,
                "stride": self.stride,
                "seed": self.seed,
                **self.meta,
            },
        )

        # contiguous split in physical space
        train_s, val_s, test_s = split_contiguous(
            raw.states, *self.split
        )

        # windowed datasets (+ optional standardization)
        train_ds, val_ds, test_ds, scaler = build_datasets(
            train_s,
            val_s,
            test_s,
            self.window_size,
            self.stride,
            self.standardize,
        )

        # record standardization info for reproducibility
        if scaler is not None:
            raw.meta["train_mean"] = scaler.mean_.tolist()
            raw.meta["train_std"] = scaler.std_.tolist()

        return DatasetSplits(raw, train_ds, val_ds, test_ds)


class ChaosDataBuilder:
    """
    Generate synthetic continuous time-series using the Lorenz system (chaotic attractor).

    Produces states: np.ndarray of shape (T, 3)
    Then splits contiguous into train/val/test and builds window datasets.
    """

    def __init__(
            self,
            num_steps: int,
            dt: float = 0.01,
            window_size: int = 32,
            stride: int = 1,
            noise_std: float = 0.0,
            split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
            seed: int = 0,
            standardize: bool = True,
            # Lorenz params
            sigma: float = 10.0,
            rho: float = 28.0,
            beta: float = 8.0 / 3.0,
            # IVP solver config
            solver_method: str = "RK45",
            rtol: float = 1e-8,
            atol: float = 1e-10,
            # optional initial condition (3,)
            initial_state: Optional[np.ndarray] = None,
            meta: Optional[Dict[str, Any]] = None,
    ):
        if num_steps <= 1:
            raise ValueError("num_steps must be > 1")
        if dt <= 0:
            raise ValueError("dt must be > 0")
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        if stride <= 0:
            raise ValueError("stride must be > 0")
        if noise_std < 0:
            raise ValueError("noise_std must be >= 0")
        if len(split) != 3 or not np.isclose(sum(split), 1.0):
            raise ValueError("split must be 3-tuple that sums to 1.0")

        self.num_steps = int(num_steps)
        self.dt = float(dt)
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.noise_std = float(noise_std)
        self.split = tuple(float(x) for x in split)
        self.seed = int(seed)
        self.standardize = bool(standardize)

        self.sigma = float(sigma)
        self.rho = float(rho)
        self.beta = float(beta)

        self.solver_method = str(solver_method)
        self.rtol = float(rtol)
        self.atol = float(atol)

        self.initial_state = None if initial_state is None else np.asarray(initial_state, dtype=np.float64).reshape(3)
        self.meta = meta or {}

    def _generate_lorenz_states(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)

        if self.initial_state is None:
            # 生成一个可复现的初值（别太大，避免刚开始数值爆得很厉害）
            init = rng.normal(loc=0.0, scale=1.0, size=(3,)).astype(np.float64)
        else:
            init = self.initial_state.astype(np.float64)

        t0 = 0.0
        t1 = (self.num_steps - 1) * self.dt
        t_eval = np.linspace(t0, t1, self.num_steps, dtype=np.float64)

        sigma, rho, beta = self.sigma, self.rho, self.beta

        def lorenz_ode(t: float, s: np.ndarray) -> np.ndarray:
            x, y, z = s
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            return np.array([dx, dy, dz], dtype=np.float64)

        sol = solve_ivp(
            fun=lorenz_ode,
            t_span=(t0, t1),
            y0=init,
            t_eval=t_eval,
            method=self.solver_method,
            rtol=self.rtol,
            atol=self.atol,
        )
        if not sol.success:
            raise RuntimeError(f"solve_ivp failed: {sol.message}")

        states = sol.y.T.astype(np.float32)  # (T,3)

        if self.noise_std > 0:
            states = states + rng.normal(0.0, self.noise_std, size=states.shape).astype(np.float32)

        return states

    def build(self):  # -> DatasetSplits
        states = self._generate_lorenz_states()
        if states.ndim != 2 or states.shape[1] != 3:
            raise ValueError(f"Lorenz states must have shape (T,3), got {states.shape}")

        raw = TimeSeriesData(
            states=states,
            meta={
                "source": "lorenz",
                "dt": self.dt,
                "split": list(self.split),
                "standardize": self.standardize,
                "window_size": self.window_size,
                "stride": self.stride,
                "noise_std": self.noise_std,
                "seed": self.seed,
                "sigma": self.sigma,
                "rho": self.rho,
                "beta": self.beta,
                "solver_method": self.solver_method,
                "rtol": self.rtol,
                "atol": self.atol,
                **self.meta,
            },
        )

        train_s, val_s, test_s = split_contiguous(states, *self.split)

        train_ds, val_ds, test_ds, scaler = build_datasets(
            train_s,
            val_s,
            test_s,
            window_size=self.window_size,
            stride=self.stride,
            standardize=self.standardize,
        )

        if scaler is not None:
            raw.meta["train_mean"] = scaler.mean_.tolist()
            raw.meta["train_std"] = scaler.std_.tolist()

        return DatasetSplits(raw, train_ds, val_ds, test_ds)


class CSVDataBuilder:
    """
    Build continuous time-series from CSV file.

    Assumptions:
      - First row is header
      - Remaining rows are numeric state values
      - Each row is one time step
      - Dimension D inferred from column count (not fixed)
    """

    def __init__(
            self,
            csv_path: str,
            window_size: int = 32,
            stride: int = 1,
            split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
            seed: int = 0,
            standardize: bool = True,
            delimiter: str = ",",
    ):
        self.csv_path = csv_path
        self.window_size = window_size
        self.stride = stride
        self.split = split
        self.seed = seed
        self.standardize = standardize
        self.delimiter = delimiter

        if not np.isclose(sum(split), 1.0):
            raise ValueError("Split ratios must sum to 1")

    def build(self) -> DatasetSplits:
        # deterministic style (same as other builders)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # ---- read header ----
        with open(self.csv_path, "r", encoding="utf-8") as f:
            header_line = f.readline().strip()

        if not header_line:
            raise ValueError("CSV header is empty")

        columns = [c.strip() for c in header_line.split(self.delimiter)]

        # ---- read numeric data ----
        states = np.loadtxt(
            self.csv_path,
            delimiter=self.delimiter,
            skiprows=1,
            dtype=np.float32,
        )

        if states.ndim == 1:
            states = states.reshape(1, -1)

        if states.ndim != 2:
            raise ValueError("CSV must produce (T,D) array")

        if states.shape[1] != len(columns):
            raise ValueError(
                f"Header columns ({len(columns)}) != data columns ({states.shape[1]})"
            )

        raw = TimeSeriesData(
            states=states.astype(np.float32, copy=False),
            meta={
                "source": "csv",
                "csv_path": self.csv_path,
                "columns": columns,
                "split": list(self.split),
                "standardize": self.standardize,
                "window_size": self.window_size,
                "stride": self.stride,
                "seed": self.seed,
            },
        )

        # ---- contiguous split ----
        train_s, val_s, test_s = split_contiguous(
            states, *self.split
        )

        # ---- build sliding-window datasets ----
        train_ds, val_ds, test_ds, scaler = build_datasets(
            train_s,
            val_s,
            test_s,
            self.window_size,
            self.stride,
            self.standardize,
        )

        if scaler is not None:
            raw.meta["train_mean"] = scaler.mean_.tolist()
            raw.meta["train_std"] = scaler.std_.tolist()

        return DatasetSplits(raw, train_ds, val_ds, test_ds)