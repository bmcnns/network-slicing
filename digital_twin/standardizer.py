# standardizer.py

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass

import numpy as np


@dataclass
class Standardizer:
    """
    Z-score standardizer:
        x_norm = (x - mean) / (std + eps)
    """
    mean: np.ndarray
    std: np.ndarray
    eps: float = 1e-8

    @classmethod
    def fit(cls, x: np.ndarray, eps: float = 1e-8) -> "Standardizer":
        x = np.asarray(x, dtype=np.float32)
        mean = x.mean(axis=0).astype(np.float32)
        std = x.std(axis=0).astype(np.float32)
        return cls(mean=mean, std=std, eps=eps)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return (x - self.mean) / (self.std + self.eps)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return x * (self.std + self.eps) + self.mean

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "mean": self.mean,
                    "std": self.std,
                    "eps": self.eps,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "Standardizer":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return cls(
            mean=np.asarray(obj["mean"], dtype=np.float32),
            std=np.asarray(obj["std"], dtype=np.float32),
            eps=float(obj["eps"]),
        )