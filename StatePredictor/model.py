from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


class LSTMStatePredictor(nn.Module):
    """
    LSTM-based next-state predictor.

    Given a window of past states:
        x_seq: (B, K, D)
    predict the next state:
        y_pred: (B, D)

    This is the standard "state predictor" form used in world-model / digital-twin demos.
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 64,
            num_layers: int = 1,
            dropout: float = 0.0,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if num_layers <= 0:
            raise ValueError("num_layers must be > 0")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")

        # PyTorch applies dropout between LSTM layers only when num_layers > 1
        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        # Optional: a small head to map hidden -> state dim
        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.input_dim),
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_seq: (B, K, D)

        Returns:
            y_pred: (B, D)
        """
        if x_seq.ndim != 3:
            raise ValueError(f"x_seq must have shape (B,K,D), got {tuple(x_seq.shape)}")
        if x_seq.size(-1) != self.input_dim:
            raise ValueError(
                f"Last dim of x_seq must be input_dim={self.input_dim}, got {x_seq.size(-1)}"
            )

        out, _ = self.lstm(x_seq)  # out: (B, K, H)
        last = out[:, -1, :]  # (B, H)
        y_pred = self.head(last)  # (B, D)
        return y_pred
