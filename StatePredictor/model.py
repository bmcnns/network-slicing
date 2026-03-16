from __future__ import annotations

import torch
import torch.nn as nn


class LSTMStatePredictor(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 8,
        num_layers: int = 2,
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

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)

        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.input_dim),
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_seq: (B, K, input_dim)

        Returns:
            next_state_pred: (B, input_dim)
        """

        if x_seq.ndim != 3:
            raise ValueError(
                f"x_seq must have shape (B,K,input_dim), got {tuple(x_seq.shape)}"
            )

        if x_seq.size(-1) != self.input_dim:
            raise ValueError(
                f"Last dim must be input_dim={self.input_dim}"
            )

        out, _ = self.lstm(x_seq)

        last = out[:, -1, :]

        y_pred = self.head(last)

        return y_pred