# sequence_model.py

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class SequenceStatePredictorLSTM(nn.Module):
    """
    Sequence state predictor (many-to-many).

    Input:
        state:  (B, T, state_dim)
        action: (B, T, action_dim) or None

    Output:
        pred_next_states: (B, T, state_dim)
        new_hidden: (h_n, c_n)
        lstm_out: (B, T, hidden_dim)

    Notes:
    - use_action is only for the state predictor.
    - This model can also accept single-step input (B, D) / (B, A),
      but the main intended use is sequence training with (B, T, D).
    """

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dim: int = 64,
            num_layers: int = 1,
            dropout: float = 0.0,
            use_action: bool = True,
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_action = use_action

        lstm_dropout = dropout if num_layers > 1 else 0.0
        input_dim = state_dim + action_dim if use_action else state_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )

        self.head = nn.Linear(hidden_dim, state_dim)

    def init_hidden(
            self,
            batch_size: int,
            device: torch.device | str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return h0, c0

    def forward(
            self,
            state: torch.Tensor,
            action: Optional[torch.Tensor] = None,
            hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        squeezed = False

        # Support single-step input for convenience
        if state.dim() == 2:
            state = state.unsqueeze(1)  # (B, 1, D)
            if action is not None and action.dim() == 2:
                action = action.unsqueeze(1)  # (B, 1, A)
            squeezed = True

        if state.dim() != 3:
            raise ValueError("state must have shape (B, D) or (B, T, D)")

        if self.use_action:
            if action is None:
                raise ValueError("action must be provided when use_action=True")
            if action.dim() != 3:
                raise ValueError("action must have shape (B, A) or (B, T, A)")
            x = torch.cat([state, action], dim=-1)
        else:
            x = state

        lstm_out, new_hidden = self.lstm(x, hidden)  # (B, T, H)
        pred_next_states = self.head(lstm_out)  # (B, T, state_dim)

        if squeezed:
            pred_next_states = pred_next_states.squeeze(1)  # (B, D)
            lstm_out = lstm_out.squeeze(1)  # (B, H)

        return pred_next_states, new_hidden, lstm_out


class SequenceRewardPredictorDNN(nn.Module):
    """
    Sequence reward predictor (many-to-many).

    Input:
        state:      (B, T, state_dim)
        action:     (B, T, action_dim)
        hidden_feat:(B, T, hidden_dim) or (B, T, num_layers * hidden_dim)

    Output:
        pred_rewards: (B, T, 1)

    Notes:
    - No use_action switch here by design.
    - Reward predictor always uses action.
    - hidden_feat is expected to come from state_model's lstm_out
      or another per-time-step hidden representation.
    """

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dim: int = 64,
            num_layers: int = 1,
            mlp_hidden_dims: tuple[int, int] = (128, 64),
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # For sequence training, the cleanest choice is to use per-time-step
        # hidden features of size hidden_dim (i.e. lstm_out).
        # If later you want to pass a concatenated multi-layer feature,
        # this class can still accept it as long as the last dim matches input_dim.
        h1, h2 = mlp_hidden_dims

        # Default expected hidden feature size = hidden_dim
        self.default_hidden_feat_dim = hidden_dim
        input_dim = state_dim + action_dim + self.default_hidden_feat_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )

    def forward(
            self,
            state: torch.Tensor,
            action: torch.Tensor,
            hidden_feat: torch.Tensor,
    ) -> torch.Tensor:
        # Support single-step input for convenience
        squeezed = False
        if state.dim() == 2:
            state = state.unsqueeze(1)  # (B, 1, D)
            action = action.unsqueeze(1)  # (B, 1, A)
            hidden_feat = hidden_feat.unsqueeze(1)  # (B, 1, Hf)
            squeezed = True

        if state.dim() != 3:
            raise ValueError("state must have shape (B, D) or (B, T, D)")
        if action.dim() != 3:
            raise ValueError("action must have shape (B, A) or (B, T, A)")
        if hidden_feat.dim() != 3:
            raise ValueError("hidden_feat must have shape (B, H) or (B, T, H)")

        x = torch.cat([state, action, hidden_feat], dim=-1)

        # nn.Sequential over the last dim works for (..., input_dim)
        pred_rewards = self.net(x)  # (B, T, 1)

        if squeezed:
            pred_rewards = pred_rewards.squeeze(1)  # (B, 1)

        return pred_rewards

