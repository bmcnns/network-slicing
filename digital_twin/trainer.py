# sequence_trainer.py

from __future__ import annotations

from typing import Tuple, Optional, List

import torch
import torch.nn as nn


def _compute_state_per_dim_mse(
    pred_next_states: torch.Tensor,
    true_next_states: torch.Tensor,
) -> torch.Tensor:
    """
    pred_next_states: (B, T, D)
    true_next_states: (B, T, D)
    return: (D, )
    """
    return ((pred_next_states - true_next_states) ** 2).mean(dim=(0, 1))


def _run_state_epoch(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip_norm: Optional[float] = None,
) -> Tuple[float, List[float]]:
    """
    Sequence state training (many-to-many)

    Loss:
        MSE over (B, T, D)
    """
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_batches = 0
    per_dim_sum = None

    for states, actions, _, next_states in loader:
        states = states.to(device).float()           # (B, T, D)
        actions = actions.to(device).float()         # (B, T, A)
        next_states = next_states.to(device).float() # (B, T, D)

        batch_size = states.shape[0]
        hidden = model.init_hidden(batch_size=batch_size, device=device)

        if training:
            optimizer.zero_grad()

        pred_next_states, _, _ = model(states, actions, hidden=hidden)

        loss = criterion(pred_next_states, next_states)

        if training:
            loss.backward()

            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            optimizer.step()

        batch_loss = loss.detach().item()

        batch_per_dim = _compute_state_per_dim_mse(
            pred_next_states.detach(),
            next_states.detach(),
        )

        total_loss += batch_loss
        total_batches += 1

        if per_dim_sum is None:
            per_dim_sum = batch_per_dim
        else:
            per_dim_sum += batch_per_dim

    if total_batches == 0:
        raise ValueError("Empty loader")

    avg_loss = total_loss / total_batches
    avg_per_dim = (per_dim_sum / total_batches).cpu().tolist()

    return avg_loss, avg_per_dim


def _run_reward_epoch(
    state_model: torch.nn.Module,
    reward_model: torch.nn.Module,
    loader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip_norm: Optional[float] = None,
) -> float:
    """
    Sequence reward training (many-to-many)

    state_model is frozen, only used to produce lstm_out.
    reward_model is trained.
    """
    training = optimizer is not None

    state_model.eval()
    reward_model.train(training)

    total_loss = 0.0
    total_batches = 0

    for states, actions, rewards, _ in loader:
        states = states.to(device).float()   # (B, T, D)
        actions = actions.to(device).float() # (B, T, A)
        rewards = rewards.to(device).float() # (B, T, 1)

        batch_size = states.shape[0]
        hidden = state_model.init_hidden(batch_size=batch_size, device=device)

        if training:
            optimizer.zero_grad()

        with torch.no_grad():
            _, _, lstm_out = state_model(states, actions, hidden=hidden)

        pred_rewards = reward_model(states, actions, lstm_out)

        loss = criterion(pred_rewards, rewards)

        if training:
            loss.backward()

            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(reward_model.parameters(), grad_clip_norm)

            optimizer.step()

        total_loss += loss.detach().item()
        total_batches += 1

    if total_batches == 0:
        raise ValueError("Empty loader")

    return total_loss / total_batches


def train_sequence_models(
    state_model: torch.nn.Module,
    reward_model: torch.nn.Module,
    train_loader,
    val_loader,
    device: str = "cpu",
    epochs: int = 100,
    lr: float = 1e-3,
    grad_clip_norm: Optional[float] = None,
):
    """
    Full training loop (state first, then reward)

    Returns:
        history dict
    """
    device_t = torch.device(device)

    state_model.to(device_t)
    reward_model.to(device_t)

    state_optimizer = torch.optim.Adam(state_model.parameters(), lr=lr)
    reward_optimizer = torch.optim.Adam(reward_model.parameters(), lr=lr)

    criterion = nn.MSELoss()

    history = {
        "state_train_loss": [],
        "state_val_loss": [],
        "state_per_dim": [],
        "reward_train_loss": [],
        "reward_val_loss": [],
    }

    # =====================
    # 1. Train State Model
    # =====================
    for epoch in range(epochs):
        train_loss, train_per_dim = _run_state_epoch(
            model=state_model,
            loader=train_loader,
            device=device_t,
            criterion=criterion,
            optimizer=state_optimizer,
            grad_clip_norm=grad_clip_norm,
        )

        val_loss, val_per_dim = _run_state_epoch(
            model=state_model,
            loader=val_loader,
            device=device_t,
            criterion=criterion,
            optimizer=None,
        )

        history["state_train_loss"].append(train_loss)
        history["state_val_loss"].append(val_loss)
        history["state_per_dim"].append(train_per_dim)

        print(f"[State][Epoch {epoch}] train={train_loss:.6f} val={val_loss:.6f}")

    # =====================
    # 2. Train Reward Model
    # =====================
    for epoch in range(epochs):
        train_loss = _run_reward_epoch(
            state_model=state_model,
            reward_model=reward_model,
            loader=train_loader,
            device=device_t,
            criterion=criterion,
            optimizer=reward_optimizer,
            grad_clip_norm=grad_clip_norm,
        )

        val_loss = _run_reward_epoch(
            state_model=state_model,
            reward_model=reward_model,
            loader=val_loader,
            device=device_t,
            criterion=criterion,
            optimizer=None,
        )

        history["reward_train_loss"].append(train_loss)
        history["reward_val_loss"].append(val_loss)

        print(f"[Reward][Epoch {epoch}] train={train_loss:.6f} val={val_loss:.6f}")

    return history
