# sequence_dataloader.py

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_npz_transitions(npz_path: str) -> Dict[str, np.ndarray]:
    data = np.load(npz_path)
    return {
        "states": data["states"].astype(np.float32),
        "actions": data["actions"].astype(np.float32),
        "rewards": data["rewards"].astype(np.float32),
        "next_states": data["next_states"].astype(np.float32),
    }


def build_sequence_chunks(
        transitions: Dict[str, np.ndarray],
        seq_len: int,
        stride: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build continuous sequence chunks for sequence training.

    Input arrays:
        states:      (N, state_dim)
        actions:     (N, action_dim)
        rewards:     (N, 1)
        next_states: (N, state_dim)

    Output arrays:
        states_seq:      (M, T, state_dim)
        actions_seq:     (M, T, action_dim)
        rewards_seq:     (M, T, 1)
        next_states_seq: (M, T, state_dim)

    Semantics:
        for each chunk:
            states[t:t+T]
            actions[t:t+T]
            rewards[t:t+T]
            next_states[t:t+T]
    """
    states = transitions["states"]
    actions = transitions["actions"]
    rewards = transitions["rewards"]
    next_states = transitions["next_states"]

    n = states.shape[0]
    if n < seq_len:
        raise ValueError(
            f"Not enough transitions for seq_len={seq_len}. Got n={n}."
        )

    if stride <= 0:
        raise ValueError("stride must be positive.")

    states_seq = []
    actions_seq = []
    rewards_seq = []
    next_states_seq = []

    for start in range(0, n - seq_len + 1, stride):
        end = start + seq_len

        states_seq.append(states[start:end])
        actions_seq.append(actions[start:end])
        rewards_seq.append(rewards[start:end])
        next_states_seq.append(next_states[start:end])

    return (
        np.asarray(states_seq, dtype=np.float32),
        np.asarray(actions_seq, dtype=np.float32),
        np.asarray(rewards_seq, dtype=np.float32),
        np.asarray(next_states_seq, dtype=np.float32),
    )


def build_sequence_dataloaders(
        npz_path: str,
        seq_len: int,
        stride: int,
        train_ratio: float,
        batch_size: int,
        shuffle: bool = False,
):
    """
    Build train/val dataloaders for sequence training.

    This is a time-ordered split:
        train = first train_ratio of chunks
        val   = remaining chunks

    Returns:
        train_loader
        val_loader
        dims: {"state_dim": ..., "action_dim": ...}
    """
    transitions = load_npz_transitions(npz_path)

    states_seq, actions_seq, rewards_seq, next_states_seq = build_sequence_chunks(
        transitions=transitions,
        seq_len=seq_len,
        stride=stride,
    )

    num_seq = states_seq.shape[0]
    split_idx = int(num_seq * train_ratio)
    split_idx = max(1, min(split_idx, num_seq - 1))

    train_states = torch.from_numpy(states_seq[:split_idx])
    train_actions = torch.from_numpy(actions_seq[:split_idx])
    train_rewards = torch.from_numpy(rewards_seq[:split_idx])
    train_next_states = torch.from_numpy(next_states_seq[:split_idx])

    val_states = torch.from_numpy(states_seq[split_idx:])
    val_actions = torch.from_numpy(actions_seq[split_idx:])
    val_rewards = torch.from_numpy(rewards_seq[split_idx:])
    val_next_states = torch.from_numpy(next_states_seq[split_idx:])

    train_dataset = TensorDataset(
        train_states,
        train_actions,
        train_rewards,
        train_next_states,
    )

    val_dataset = TensorDataset(
        val_states,
        val_actions,
        val_rewards,
        val_next_states,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    dims = {
        "state_dim": int(transitions["states"].shape[1]),
        "action_dim": int(transitions["actions"].shape[1]),
    }

    return train_loader, val_loader, dims
