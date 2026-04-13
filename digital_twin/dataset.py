# collect_transitions.py

from __future__ import annotations

import json
import os

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from simulator.gym_env import NetworkSlicingGymContinuous
from digital_twin.standardizer import Standardizer


def ensure_1d_float_array(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.reshape(-1)


def collect_transitions(
        num_steps: int,
        agent: Optional[str] = "models/PPO-Network-Slicing",
        deterministic: bool = True,
        seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Collect transitions from the real simulator.

    Stored fields:
        - states:      (N, state_dim)
        - actions:     (N, action_dim)
        - rewards:     (N, 1)
        - next_states: (N, state_dim)

    Notes:
    - We do NOT store done.
    - If env returns terminated/truncated, we reset internally and continue.
    """
    env = NetworkSlicingGymContinuous()

    if seed is not None:
        obs, _ = env.reset(seed=seed)
    else:
        obs, _ = env.reset()

    if agent:
        agent = PPO.load(agent, device="cpu")

    states: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    rewards: List[List[float]] = []
    next_states: List[np.ndarray] = []

    for _ in range(num_steps):
        state = ensure_1d_float_array(obs)

        if agent is None:
            action = env.action_space.sample()
        else:

            action, _ = agent.predict(obs, deterministic=deterministic)

        action = ensure_1d_float_array(action)

        next_obs, reward, terminated, truncated, info = env.step(action)
        next_state = ensure_1d_float_array(next_obs)
        applied_action = ensure_1d_float_array(
            info.get("applied_action_ratio", action)
        )

        states.append(state)
        actions.append(applied_action)
        rewards.append([float(reward)])
        next_states.append(next_state)

        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs

    states_np = np.asarray(states, dtype=np.float32)
    actions_np = np.asarray(actions, dtype=np.float32)
    rewards_np = np.asarray(rewards, dtype=np.float32)
    next_states_np = np.asarray(next_states, dtype=np.float32)

    return {
        "states": states_np,
        "actions": actions_np,
        "rewards": rewards_np,
        "next_states": next_states_np,
    }


def transitions_to_dataframe(data: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Convert transition arrays to a flat dataframe for CSV saving.

    Column naming:
        s_0, s_1, ...
        a_0, a_1, ...
        r
        ns_0, ns_1, ...
    """
    states = data["states"]
    actions = data["actions"]
    rewards = data["rewards"]
    next_states = data["next_states"]

    n = states.shape[0]
    if not (actions.shape[0] == n and rewards.shape[0] == n and next_states.shape[0] == n):
        raise ValueError("All transition arrays must have same first dimension.")

    df_dict: Dict[str, np.ndarray] = {}

    for i in range(states.shape[1]):
        df_dict[f"s_{i}"] = states[:, i]

    for i in range(actions.shape[1]):
        df_dict[f"a_{i}"] = actions[:, i]

    df_dict["r"] = rewards[:, 0]

    for i in range(next_states.shape[1]):
        df_dict[f"ns_{i}"] = next_states[:, i]

    return pd.DataFrame(df_dict)


def fit_standardizers(
        data: Dict[str, np.ndarray],
) -> Dict[str, Standardizer]:
    """
    Fit standardizers.

    Final fixed choice here:
    - normalize states
    - normalize rewards
    - do NOT normalize actions
    """
    state_stack = np.concatenate([data["states"], data["next_states"]], axis=0)
    reward_stack = data["rewards"]
    action_stack = data["actions"]

    state_std = Standardizer.fit(state_stack)
    reward_std = Standardizer.fit(reward_stack)
    action_std = Standardizer.fit(action_stack)

    return {
        "state": state_std,
        "reward": reward_std,
        "action": action_std,
    }


def normalize_transitions(
        data: Dict[str, np.ndarray],
        standardizers: Dict[str, Standardizer],
) -> Dict[str, np.ndarray]:
    """
    Normalize transitions using fitted standardizers.

    Fixed final rule:
    - states      -> normalized
    - next_states -> normalized
    - rewards     -> normalized
    - actions     -> unchanged
    """
    state_std = standardizers["state"]
    reward_std = standardizers["reward"]
    action_std = standardizers["action"]

    states_norm = state_std.transform(data["states"])
    next_states_norm = state_std.transform(data["next_states"])
    rewards_norm = reward_std.transform(data["rewards"])
    actions_norm = action_std.transform(data["actions"])

    return {
        "states": states_norm.astype(np.float32),
        "actions": actions_norm.astype(np.float32),
        "rewards": rewards_norm.astype(np.float32),
        "next_states": next_states_norm.astype(np.float32),
    }


def save_npz(data: Dict[str, np.ndarray], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        states=data["states"],
        actions=data["actions"],
        rewards=data["rewards"],
        next_states=data["next_states"],
    )


def save_metadata(
        raw_data: Dict[str, np.ndarray],
        norm_data: Dict[str, np.ndarray],
        save_dir: str,
) -> None:
    meta = {
        "num_transitions": int(raw_data["states"].shape[0]),
        "state_dim": int(raw_data["states"].shape[1]),
        "action_dim": int(raw_data["actions"].shape[1]),
        "reward_dim": int(raw_data["rewards"].shape[1]),
        "raw_csv": os.path.join(save_dir, "transitions_raw.csv"),
        "norm_csv": os.path.join(save_dir, "transitions_norm.csv"),
        "norm_npz": os.path.join(save_dir, "transitions_norm.npz"),
        "state_standardizer": os.path.join(save_dir, "state_standardizer.pkl"),
        "reward_standardizer": os.path.join(save_dir, "reward_standardizer.pkl"),
        "action_standardizer": os.path.join(save_dir, "action_standardizer.pkl"),
    }

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "transitions_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def save_all(
        raw_data: Dict[str, np.ndarray],
        norm_data: Dict[str, np.ndarray],
        standardizers: Dict[str, Standardizer],
        save_dir: str = "data",
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    raw_csv_path = os.path.join(save_dir, "transitions_raw.csv")
    norm_csv_path = os.path.join(save_dir, "transitions_norm.csv")
    norm_npz_path = os.path.join(save_dir, "transitions_norm.npz")

    state_std_path = os.path.join(save_dir, "state_standardizer.pkl")
    reward_std_path = os.path.join(save_dir, "reward_standardizer.pkl")
    action_std_path = os.path.join(save_dir, "action_standardizer.pkl")

    transitions_to_dataframe(raw_data).to_csv(raw_csv_path, index=False)
    transitions_to_dataframe(norm_data).to_csv(norm_csv_path, index=False)
    save_npz(norm_data, norm_npz_path)

    standardizers["state"].save(state_std_path)
    standardizers["reward"].save(reward_std_path)
    standardizers["action"].save(action_std_path)

    save_metadata(raw_data, norm_data, save_dir)
