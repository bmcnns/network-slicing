from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import numpy as np
import torch

from digital_twin.metrics import summarize_vector_metrics


# =========================================================
# Config for rollout metrics
# =========================================================

DEFAULT_SLIDING_WINDOW = 50
DEFAULT_LAG_MAX = 50


# =========================================================
# IO
# =========================================================

def load_npz_transitions(npz_path: str) -> Dict[str, np.ndarray]:
    data = np.load(npz_path)
    return {
        "states": data["states"].astype(np.float32),
        "actions": data["actions"].astype(np.float32),
        "rewards": data["rewards"].astype(np.float32),
        "next_states": data["next_states"].astype(np.float32),
    }


def save_rollout_result(result: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    np_dict = {}
    meta_dict = {}

    for k, v in result.items():
        if isinstance(v, np.ndarray):
            np_dict[k] = v
        else:
            meta_dict[k] = v

    np.savez_compressed(path, **np_dict)

    meta_path = os.path.splitext(path)[0] + "_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2)


# =========================================================
# Helpers
# =========================================================

def _to_tensor_seq(x: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    x: (T, D) -> tensor (1, T, D)
    """
    return torch.from_numpy(np.asarray(x, dtype=np.float32)[None, ...]).to(device).float()


def _inverse_state_if_needed(x: np.ndarray, state_standardizer, normalized_input: bool) -> np.ndarray:
    if normalized_input and state_standardizer is not None:
        return state_standardizer.inverse_transform(x).astype(np.float32)
    return x.astype(np.float32)


def _inverse_reward_if_needed(x: np.ndarray, reward_standardizer, normalized_input: bool) -> np.ndarray:
    if normalized_input and reward_standardizer is not None:
        return reward_standardizer.inverse_transform(x).astype(np.float32)
    return x.astype(np.float32)


def _sequence_predict_chunk(
    state_model,
    reward_model,
    state_chunk: np.ndarray,          # (T, D)
    action_chunk: np.ndarray,         # (T, A)
    device: torch.device,
    state_use_action: bool = True,
):
    """
    Predict a whole chunk.

    Returns:
        pred_next_states_chunk: (T, D)
        pred_rewards_chunk:     (T, 1)
    """
    state_model.eval()
    reward_model.eval()

    states_t = _to_tensor_seq(state_chunk, device)   # (1, T, D)
    actions_t = _to_tensor_seq(action_chunk, device) # (1, T, A)

    hidden = state_model.init_hidden(batch_size=1, device=device)

    with torch.no_grad():
        if state_use_action:
            pred_next_states_t, _, lstm_out = state_model(states_t, actions_t, hidden=hidden)
        else:
            pred_next_states_t, _, lstm_out = state_model(states_t, None, hidden=hidden)

        pred_rewards_t = reward_model(states_t, actions_t, lstm_out)

    pred_next_states = pred_next_states_t.squeeze(0).cpu().numpy().astype(np.float32)  # (T, D)
    pred_rewards = pred_rewards_t.squeeze(0).cpu().numpy().astype(np.float32)           # (T, 1)

    return pred_next_states, pred_rewards


def _attach_rollout_metrics(
    result: Dict[str, Any],
    sliding_window: int = DEFAULT_SLIDING_WINDOW,
    lag_max: int = DEFAULT_LAG_MAX,
) -> Dict[str, Any]:
    """
    Add dynamics-learning metrics into rollout result dict.
    """
    true_next = np.asarray(result["true_next_states"], dtype=np.float32)
    pred_next = np.asarray(result["pred_next_states"], dtype=np.float32)

    true_reward = np.asarray(result["true_rewards"], dtype=np.float32).reshape(-1)
    pred_reward = np.asarray(result["pred_rewards"], dtype=np.float32).reshape(-1)

    state_metrics = summarize_vector_metrics(
        y_true=true_next,
        y_pred=pred_next,
        sliding_window=sliding_window,
        lag_max=lag_max,
    )

    reward_metrics = summarize_vector_metrics(
        y_true=true_reward,
        y_pred=pred_reward,
        sliding_window=sliding_window,
        lag_max=lag_max,
    )

    # -------------------------
    # State metrics
    # -------------------------
    result["state_rse_mean"] = float(state_metrics["rse_mean"])
    result["state_rse_flattened"] = float(state_metrics["rse_flattened"])
    result["state_rse_per_dim"] = state_metrics["rse_per_dim"].astype(np.float32)

    result["state_corr_mean"] = float(state_metrics["corr_mean"])
    result["state_corr_per_dim"] = state_metrics["corr_per_dim"].astype(np.float32)

    result["state_amp_ratio_mean"] = float(state_metrics["amp_ratio_mean"])
    result["state_amp_ratio_per_dim"] = state_metrics["amp_ratio_per_dim"].astype(np.float32)

    result["state_delta_rse_mean"] = float(state_metrics["delta_rse_mean"])
    result["state_delta_rse_flattened"] = float(state_metrics["delta_rse_flattened"])
    result["state_delta_rse_per_dim"] = state_metrics["delta_rse_per_dim"].astype(np.float32)

    result["state_delta_corr_mean"] = float(state_metrics["delta_corr_mean"])
    result["state_delta_corr_per_dim"] = state_metrics["delta_corr_per_dim"].astype(np.float32)

    result["state_best_lag_mean"] = float(state_metrics["best_lag_mean"])
    result["state_best_lag_per_dim"] = state_metrics["best_lag_per_dim"].astype(np.float32)

    result["state_best_lag_corr_mean"] = float(state_metrics["best_lag_corr_mean"])
    result["state_best_lag_corr_per_dim"] = state_metrics["best_lag_corr_per_dim"].astype(np.float32)

    result["state_sliding_rse_per_dim"] = state_metrics["sliding_rse_per_dim"].astype(np.float32)
    result["state_sliding_corr_per_dim"] = state_metrics["sliding_corr_per_dim"].astype(np.float32)

    # -------------------------
    # Reward metrics
    # -------------------------
    result["reward_rse"] = float(reward_metrics["rse"])
    result["reward_corr"] = float(reward_metrics["corr"])
    result["reward_amp_ratio"] = float(reward_metrics["amp_ratio"])

    result["reward_delta_rse"] = float(reward_metrics["delta_rse"])
    result["reward_delta_corr"] = float(reward_metrics["delta_corr"])

    result["reward_best_lag"] = float(reward_metrics["best_lag"])
    result["reward_best_lag_corr"] = float(reward_metrics["best_lag_corr"])

    result["reward_sliding_rse"] = reward_metrics["sliding_rse"].astype(np.float32)
    result["reward_sliding_corr"] = reward_metrics["sliding_corr"].astype(np.float32)

    result["metrics_sliding_window"] = int(sliding_window)
    result["metrics_lag_max"] = int(lag_max)

    return result


# =========================================================
# Rollout
# =========================================================

def rollout_t1_sequence(
    state_model,
    reward_model,
    transitions: Dict[str, np.ndarray],
    seq_len: int,
    device: str = "cpu",
    state_standardizer=None,
    reward_standardizer=None,
    start_idx: int = 0,
    num_chunks: Optional[int] = None,
    normalized_input: bool = True,
    state_use_action: bool = True,
    sliding_window: int = DEFAULT_SLIDING_WINDOW,
    lag_max: int = DEFAULT_LAG_MAX,
) -> Dict[str, Any]:
    """
    T+1 rollout for sequence training.

    For each chunk:
        input  = true states[start:start+seq_len]
        target = true next_states[start:start+seq_len]

    We compare the whole predicted chunk against the whole true chunk.
    """
    device_t = torch.device(device)

    states = transitions["states"]
    actions = transitions["actions"]
    rewards = transitions["rewards"]
    next_states = transitions["next_states"]

    n = states.shape[0]
    max_start = n - seq_len + 1
    if max_start <= 0:
        raise ValueError("Not enough data for seq_len.")

    if start_idx < 0 or start_idx >= max_start:
        raise ValueError("start_idx out of range")

    if num_chunks is None:
        num_chunks = max_start - start_idx

    end_idx = min(start_idx + num_chunks, max_start)

    pred_next_all = []
    true_next_all = []
    pred_reward_all = []
    true_reward_all = []
    input_state_all = []
    input_action_all = []

    for start in range(start_idx, end_idx):
        end = start + seq_len

        state_chunk = states[start:end]         # (T, D)
        action_chunk = actions[start:end]       # (T, A)
        true_next_chunk = next_states[start:end]
        true_reward_chunk = rewards[start:end]

        pred_next_chunk, pred_reward_chunk = _sequence_predict_chunk(
            state_model=state_model,
            reward_model=reward_model,
            state_chunk=state_chunk,
            action_chunk=action_chunk,
            device=device_t,
            state_use_action=state_use_action,
        )

        input_state_all.append(state_chunk)
        input_action_all.append(action_chunk)
        pred_next_all.append(pred_next_chunk)
        true_next_all.append(true_next_chunk)
        pred_reward_all.append(pred_reward_chunk)
        true_reward_all.append(true_reward_chunk)

    input_states = np.concatenate(input_state_all, axis=0)     # (N_eval, D)
    input_actions = np.concatenate(input_action_all, axis=0)   # (N_eval, A)
    pred_next = np.concatenate(pred_next_all, axis=0)          # (N_eval, D)
    true_next = np.concatenate(true_next_all, axis=0)          # (N_eval, D)
    pred_reward = np.concatenate(pred_reward_all, axis=0)      # (N_eval, 1)
    true_reward = np.concatenate(true_reward_all, axis=0)      # (N_eval, 1)

    input_states_out = _inverse_state_if_needed(input_states, state_standardizer, normalized_input)
    pred_next_out = _inverse_state_if_needed(pred_next, state_standardizer, normalized_input)
    true_next_out = _inverse_state_if_needed(true_next, state_standardizer, normalized_input)

    pred_reward_out = _inverse_reward_if_needed(pred_reward, reward_standardizer, normalized_input)
    true_reward_out = _inverse_reward_if_needed(true_reward, reward_standardizer, normalized_input)

    result = {
        "mode": "sequence_t1",
        "seq_len": int(seq_len),
        "start_idx": int(start_idx),
        "num_chunks": int(end_idx - start_idx),
        "input_states": input_states_out.astype(np.float32),
        "input_actions": input_actions.astype(np.float32),
        "pred_next_states": pred_next_out.astype(np.float32),
        "true_next_states": true_next_out.astype(np.float32),
        "pred_rewards": pred_reward_out.astype(np.float32),
        "true_rewards": true_reward_out.astype(np.float32),
    }


    return _attach_rollout_metrics(
        result=result,
        sliding_window=sliding_window,
        lag_max=lag_max,
    )


def rollout_tn_sequence(
    state_model,
    reward_model,
    transitions: Dict[str, np.ndarray],
    seq_len: int,
    horizon: int,
    device: str = "cpu",
    state_standardizer=None,
    reward_standardizer=None,
    start_idx: int = 0,
    normalized_input: bool = True,
    state_use_action: bool = True,
    sliding_window: int = DEFAULT_SLIDING_WINDOW,
    lag_max: int = DEFAULT_LAG_MAX,
) -> Dict[str, Any]:
    """
    T+N rollout for sequence training.

    Initial chunk:
        true states[start_idx : start_idx + seq_len]

    Then autoregressively:
        - predict the whole next-state chunk
        - use ONLY the last predicted next state as the next generated point
        - shift the buffer by 1
        - append predicted last point
        - actions remain true action sequence from dataset

    Targets:
        compare predicted generated states against the true future next_states.
    """
    device_t = torch.device(device)

    states = transitions["states"]
    actions = transitions["actions"]
    rewards = transitions["rewards"]
    next_states = transitions["next_states"]

    n = states.shape[0]
    if start_idx < 0 or start_idx + seq_len > n:
        raise ValueError("Invalid start_idx or seq_len for initial chunk.")

    max_future = n - (start_idx + seq_len) + 1
    if max_future <= 0:
        raise ValueError("No future steps available after initial chunk.")

    horizon = min(horizon, max_future)

    # Initial true chunk
    state_buffer = states[start_idx:start_idx + seq_len].copy()    # (T, D)

    pred_next_list = []
    true_next_list = []
    pred_reward_list = []
    true_reward_list = []
    input_last_state_list = []
    input_last_action_list = []

    for step in range(horizon):
        chunk_start = start_idx + step
        chunk_end = chunk_start + seq_len

        action_chunk = actions[chunk_start:chunk_end]  # always use true action chunk

        pred_next_chunk, pred_reward_chunk = _sequence_predict_chunk(
            state_model=state_model,
            reward_model=reward_model,
            state_chunk=state_buffer,
            action_chunk=action_chunk,
            device=device_t,
            state_use_action=state_use_action,
        )

        # Use the LAST predicted next state / reward as the rollout output
        pred_next_t = pred_next_chunk[-1]       # (D,)
        pred_reward_t = pred_reward_chunk[-1]   # (1,)

        # True target aligned with the last position
        true_next_t = next_states[chunk_end - 1]   # (D,)
        true_reward_t = rewards[chunk_end - 1]     # (1,)

        input_last_state_list.append(state_buffer[-1].copy())
        input_last_action_list.append(action_chunk[-1].copy())

        pred_next_list.append(pred_next_t.copy())
        true_next_list.append(true_next_t.copy())
        pred_reward_list.append(pred_reward_t.copy())
        true_reward_list.append(true_reward_t.copy())

        # Shift buffer and append predicted state
        state_buffer = np.concatenate(
            [state_buffer[1:], pred_next_t[None, :]],
            axis=0,
        )

    input_states = np.asarray(input_last_state_list, dtype=np.float32)
    input_actions = np.asarray(input_last_action_list, dtype=np.float32)
    pred_next = np.asarray(pred_next_list, dtype=np.float32)
    true_next = np.asarray(true_next_list, dtype=np.float32)
    pred_reward = np.asarray(pred_reward_list, dtype=np.float32).reshape(-1, 1)
    true_reward = np.asarray(true_reward_list, dtype=np.float32).reshape(-1, 1)

    input_states_out = _inverse_state_if_needed(input_states, state_standardizer, normalized_input)
    pred_next_out = _inverse_state_if_needed(pred_next, state_standardizer, normalized_input)
    true_next_out = _inverse_state_if_needed(true_next, state_standardizer, normalized_input)

    pred_reward_out = _inverse_reward_if_needed(pred_reward, reward_standardizer, normalized_input)
    true_reward_out = _inverse_reward_if_needed(true_reward, reward_standardizer, normalized_input)

    result = {
        "mode": "sequence_tn",
        "seq_len": int(seq_len),
        "start_idx": int(start_idx),
        "horizon": int(horizon),
        "input_states": input_states_out.astype(np.float32),
        "input_actions": input_actions.astype(np.float32),
        "pred_next_states": pred_next_out.astype(np.float32),
        "true_next_states": true_next_out.astype(np.float32),
        "pred_rewards": pred_reward_out.astype(np.float32),
        "true_rewards": true_reward_out.astype(np.float32),
    }

    return _attach_rollout_metrics(
        result=result,
        sliding_window=sliding_window,
        lag_max=lag_max,
    )
