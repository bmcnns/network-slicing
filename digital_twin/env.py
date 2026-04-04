from __future__ import annotations

import copy
import json
import os
import shutil
from datetime import datetime
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from digital_twin.dataloader import build_sequence_dataloaders
from digital_twin.dataset import (
    collect_transitions,
    fit_standardizers,
    normalize_transitions,
    save_all,
)
from digital_twin.models import (
    SequenceRewardPredictorDNN,
    SequenceStatePredictorLSTM,
)
from digital_twin.plotting import plot_all_rollout, plot_sequence_training_loss
from digital_twin.rollout import (
    load_npz_transitions,
    rollout_t1_sequence,
    rollout_tn_sequence,
    save_rollout_result,
)
from digital_twin.standardizer import Standardizer
from digital_twin.trainer import train_sequence_models


class DigitalTwinEnv(gym.Env):
    """
    Digital Twin gym environment.

    Important invariant:
    Internal rolling buffers are always stored in MODEL SPACE.

    - normalized_input=True  -> buffers contain normalized states/actions
    - normalized_input=False -> buffers contain raw states/actions

    This makes env.step() equivalent to one online step of rollout_tn_sequence:
    use the current rolling state/action window in model space, predict a full chunk,
    take only the last predicted next-state/reward, shift by one, append the predicted
    next state, and return outputs converted back to raw env space.
    """

    metadata = {"render_modes": []}

    DEFAULT_CONFIG: Dict[str, Any] = {
        # ===== Core =====
        "device": "cpu",
        "artifact_root": "artifacts/versions",
        "state_dim": None,
        "action_dim": None,
        "seq_len": 64,
        "state_use_action": True,
        "normalized_input": True,

        # ===== Collection =====
        "num_collect_steps": 50_000,
        "agent_path": None,
        "deterministic_action": True,
        "seed": 0,

        # ===== Dataloader =====
        "stride": 32,
        "train_ratio": 0.8,
        "batch_size": 16,
        "shuffle": False,

        # ===== State model =====
        "state_hidden_dim": 64,
        "state_num_layers": 1,
        "state_dropout": 0.0,

        # ===== Reward model =====
        "reward_hidden_dim": 64,
        "reward_num_layers": 1,
        "reward_mlp_hidden_dims": (128, 64),

        # ===== Training =====
        "epochs": 300,
        "lr": 1e-3,
        "grad_clip_norm": 1.0,

        # ===== Rollout =====
        "rollout_start_idx": 0,
        "rollout_num_chunks": 20,
        "rollout_horizon": 200,
        "rollout_sliding_window": 32,
        "rollout_lag_max": 20,

        # ===== Plot =====
        "plot_show": False,
        "plot_max_points": 500,
        "plot_smooth_window": 20,

        # ===== Runtime =====
        "current_version": None,
        "current_paths": None,
    }

    def __init__(self, config: Optional[dict] = None):
        super().__init__()

        self.config: Dict[str, Any] = copy.deepcopy(self.DEFAULT_CONFIG)

        self.device: str = "cpu"
        self.seq_len: int = 64
        self.state_dim: Optional[int] = None
        self.action_dim: Optional[int] = None
        self.state_use_action: bool = True
        self.normalized_input: bool = True

        self.state_model: Optional[torch.nn.Module] = None
        self.reward_model: Optional[torch.nn.Module] = None

        self.state_standardizer: Optional[Standardizer] = None
        self.reward_standardizer: Optional[Standardizer] = None
        self.action_standardizer: Optional[Standardizer] = None

        # Stored in MODEL SPACE.
        # obs_buffer holds the last seq_len states.
        # action_buffer holds the last seq_len-1 historical actions; the current
        # action passed into step() is appended transiently to form the full
        # action chunk of length seq_len. This matches rollout_tn_sequence,
        # whose first prediction uses actions[start_idx : start_idx + seq_len]
        # while the returned observation is states[start_idx + seq_len - 1].
        self.obs_buffer: Optional[np.ndarray] = None
        self.action_buffer: Optional[np.ndarray] = None
        self._cached_norm_transitions: Optional[Dict[str, np.ndarray]] = None

        self.observation_space = None
        self.action_space = None

        if config is not None:
            self.set_config(config)

    # =========================================================
    # Config only
    # =========================================================

    def set_config(self, config: dict) -> None:
        self.config.update(config)

        self.device = str(self.config["device"])
        self.seq_len = int(self.config["seq_len"])
        self.state_dim = self.config["state_dim"]
        self.action_dim = self.config["action_dim"]
        self.state_use_action = bool(self.config["state_use_action"])
        self.normalized_input = bool(self.config["normalized_input"])

        if self.state_dim is not None and self.action_dim is not None:
            self._build_spaces()

    # =========================================================
    # Versioning
    # =========================================================

    def _make_version_id(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _build_version_paths(self, version_id: str) -> Dict[str, str]:
        root = self.config["artifact_root"]
        version_root = os.path.join(root, version_id)

        data_dir = os.path.join(version_root, "data")
        model_dir = os.path.join(version_root, "models")
        log_dir = os.path.join(version_root, "logs")
        plot_dir = os.path.join(version_root, "plots")

        return {
            "version_id": version_id,
            "version_root": version_root,
            "data_dir": data_dir,
            "model_dir": model_dir,
            "log_dir": log_dir,
            "plot_dir": plot_dir,
            "raw_csv_path": os.path.join(data_dir, "transitions_raw.csv"),
            "norm_csv_path": os.path.join(data_dir, "transitions_norm.csv"),
            "norm_npz_path": os.path.join(data_dir, "transitions_norm.npz"),
            "state_std_path": os.path.join(data_dir, "state_standardizer.pkl"),
            "action_std_path": os.path.join(data_dir, "action_standardizer.pkl"),
            "reward_std_path": os.path.join(data_dir, "reward_standardizer.pkl"),
            "state_model_path": os.path.join(model_dir, "sequence_state_model.pt"),
            "reward_model_path": os.path.join(model_dir, "sequence_reward_model.pt"),
            "train_history_path": os.path.join(log_dir, "sequence_train_history.json"),
            "rollout_t1_path": os.path.join(log_dir, "sequence_rollout_t1.npz"),
            "rollout_tn_path": os.path.join(log_dir, "sequence_rollout_tn.npz"),
        }

    def _activate_version_paths(self, paths: Dict[str, str]) -> None:
        self.config["current_version"] = paths["version_id"]
        self.config["current_paths"] = paths

    def get_current_version_paths(self) -> Dict[str, str]:
        current = self.config.get("current_paths")
        return copy.deepcopy(current) if current is not None else {}

    def load_version(self, version_id: str) -> None:
        paths = self._build_version_paths(version_id)
        self._activate_version_paths(paths)
        self.load_assets(paths=paths)

    # =========================================================
    # Spaces / models / assets
    # =========================================================

    def _build_spaces(self) -> None:
        if self.state_dim is None or self.action_dim is None:
            return

        self.observation_space = spaces.Box(
            low=0.0,
            high=1000.0,
            shape=(int(self.state_dim),),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(int(self.action_dim),),
            dtype=np.float32,
        )

    def _infer_dims_from_npz(self, npz_path: str) -> None:
        transitions = load_npz_transitions(npz_path)
        self.state_dim = int(transitions["states"].shape[1])
        self.action_dim = int(transitions["actions"].shape[1])
        self._build_spaces()

    def _build_models(self) -> None:
        if self.state_dim is None or self.action_dim is None:
            raise ValueError("state_dim/action_dim must be known before building models.")

        self.state_model = SequenceStatePredictorLSTM(
            state_dim=int(self.state_dim),
            action_dim=int(self.action_dim),
            hidden_dim=int(self.config["state_hidden_dim"]),
            num_layers=int(self.config["state_num_layers"]),
            dropout=float(self.config["state_dropout"]),
            use_action=bool(self.state_use_action),
        )
        self.reward_model = SequenceRewardPredictorDNN(
            state_dim=int(self.state_dim),
            action_dim=int(self.action_dim),
            hidden_dim=int(self.config["reward_hidden_dim"]),
            num_layers=int(self.config["reward_num_layers"]),
            mlp_hidden_dims=tuple(self.config["reward_mlp_hidden_dims"]),
        )

        self.state_model.to(self.device).eval()
        self.reward_model.to(self.device).eval()

    def load_assets(self, paths: Optional[Dict[str, str]] = None) -> None:
        if paths is None:
            paths = self.config.get("current_paths")
        if not paths:
            raise ValueError("No version paths available. Train first or call load_version().")

        norm_npz_path = paths["norm_npz_path"]
        if self.state_dim is None or self.action_dim is None:
            self._infer_dims_from_npz(norm_npz_path)

        self._build_models()

        if os.path.exists(paths["state_model_path"]):
            self.state_model.load_state_dict(torch.load(paths["state_model_path"], map_location=self.device))
        if os.path.exists(paths["reward_model_path"]):
            self.reward_model.load_state_dict(torch.load(paths["reward_model_path"], map_location=self.device))
        if os.path.exists(paths["state_std_path"]):
            self.state_standardizer = Standardizer.load(paths["state_std_path"])
        if os.path.exists(paths["reward_std_path"]):
            self.reward_standardizer = Standardizer.load(paths["reward_std_path"])
        if os.path.exists(paths["action_std_path"]):
            self.action_standardizer = Standardizer.load(paths["action_std_path"])

        self.state_model.eval()
        self.reward_model.eval()

    # =========================================================
    # Training / rollout / plot helpers
    # =========================================================

    def _ensure_dirs_for_paths(self, paths: Dict[str, str]) -> None:
        for key in ("version_root", "data_dir", "model_dir", "log_dir", "plot_dir"):
            os.makedirs(paths[key], exist_ok=True)

    def _save_json(self, data: Dict[str, Any], path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _run_collect_to_version(self, paths: Dict[str, str]) -> None:
        self._ensure_dirs_for_paths(paths)

        raw_data = collect_transitions(
            num_steps=int(self.config["num_collect_steps"]),
            agent=self.config["agent_path"],
            deterministic=bool(self.config["deterministic_action"]),
            seed=int(self.config["seed"]),
        )
        standardizers = fit_standardizers(raw_data)
        norm_data = normalize_transitions(raw_data, standardizers)

        save_all(
            raw_data=raw_data,
            norm_data=norm_data,
            standardizers=standardizers,
            save_dir=paths["data_dir"],
        )

    def _run_train_to_version(self, paths: Dict[str, str], rebuild: bool = True) -> Dict[str, Any]:
        self._ensure_dirs_for_paths(paths)

        train_loader, val_loader, dims = build_sequence_dataloaders(
            npz_path=paths["norm_npz_path"],
            seq_len=int(self.config["seq_len"]),
            stride=int(self.config["stride"]),
            train_ratio=float(self.config["train_ratio"]),
            batch_size=int(self.config["batch_size"]),
            shuffle=bool(self.config["shuffle"]),
        )

        self.state_dim = dims["state_dim"]
        self.action_dim = dims["action_dim"]
        self._build_spaces()

        if rebuild or self.state_model is None or self.reward_model is None:
            self._build_models()

        history = train_sequence_models(
            state_model=self.state_model,
            reward_model=self.reward_model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=self.device,
            epochs=int(self.config["epochs"]),
            lr=float(self.config["lr"]),
            grad_clip_norm=self.config["grad_clip_norm"],
        )

        torch.save(self.state_model.state_dict(), paths["state_model_path"])
        torch.save(self.reward_model.state_dict(), paths["reward_model_path"])
        self._save_json(history, paths["train_history_path"])
        return history

    def _run_rollout_to_version(self, paths: Dict[str, str]) -> None:
        transitions = load_npz_transitions(paths["norm_npz_path"])
        state_standardizer = Standardizer.load(paths["state_std_path"])
        reward_standardizer = Standardizer.load(paths["reward_std_path"])

        self.load_assets(paths=paths)

        result_t1 = rollout_t1_sequence(
            state_model=self.state_model,
            reward_model=self.reward_model,
            transitions=transitions,
            seq_len=int(self.config["seq_len"]),
            device=self.device,
            state_standardizer=state_standardizer,
            reward_standardizer=reward_standardizer,
            start_idx=int(self.config["rollout_start_idx"]),
            num_chunks=int(self.config["rollout_num_chunks"]),
            normalized_input=bool(self.config["normalized_input"]),
            state_use_action=bool(self.state_use_action),
            sliding_window=int(self.config["rollout_sliding_window"]),
            lag_max=int(self.config["rollout_lag_max"]),
        )
        result_tn = rollout_tn_sequence(
            state_model=self.state_model,
            reward_model=self.reward_model,
            transitions=transitions,
            seq_len=int(self.config["seq_len"]),
            horizon=int(self.config["rollout_horizon"]),
            device=self.device,
            state_standardizer=state_standardizer,
            reward_standardizer=reward_standardizer,
            start_idx=int(self.config["rollout_start_idx"]),
            normalized_input=bool(self.config["normalized_input"]),
            state_use_action=bool(self.state_use_action),
            sliding_window=int(self.config["rollout_sliding_window"]),
            lag_max=int(self.config["rollout_lag_max"]),
        )

        save_rollout_result(result_t1, paths["rollout_t1_path"])
        save_rollout_result(result_tn, paths["rollout_tn_path"])

    def _run_plot_to_version(self, paths: Dict[str, str], run_rollout: bool = True) -> Dict[str, Any]:
        plot_paths: Dict[str, Any] = {}
        plot_paths["training"] = plot_sequence_training_loss(
            history_path=paths["train_history_path"],
            save_dir=paths["plot_dir"],
            show=bool(self.config["plot_show"]),
        )

        if run_rollout:
            plot_paths["t1"] = plot_all_rollout(
                rollout_npz_path=paths["rollout_t1_path"],
                save_dir=paths["plot_dir"],
                tag="dt_t1",
                show=bool(self.config["plot_show"]),
                max_points=self.config["plot_max_points"],
                smooth_window=self.config["plot_smooth_window"],
            )
            plot_paths["tn"] = plot_all_rollout(
                rollout_npz_path=paths["rollout_tn_path"],
                save_dir=paths["plot_dir"],
                tag="dt_tn",
                show=bool(self.config["plot_show"]),
                max_points=self.config["plot_max_points"],
                smooth_window=self.config["plot_smooth_window"],
            )
        return plot_paths

    # =========================================================
    # Gym API helpers
    # =========================================================

    def _load_norm_transitions_cache(self) -> Dict[str, np.ndarray]:
        if self._cached_norm_transitions is not None:
            return self._cached_norm_transitions
        current_paths = self.config.get("current_paths")
        if not current_paths or not os.path.exists(current_paths["norm_npz_path"]):
            raise RuntimeError("No normalized transition file available for dataset-based reset().")
        self._cached_norm_transitions = load_npz_transitions(current_paths["norm_npz_path"])
        return self._cached_norm_transitions

    def _require_ready(self) -> None:
        if self.state_model is None or self.reward_model is None:
            raise RuntimeError("Assets not loaded. Call load_assets() or train() first.")
        if self.state_dim is None or self.action_dim is None:
            raise RuntimeError("state_dim/action_dim unknown.")
        if self.observation_space is None or self.action_space is None:
            self._build_spaces()

    def _state_to_model_space(self, state_chunk: np.ndarray, already_in_model_space: bool = False) -> np.ndarray:
        state_chunk = np.asarray(state_chunk, dtype=np.float32)
        if not self.normalized_input or already_in_model_space:
            return state_chunk.astype(np.float32)
        if self.state_standardizer is None:
            raise RuntimeError("normalized_input=True but state standardizer is not loaded.")
        return self.state_standardizer.transform(state_chunk).astype(np.float32)

    def _action_to_model_space(self, action_chunk: np.ndarray, already_in_model_space: bool = False) -> np.ndarray:
        action_chunk = np.asarray(action_chunk, dtype=np.float32)
        if not self.normalized_input or already_in_model_space:
            return action_chunk.astype(np.float32)
        if self.action_standardizer is None:
            raise RuntimeError("normalized_input=True but action standardizer is not loaded.")
        return self.action_standardizer.transform(action_chunk).astype(np.float32)

    def _state_from_model_space(self, state_chunk: np.ndarray) -> np.ndarray:
        state_chunk = np.asarray(state_chunk, dtype=np.float32)
        if not self.normalized_input:
            return state_chunk.astype(np.float32)
        if self.state_standardizer is None:
            raise RuntimeError("normalized_input=True but state standardizer is not loaded.")
        return self.state_standardizer.inverse_transform(state_chunk).astype(np.float32)

    def _reward_from_model_space(self, reward_value: float) -> float:
        if not self.normalized_input:
            return float(reward_value)
        if self.reward_standardizer is None:
            raise RuntimeError("normalized_input=True but reward standardizer is not loaded.")
        reward = self.reward_standardizer.inverse_transform(
            np.array([[reward_value]], dtype=np.float32)
        )[0, 0]
        return float(reward)

    def _predict_chunk(self, state_chunk: np.ndarray, action_chunk: np.ndarray):
        states_t = torch.from_numpy(np.asarray(state_chunk, dtype=np.float32)[None, ...]).to(self.device).float()
        actions_t = torch.from_numpy(np.asarray(action_chunk, dtype=np.float32)[None, ...]).to(self.device).float()

        hidden = self.state_model.init_hidden(batch_size=1, device=self.device)

        with torch.no_grad():
            if self.state_use_action:
                pred_next_states_t, _, lstm_out = self.state_model(states_t, actions_t, hidden=hidden)
            else:
                pred_next_states_t, _, lstm_out = self.state_model(states_t, None, hidden=hidden)
            pred_rewards_t = self.reward_model(states_t, actions_t, lstm_out)

        pred_next_states = pred_next_states_t.squeeze(0).cpu().numpy().astype(np.float32)
        pred_rewards = pred_rewards_t.squeeze(0).cpu().numpy().astype(np.float32)
        return pred_next_states, pred_rewards

    # =========================================================
    # Gym API
    # =========================================================

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self.state_dim is None or self.action_dim is None:
            current_paths = self.config.get("current_paths")
            if current_paths is not None and os.path.exists(current_paths["norm_npz_path"]):
                self._infer_dims_from_npz(current_paths["norm_npz_path"])
            else:
                raise RuntimeError("Need dimensions or a loaded version before reset().")

        options = options or {}
        init_obs_seq = options.get("initial_observation_sequence")
        init_action_seq = options.get("initial_action_sequence")
        seqs_are_normalized = bool(options.get("initial_sequences_are_normalized", False))
        obs_is_normalized = bool(options.get("initial_observation_sequence_is_normalized", seqs_are_normalized))
        action_is_normalized = bool(options.get("initial_action_sequence_is_normalized", seqs_are_normalized))

        if init_obs_seq is None or init_action_seq is None:
            reset_mode = str(options.get("reset_mode", "dataset")).lower()
            if reset_mode in {"full_zero", "full_zeros"}:
                reset_mode = "zeros"
            if reset_mode not in {"dataset", "zeros"}:
                raise ValueError("reset_mode must be 'dataset', 'zeros', or 'full_zero'.")
        else:
            reset_mode = "provided"

        # State history always has length seq_len.
        if init_obs_seq is not None:
            obs_seq = np.asarray(init_obs_seq, dtype=np.float32)
            if obs_seq.shape != (self.seq_len, int(self.state_dim)):
                raise ValueError(
                    f"initial_observation_sequence must have shape ({self.seq_len}, {self.state_dim})"
                )
            self.obs_buffer = self._state_to_model_space(obs_seq, already_in_model_space=obs_is_normalized)
        elif reset_mode == "dataset":
            transitions = self._load_norm_transitions_cache()
            states = transitions["states"]
            n = int(states.shape[0])
            if n < self.seq_len:
                raise RuntimeError(f"Normalized transition dataset shorter than seq_len={self.seq_len}.")
            start_idx = int(options.get("dataset_start_idx", self.config.get("rollout_start_idx", 0) or 0))
            if start_idx < 0 or start_idx + self.seq_len > n:
                raise ValueError(f"dataset_start_idx out of range for seq_len={self.seq_len}, n={n}.")
            self.obs_buffer = states[start_idx:start_idx + self.seq_len].copy().astype(np.float32)
        else:
            self.obs_buffer = np.zeros((self.seq_len, int(self.state_dim)), dtype=np.float32)

        # Historical action history has length seq_len-1.
        hist_len = max(self.seq_len - 1, 0)
        if init_action_seq is not None:
            action_seq = np.asarray(init_action_seq, dtype=np.float32)
            # Accept either the full aligned chunk of length seq_len, or only the
            # past history of length seq_len-1. In the former case we drop the
            # last action because that should be supplied to step().
            if action_seq.shape == (self.seq_len, int(self.action_dim)):
                action_seq = action_seq[:-1]
            elif action_seq.shape != (hist_len, int(self.action_dim)):
                raise ValueError(
                    f"initial_action_sequence must have shape ({self.seq_len}, {self.action_dim}) or ({hist_len}, {self.action_dim})"
                )
            self.action_buffer = self._action_to_model_space(
                action_seq,
                already_in_model_space=action_is_normalized,
            )
        elif reset_mode == "dataset":
            transitions = self._load_norm_transitions_cache()
            actions = transitions["actions"]
            n = int(actions.shape[0])
            start_idx = int(options.get("dataset_start_idx", self.config.get("rollout_start_idx", 0) or 0))
            if start_idx < 0 or start_idx + self.seq_len > n:
                raise ValueError(f"dataset_start_idx out of range for seq_len={self.seq_len}, n={n}.")
            self.action_buffer = actions[start_idx:start_idx + hist_len].copy().astype(np.float32)
        else:
            self.action_buffer = np.zeros((hist_len, int(self.action_dim)), dtype=np.float32)

        obs = self._state_from_model_space(self.obs_buffer[-1])
        info = {
            "reset_mode": reset_mode,
            "state_buffer_model": self.obs_buffer.copy(),
            "action_history_model": self.action_buffer.copy(),
        }
        return obs.astype(np.float32), info

    def step(self, action):
        self._require_ready()

        if self.obs_buffer is None or self.action_buffer is None:
            raise RuntimeError("Call reset() before step().")

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != int(self.action_dim):
            raise ValueError(f"action must have shape ({self.action_dim},), got {action.shape}")

        action_model = self._action_to_model_space(action)

        # Online equivalent of rollout_tn_sequence:
        # state_chunk = last seq_len states
        # action_chunk = last seq_len-1 historical actions + current action
        if self.seq_len > 1:
            action_chunk_model = np.concatenate([self.action_buffer, action_model[None, :]], axis=0)
        else:
            action_chunk_model = action_model[None, :]
        state_chunk_model = self.obs_buffer.copy()

        if action_chunk_model.shape != (self.seq_len, int(self.action_dim)):
            raise RuntimeError(
                f"Aligned action chunk has wrong shape: expected ({self.seq_len}, {self.action_dim}), got {action_chunk_model.shape}"
            )

        pred_next_chunk, pred_reward_chunk = self._predict_chunk(
            state_chunk=state_chunk_model,
            action_chunk=action_chunk_model,
        )

        pred_next_model = pred_next_chunk[-1]
        pred_reward_model = float(pred_reward_chunk[-1, 0])

        # Update histories for the NEXT decision.
        self.obs_buffer = np.concatenate([self.obs_buffer[1:], pred_next_model[None, :]], axis=0)
        if self.seq_len > 1:
            self.action_buffer = np.concatenate([self.action_buffer[1:], action_model[None, :]], axis=0)
        else:
            self.action_buffer = np.zeros((0, int(self.action_dim)), dtype=np.float32)

        next_obs = self._state_from_model_space(pred_next_model)
        reward = self._reward_from_model_space(pred_reward_model)

        terminated = False
        truncated = False
        info = {
            "pred_next_model": pred_next_model.astype(np.float32),
            "pred_reward_model": float(pred_reward_model),
            "state_chunk_model": state_chunk_model.copy(),
            "action_chunk_model": action_chunk_model.copy(),
            "state_buffer_model": self.obs_buffer.copy(),
            "action_history_model": self.action_buffer.copy(),
        }

        return next_obs.astype(np.float32), float(reward), terminated, truncated, info

    # =========================================================
    # Orchestration
    # =========================================================

    def train(
        self,
        rebuild: bool = True,
        run_collect: bool = False,
        run_rollout: bool = True,
        run_plot: bool = True,
        sim_agent: str = None,
    ) -> Dict[str, Any]:
        """
        Create a new version each time.
        """
        version_id = self._make_version_id()
        paths = self._build_version_paths(version_id)
        self._activate_version_paths(paths)
        self._ensure_dirs_for_paths(paths)

        if run_collect:
            if sim_agent is not None:
                self.config["agent_path"] = sim_agent
            self._run_collect_to_version(paths)
        else:
            source_npz = self.config.get("source_norm_npz_path")
            source_state_std = self.config.get("source_state_std_path")
            source_action_std = self.config.get("source_action_std_path")
            source_reward_std = self.config.get("source_reward_std_path")

            if source_npz is None:
                raise ValueError("run_collect=False requires source_norm_npz_path in config.")

            shutil.copy2(source_npz, paths["norm_npz_path"])

            if self.config.get("source_raw_csv_path") and os.path.exists(self.config["source_raw_csv_path"]):
                shutil.copy2(self.config["source_raw_csv_path"], paths["raw_csv_path"])
            if self.config.get("source_norm_csv_path") and os.path.exists(self.config["source_norm_csv_path"]):
                shutil.copy2(self.config["source_norm_csv_path"], paths["norm_csv_path"])
            if source_state_std and os.path.exists(source_state_std):
                shutil.copy2(source_state_std, paths["state_std_path"])
            if source_action_std and os.path.exists(source_action_std):
                shutil.copy2(source_action_std, paths["action_std_path"])
            if source_reward_std and os.path.exists(source_reward_std):
                shutil.copy2(source_reward_std, paths["reward_std_path"])

        history = self._run_train_to_version(paths, rebuild=rebuild)

        if run_rollout:
            self._run_rollout_to_version(paths)

        plot_paths = {}
        if run_plot:
            plot_paths = self._run_plot_to_version(paths, run_rollout=run_rollout)

        self.load_assets(paths=paths)

        return {
            "version_id": version_id,
            "paths": copy.deepcopy(paths),
            "history": history,
            "plot_paths": plot_paths,
        }