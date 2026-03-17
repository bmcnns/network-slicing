import os
import json
import gymnasium as gym
import numpy as np
import torch

from StatePredictor.data_builder import (
    Standardizer,
    TimeSeriesData,
    DatasetSplits,
    split_contiguous,
    build_datasets,
)
from StatePredictor.train import train_from_splits, load_trained_model


class DigitalTwin(gym.Env):
    """The state-predictor LSTM network that simulates the base-station's dynamics."""

    def __init__(
        self,
        model_path: str = "./model_components/lstm_dt.pt",
        scaler_path: str = "./model_components/standardizer.json",
        run_config_path: str = "./model_components/run_config.json",
        log_path: str = "./log/train_log.jsonl",
        window_size: int = 32,
        stride: int = 1,
        split=(0.7, 0.15, 0.15),
        standardize: bool = True,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        epochs: int = 50,
        batch_size: int = 128,
        lr: float = 1e-3,
        max_pairs: int = 16,
        device: str = "cpu",
    ):
        """Initialize the digital twin."""
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1000.0,
            shape=(3,),
            dtype=np.float32
        )

        self.action_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )

        # -------- artifact paths --------
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.run_config_path = run_config_path
        self.log_path = log_path

        # -------- train config --------
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.split = split
        self.standardize = bool(standardize)

        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)

        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.max_pairs = int(max_pairs)

        self.device = torch.device(device)

        # -------- runtime members --------
        self.model = None
        self.scaler = None
        self.run_cfg = None

        # optional: auto-load if files already exist
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self._reload_artifacts()

    def _reload_artifacts(self):
        """Reload trained model + scaler into class members."""
        self.model, model_cfg = load_trained_model(
            model_path=self.model_path,
            device=self.device,
        )
        self.scaler = Standardizer.load(self.scaler_path)

        if os.path.exists(self.run_config_path):
            with open(self.run_config_path, "r", encoding="utf-8") as f:
                self.run_cfg = json.load(f)
        else:
            self.run_cfg = model_cfg

    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
        return np.zeros((3,), dtype=np.float32), {}

    def retrain(self, states, rewards):
        """
        Retrain the twin network with updated data.

        Current version:
          - trains state predictor only
          - rewards argument is kept for API compatibility, but not used yet

        Args:
            states: np.ndarray of shape (T, D)
            rewards: kept for future reward-head training
        """
        states = np.asarray(states, dtype=np.float32)

        if states.ndim != 2:
            raise ValueError(f"`states` must be 2D (T,D), got shape {states.shape}")

        if states.shape[0] <= self.window_size + 1:
            raise ValueError(
                f"Need more time steps than window_size. "
                f"Got T={states.shape[0]}, window_size={self.window_size}"
            )

        # rewards currently unused, but keep basic sanity check if provided
        if rewards is not None:
            rewards = np.asarray(rewards)
            if rewards.shape[0] != states.shape[0] and rewards.shape[0] != states.shape[0] - 1:
                raise ValueError(
                    f"`rewards` length should usually match T or T-1. "
                    f"Got rewards.shape={rewards.shape}, states.shape={states.shape}"
                )

        # -------- build dataset splits from provided states --------
        raw = TimeSeriesData(
            states=states,
            meta={
                "source": "digital_twin_retrain",
                "split": list(self.split),
                "standardize": self.standardize,
                "window_size": self.window_size,
                "stride": self.stride,
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
            scaler_save_path=self.scaler_path,   # important: save scaler for later reload
        )

        splits = DatasetSplits(
            raw=raw,
            train=train_ds,
            val=val_ds,
            test=test_ds,
        )

        input_dim = int(states.shape[1])

        extra_run_cfg = {
            "device": str(self.device),
            "input_dim": input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "loss": "MSE",
            "model_path": self.model_path,
            "log_path": self.log_path,
            "retrain_mode": True,
        }

        # -------- train using existing training module --------
        self.model, self.run_cfg = train_from_splits(
            splits=splits,
            device=self.device,
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            log_path=self.log_path,
            model_path=self.model_path,
            run_config_path=self.run_config_path,
            max_pairs=self.max_pairs,
            extra_run_cfg=extra_run_cfg,
        )

        # -------- reload scaler/model from disk to keep env in sync --------
        self._reload_artifacts()

        return {
            "status": "ok",
            "num_states": int(states.shape[0]),
            "state_dim": int(states.shape[1]),
            "model_path": self.model_path,
            "scaler_path": self.scaler_path,
            "run_config_path": self.run_config_path,
        }

    def step(self, action):
        # Normalize action to ensure sum equals 1.0
        obs, rew, term, trunc, info = [], -1.0, False, False, {}
        return obs, rew, term, trunc, info