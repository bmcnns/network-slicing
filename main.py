from __future__ import annotations

import os
import torch
import gymnasium as gym
import sys

sys.modules["gym"] = gym

from digital_twin.pipeline import run_collect, run_train, run_rollout, run_plot

# =========================================================
# Global Config
# =========================================================

MODE = "rollout"  # "collect" | "train" | "rollout" | "plot" | "all"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
CONFIG_ID = 9 # todo: update before training
# -------------------------
# Data collection
# -------------------------
NUM_COLLECT_STEPS = 50_000
AGENT_PATH = "Simulator/models/PPO-Network-Slicing"
DETERMINISTIC_ACTION = True

# -------------------------
# Paths
# -------------------------
DATA_DIR = "artifacts/data"
MODEL_DIR = f"artifacts/config_{CONFIG_ID}/checkpoints"
LOG_DIR = f"artifacts/config_{CONFIG_ID}/logs"
PLOT_DIR = f"artifacts/config_{CONFIG_ID}/plots"
STANDARDIZER_DIR = "artifacts/standardizers"

RAW_CSV_PATH = os.path.join(DATA_DIR, "transitions_raw.csv")
NORM_CSV_PATH = os.path.join(DATA_DIR, "transitions_norm.csv")
NORM_NPZ_PATH = os.path.join(DATA_DIR, "transitions_norm.npz")

STATE_STD_PATH = os.path.join(STANDARDIZER_DIR, "state_standardizer.pkl")
ACTION_STD_PATH = os.path.join(STANDARDIZER_DIR, "action_standardizer.pkl")
REWARD_STD_PATH = os.path.join(STANDARDIZER_DIR, "reward_standardizer.pkl")

STATE_MODEL_PATH = os.path.join(MODEL_DIR, "sequence_state_model.pt")
REWARD_MODEL_PATH = os.path.join(MODEL_DIR, "sequence_reward_model.pt")
TRAIN_HISTORY_PATH = os.path.join(LOG_DIR, "sequence_train_history.json")

ROLLOUT_T1_PATH = os.path.join(LOG_DIR, "sequence_rollout_t1.npz")
ROLLOUT_TN_PATH = os.path.join(LOG_DIR, "sequence_rollout_tn.npz")

# -------------------------
# Sequence Dataloader
# -------------------------
SEQ_LEN = 64
STRIDE = 32
TRAIN_RATIO = 0.8
BATCH_SIZE = 16
SHUFFLE = False

<<<<<<< HEAD
# -------------------------
# Sequence Model
# -------------------------
STATE_HIDDEN_DIM = 64
STATE_NUM_LAYERS = 1
STATE_DROPOUT = 0.0
STATE_USE_ACTION = True  # only for state predictor
=======
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )
>>>>>>> c967e10 (network simulator v2)

REWARD_HIDDEN_DIM = 64
REWARD_NUM_LAYERS = 1
REWARD_MLP_HIDDEN_DIMS = (128, 64)

# -------------------------
# Train
# -------------------------
EPOCHS = 1000
LR = 1e-3
GRAD_CLIP_NORM = 1.0

# -------------------------
# Rollout
# -------------------------
ROLLOUT_START_IDX = 0
ROLLOUT_NUM_CHUNKS = 20
ROLLOUT_HORIZON = 200
ROLLOUT_SLIDING_WINDOW = 32
ROLLOUT_LAG_MAX = 20

# -------------------------
# Plot
# -------------------------
PLOT_SHOW = False
PLOT_MAX_POINTS = 500
PLOT_SMOOTH_WINDOW = 20

# =========================================================
# Entry
# =========================================================

if __name__ == "__main__":
<<<<<<< HEAD
    if MODE == "collect":
        run_collect(
            data_dir=DATA_DIR,
            num_collect_steps=NUM_COLLECT_STEPS,
            agent_path=AGENT_PATH,
            deterministic_action=DETERMINISTIC_ACTION,
            seed=SEED,
            raw_csv_path=RAW_CSV_PATH,
            norm_csv_path=NORM_CSV_PATH,
            norm_npz_path=NORM_NPZ_PATH,
        )

    elif MODE == "train":
        run_train(
            device=DEVICE,
            model_dir=MODEL_DIR,
            log_dir=LOG_DIR,
            norm_npz_path=NORM_NPZ_PATH,
            state_model_path=STATE_MODEL_PATH,
            reward_model_path=REWARD_MODEL_PATH,
            train_history_path=TRAIN_HISTORY_PATH,
            seq_len=SEQ_LEN,
            stride=STRIDE,
            train_ratio=TRAIN_RATIO,
            batch_size=BATCH_SIZE,
            shuffle=SHUFFLE,
            state_hidden_dim=STATE_HIDDEN_DIM,
            state_num_layers=STATE_NUM_LAYERS,
            state_dropout=STATE_DROPOUT,
            state_use_action=STATE_USE_ACTION,
            reward_hidden_dim=REWARD_HIDDEN_DIM,
            reward_num_layers=REWARD_NUM_LAYERS,
            reward_mlp_hidden_dims=REWARD_MLP_HIDDEN_DIMS,
            epochs=EPOCHS,
            lr=LR,
            grad_clip_norm=GRAD_CLIP_NORM,
        )
=======
    # train_dqn(
    #     env_entrypoint=NetworkSlicingGym,
    #     timesteps=1_000,
    #     save_path="models/DQN-Network-Slicing",
    # )

    # train_dqn(
    #     env_entrypoint=NetworkSlicingGymDiscrete,
    #     timesteps=10_000,
    #     save_path="models/DQN-Network-Slicing",
    # )
>>>>>>> c967e10 (network simulator v2)

    elif MODE == "rollout":
        run_rollout(
            device=DEVICE,
            data_dir=DATA_DIR,
            log_dir=LOG_DIR,
            norm_npz_path=NORM_NPZ_PATH,
            state_std_path=STATE_STD_PATH,
            reward_std_path=REWARD_STD_PATH,
            state_model_path=STATE_MODEL_PATH,
            reward_model_path=REWARD_MODEL_PATH,
            rollout_t1_path=ROLLOUT_T1_PATH,
            rollout_tn_path=ROLLOUT_TN_PATH,
            seq_len=SEQ_LEN,
            rollout_start_idx=ROLLOUT_START_IDX,
            rollout_num_chunks=ROLLOUT_NUM_CHUNKS,
            rollout_horizon=ROLLOUT_HORIZON,
            rollout_sliding_window=ROLLOUT_SLIDING_WINDOW,
            rollout_lag_max=ROLLOUT_LAG_MAX,
            state_hidden_dim=STATE_HIDDEN_DIM,
            state_num_layers=STATE_NUM_LAYERS,
            state_dropout=STATE_DROPOUT,
            state_use_action=STATE_USE_ACTION,
            reward_hidden_dim=REWARD_HIDDEN_DIM,
            reward_num_layers=REWARD_NUM_LAYERS,
            reward_mlp_hidden_dims=REWARD_MLP_HIDDEN_DIMS,
        )

    elif MODE == "plot":
        run_plot(
            plot_dir=PLOT_DIR,
            train_history_path=TRAIN_HISTORY_PATH,
            rollout_t1_path=ROLLOUT_T1_PATH,
            rollout_tn_path=ROLLOUT_TN_PATH,
            plot_show=PLOT_SHOW,
            plot_max_points=PLOT_MAX_POINTS,
            plot_smooth_window=PLOT_SMOOTH_WINDOW,
        )

    elif MODE == "all":
        run_collect(
            data_dir=DATA_DIR,
            num_collect_steps=NUM_COLLECT_STEPS,
            agent_path=AGENT_PATH,
            deterministic_action=DETERMINISTIC_ACTION,
            seed=SEED,
            raw_csv_path=RAW_CSV_PATH,
            norm_csv_path=NORM_CSV_PATH,
            norm_npz_path=NORM_NPZ_PATH,
        )

        run_train(
            device=DEVICE,
            model_dir=MODEL_DIR,
            log_dir=LOG_DIR,
            norm_npz_path=NORM_NPZ_PATH,
            state_model_path=STATE_MODEL_PATH,
            reward_model_path=REWARD_MODEL_PATH,
            train_history_path=TRAIN_HISTORY_PATH,
            seq_len=SEQ_LEN,
            stride=STRIDE,
            train_ratio=TRAIN_RATIO,
            batch_size=BATCH_SIZE,
            shuffle=SHUFFLE,
            state_hidden_dim=STATE_HIDDEN_DIM,
            state_num_layers=STATE_NUM_LAYERS,
            state_dropout=STATE_DROPOUT,
            state_use_action=STATE_USE_ACTION,
            reward_hidden_dim=REWARD_HIDDEN_DIM,
            reward_num_layers=REWARD_NUM_LAYERS,
            reward_mlp_hidden_dims=REWARD_MLP_HIDDEN_DIMS,
            epochs=EPOCHS,
            lr=LR,
            grad_clip_norm=GRAD_CLIP_NORM,
        )

        run_rollout(
            device=DEVICE,
            data_dir=DATA_DIR,
            log_dir=LOG_DIR,
            norm_npz_path=NORM_NPZ_PATH,
            state_std_path=STATE_STD_PATH,
            reward_std_path=REWARD_STD_PATH,
            state_model_path=STATE_MODEL_PATH,
            reward_model_path=REWARD_MODEL_PATH,
            rollout_t1_path=ROLLOUT_T1_PATH,
            rollout_tn_path=ROLLOUT_TN_PATH,
            seq_len=SEQ_LEN,
            rollout_start_idx=ROLLOUT_START_IDX,
            rollout_num_chunks=ROLLOUT_NUM_CHUNKS,
            rollout_horizon=ROLLOUT_HORIZON,
            rollout_sliding_window=ROLLOUT_SLIDING_WINDOW,
            rollout_lag_max=ROLLOUT_LAG_MAX,
            state_hidden_dim=STATE_HIDDEN_DIM,
            state_num_layers=STATE_NUM_LAYERS,
            state_dropout=STATE_DROPOUT,
            state_use_action=STATE_USE_ACTION,
            reward_hidden_dim=REWARD_HIDDEN_DIM,
            reward_num_layers=REWARD_NUM_LAYERS,
            reward_mlp_hidden_dims=REWARD_MLP_HIDDEN_DIMS,
        )

        run_plot(
            plot_dir=PLOT_DIR,
            train_history_path=TRAIN_HISTORY_PATH,
            rollout_t1_path=ROLLOUT_T1_PATH,
            rollout_tn_path=ROLLOUT_TN_PATH,
            plot_show=PLOT_SHOW,
            plot_max_points=PLOT_MAX_POINTS,
            plot_smooth_window=PLOT_SMOOTH_WINDOW,
        )

    else:
        raise ValueError(
            f"Unsupported MODE={MODE}. "
            f"Use one of: 'collect', 'train', 'rollout', 'plot', 'all'."
        )
