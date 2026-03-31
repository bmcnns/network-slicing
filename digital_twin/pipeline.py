from __future__ import annotations

from pprint import pprint

import torch

from digital_twin.dataset import (
    collect_transitions,
    fit_standardizers,
    normalize_transitions,
    save_all,
)
from digital_twin.standardizer import Standardizer
from digital_twin.models import (
    SequenceStatePredictorLSTM,
    SequenceRewardPredictorDNN,
)
from digital_twin.dataloader import build_sequence_dataloaders
from digital_twin.trainer import train_sequence_models
from digital_twin.rollout import (
    load_npz_transitions,
    save_rollout_result,
    rollout_t1_sequence,
    rollout_tn_sequence,
)
from digital_twin.plotting import (
    plot_sequence_training_loss,
    plot_all_rollout,
)
from digital_twin.utilities import ensure_dirs, save_json


def build_sequence_models(
    *,
    state_dim: int,
    action_dim: int,
    state_hidden_dim: int,
    state_num_layers: int,
    state_dropout: float,
    state_use_action: bool,
    reward_hidden_dim: int,
    reward_num_layers: int,
    reward_mlp_hidden_dims,
):
    state_model = SequenceStatePredictorLSTM(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=state_hidden_dim,
        num_layers=state_num_layers,
        dropout=state_dropout,
        use_action=state_use_action,
    )

    reward_model = SequenceRewardPredictorDNN(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=reward_hidden_dim,
        num_layers=reward_num_layers,
        mlp_hidden_dims=reward_mlp_hidden_dims,
    )

    return state_model, reward_model


def run_collect(
    *,
    data_dir: str,
    num_collect_steps: int,
    agent_path: str,
    deterministic_action: bool,
    seed: int,
    raw_csv_path: str,
    norm_csv_path: str,
    norm_npz_path: str,
) -> None:
    ensure_dirs(data_dir)

    raw_data = collect_transitions(
        num_steps=num_collect_steps,
        agent=agent_path,
        deterministic=deterministic_action,
        seed=seed,
    )

    standardizers = fit_standardizers(raw_data)
    norm_data = normalize_transitions(raw_data, standardizers)

    save_all(
        raw_data=raw_data,
        norm_data=norm_data,
        standardizers=standardizers,
        save_dir=data_dir,
    )

    print("Collection finished.")
    print(f"Saved raw csv:  {raw_csv_path}")
    print(f"Saved norm csv: {norm_csv_path}")
    print(f"Saved norm npz: {norm_npz_path}")


def run_train(
    *,
    device: str,
    model_dir: str,
    log_dir: str,
    norm_npz_path: str,
    state_model_path: str,
    reward_model_path: str,
    train_history_path: str,
    seq_len: int,
    stride: int,
    train_ratio: float,
    batch_size: int,
    shuffle: bool,
    state_hidden_dim: int,
    state_num_layers: int,
    state_dropout: float,
    state_use_action: bool,
    reward_hidden_dim: int,
    reward_num_layers: int,
    reward_mlp_hidden_dims,
    epochs: int,
    lr: float,
    grad_clip_norm: float,
) -> None:
    ensure_dirs(model_dir, log_dir)

    train_loader, val_loader, dims = build_sequence_dataloaders(
        npz_path=norm_npz_path,
        seq_len=seq_len,
        stride=stride,
        train_ratio=train_ratio,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    state_dim = dims["state_dim"]
    action_dim = dims["action_dim"]

    state_model, reward_model = build_sequence_models(
        state_dim=state_dim,
        action_dim=action_dim,
        state_hidden_dim=state_hidden_dim,
        state_num_layers=state_num_layers,
        state_dropout=state_dropout,
        state_use_action=state_use_action,
        reward_hidden_dim=reward_hidden_dim,
        reward_num_layers=reward_num_layers,
        reward_mlp_hidden_dims=reward_mlp_hidden_dims,
    )

    history = train_sequence_models(
        state_model=state_model,
        reward_model=reward_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        lr=lr,
        grad_clip_norm=grad_clip_norm,
    )

    torch.save(state_model.state_dict(), state_model_path)
    torch.save(reward_model.state_dict(), reward_model_path)
    save_json(history, train_history_path)

    print("Sequence training finished.")
    print(f"Saved state model:  {state_model_path}")
    print(f"Saved reward model: {reward_model_path}")
    print(f"Saved history:      {train_history_path}")


def run_rollout(
    *,
    device: str,
    data_dir: str,
    log_dir: str,
    norm_npz_path: str,
    state_std_path: str,
    reward_std_path: str,
    state_model_path: str,
    reward_model_path: str,
    rollout_t1_path: str,
    rollout_tn_path: str,
    seq_len: int,
    rollout_start_idx: int,
    rollout_num_chunks: int,
    rollout_horizon: int,
    rollout_sliding_window: int,
    rollout_lag_max: int,
    state_hidden_dim: int,
    state_num_layers: int,
    state_dropout: float,
    state_use_action: bool,
    reward_hidden_dim: int,
    reward_num_layers: int,
    reward_mlp_hidden_dims,
) -> None:
    ensure_dirs(data_dir, log_dir)

    transitions = load_npz_transitions(norm_npz_path)
    state_standardizer = Standardizer.load(state_std_path)
    reward_standardizer = Standardizer.load(reward_std_path)

    state_dim = int(transitions["states"].shape[1])
    action_dim = int(transitions["actions"].shape[1])

    state_model, reward_model = build_sequence_models(
        state_dim=state_dim,
        action_dim=action_dim,
        state_hidden_dim=state_hidden_dim,
        state_num_layers=state_num_layers,
        state_dropout=state_dropout,
        state_use_action=state_use_action,
        reward_hidden_dim=reward_hidden_dim,
        reward_num_layers=reward_num_layers,
        reward_mlp_hidden_dims=reward_mlp_hidden_dims,
    )

    state_model.load_state_dict(torch.load(state_model_path, map_location=device))
    reward_model.load_state_dict(torch.load(reward_model_path, map_location=device))

    state_model.to(device).eval()
    reward_model.to(device).eval()

    result_t1 = rollout_t1_sequence(
        state_model=state_model,
        reward_model=reward_model,
        transitions=transitions,
        seq_len=seq_len,
        device=device,
        state_standardizer=state_standardizer,
        reward_standardizer=reward_standardizer,
        sliding_window=rollout_sliding_window,
        lag_max=rollout_lag_max,
        start_idx=rollout_start_idx,
        num_chunks=rollout_num_chunks,
        normalized_input=True,
        state_use_action=state_use_action,
    )
    save_rollout_result(result_t1, rollout_t1_path)

    result_tn = rollout_tn_sequence(
        state_model=state_model,
        reward_model=reward_model,
        transitions=transitions,
        seq_len=seq_len,
        horizon=rollout_horizon,
        sliding_window=rollout_sliding_window,
        lag_max=rollout_lag_max,
        device=device,
        state_standardizer=state_standardizer,
        reward_standardizer=reward_standardizer,
        start_idx=rollout_start_idx,
        normalized_input=True,
        state_use_action=state_use_action,
    )
    save_rollout_result(result_tn, rollout_tn_path)

    print("Sequence rollout finished.")
    print(f"Saved T+1 rollout: {rollout_t1_path}")
    print(f"Saved T+N rollout: {rollout_tn_path}")


def run_plot(
    *,
    plot_dir: str,
    train_history_path: str,
    rollout_t1_path: str,
    rollout_tn_path: str,
    plot_show: bool,
    plot_max_points: int,
    plot_smooth_window: int,
) -> None:
    ensure_dirs(plot_dir)

    train_plot_paths = plot_sequence_training_loss(
        history_path=train_history_path,
        save_dir=plot_dir,
        show=plot_show,
    )

    t1_plot_paths = plot_all_rollout(
        rollout_npz_path=rollout_t1_path,
        save_dir=plot_dir,
        tag="sequence_t1",
        show=plot_show,
        max_points=plot_max_points,
        smooth_window=plot_smooth_window,
    )

    tn_plot_paths = plot_all_rollout(
        rollout_npz_path=rollout_tn_path,
        save_dir=plot_dir,
        tag="sequence_tn",
        show=plot_show,
        max_points=plot_max_points,
        smooth_window=plot_smooth_window,
    )

    print("Sequence plot finished.")
    print("Training plots:")
    pprint(train_plot_paths)
    print("T+1 plots:")
    pprint(t1_plot_paths)
    print("T+N plots:")
    pprint(tn_plot_paths)