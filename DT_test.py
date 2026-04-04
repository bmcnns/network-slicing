from stable_baselines3 import PPO

from digital_twin.env import DigitalTwinEnv

env = DigitalTwinEnv()

# env.set_config({
#     "device": "cuda",
#     "artifact_root": "artifacts/versions",
#
#     "seq_len": 64,
#     "state_dim": 3,
#     "action_dim": 3,
#     "state_use_action": True,
#     "normalized_input": True,
#
#     "num_collect_steps": 50000,
#     "agent_path": "Simulator/models/PPO-Network-Slicing",
#     "deterministic_action": True,
#     "seed": 0,
#
#     "stride": 32,
#     "train_ratio": 0.8,
#     "batch_size": 16,
#     "shuffle": False,
#
#     "state_hidden_dim": 64,
#     "state_num_layers": 1,
#     "state_dropout": 0.0,
#
#     "reward_hidden_dim": 64,
#     "reward_num_layers": 1,
#     "reward_mlp_hidden_dims": (128, 64),
#
#     "epochs": 1000,
#     "lr": 1e-3,
#     "grad_clip_norm": 1.0,
#
#     "rollout_start_idx": 0,
#     "rollout_num_chunks": 20,
#     "rollout_horizon": 200,
#     "rollout_sliding_window": 32,
#     "rollout_lag_max": 20,
#
#     "plot_show": False,
#     "plot_max_points": 500,
#     "plot_smooth_window": 20,
#
#     # if not rebuild dataset, use old version paths
#     "source_norm_npz_path": "artifacts/some_existing_version/data/transitions_norm.npz",
#     "source_state_std_path": "artifacts/some_existing_version/data/state_standardizer.pkl",
#     "source_action_std_path": "artifacts/some_existing_version/data/action_standardizer.pkl",
#     "source_reward_std_path": "artifacts/some_existing_version/data/reward_standardizer.pkl",
# })
#
# result = env.train(
#     rebuild=True,
#     run_collect=True,
#     run_rollout=True,
#     run_plot=True,
# )
#

agent = PPO.load('Simulator/models/PPO-Network-Slicing.zip', device="cpu")
env.load_version("20260403_113518")
obs, info = env.reset(options={"reset_mode": "full_zero"})
for _ in range(20):
    #action = env.action_space.sample()
    action, _ = agent.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    print(obs, reward, action)


