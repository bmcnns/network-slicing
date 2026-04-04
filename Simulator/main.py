import gymnasium as gym
from stable_baselines3 import DQN, PPO
from gym_env import *


def train_dqn(
    env_entrypoint,
    timesteps=50_000,
    save_path="models/DQN-Network-Slicing",
    env_id="NetworkSlicing-DQN-v0",
):
    print("=" * 50)
    print("Training DQN...")
    print("=" * 50)

    # Register env (allow different entrypoints)
    gym.register(
        id=env_id,
        entry_point=env_entrypoint,
    )

    env = gym.make(env_id)
    callback = GnuplotLoggerCallback(filename="training_dqn.dat")

    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-3,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=128,
        gamma=0.99,
        train_freq=4,
        target_update_interval=500,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
    )

    model.learn(total_timesteps=timesteps, callback=callback)
    model.save(save_path)

    print(f"DQN model saved to {save_path}")
    print("=" * 50)


def train_ppo(
    env_entrypoint,
    timesteps=100_000,
    save_path="models/PPO-Network-Slicing",
    env_id="NetworkSlicing-PPO-v0",
):
    print("=" * 50)
    print("Training PPO...")
    print("=" * 50)

    # Register env (allow different entrypoints)
    gym.register(
        id=env_id,
        entry_point=env_entrypoint,
    )

    env = gym.make(env_id)
    callback = GnuplotLoggerCallback(filename="training_ppo.dat")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )

    model.learn(total_timesteps=timesteps, callback=callback)
    model.save(save_path)

    print(f"PPO model saved to {save_path}")
    print("=" * 50)


if __name__ == "__main__":
    # train_dqn(
    #     env_entrypoint=NetworkSlicingGym,
    #     timesteps=10_000,
    #     save_path="models/DQN-Network-Slicing",
    # )

    # train_dqn(
    #     env_entrypoint=NetworkSlicingGymDiscrete,
    #     timesteps=50_000,
    #     save_path="models/DQN-Network-Slicing",
    # )

    # train_ppo(
    #     env_entrypoint=NetworkSlicingGymContinuous,
    #     timesteps=10_000,
    #     save_path="models/PPO-Network-Slicing",
    # )

    env = NetworkSlicingGymContinuous()

    # load trained agent
    agent = PPO.load("models/PPO-Network-Slicing", device='cpu')

    obs, _ = env.reset()

    for t in range(20):

        action = env.action_space.sample()
        action, _ = agent.predict(obs, deterministic=True)

        next_obs, reward, term, trunc, info = env.step(action)

        print(next_obs, reward, action)

        obs = next_obs

        if term or trunc:
            obs, _ = env.reset()