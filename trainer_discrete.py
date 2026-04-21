from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from typing import Any, Dict

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed

from simulator.gym_env import NetworkSlicingGymStarsBars


DEFAULT_PPO_KWARGS: Dict[str, Any] = {
    "learning_rate": 3e-4,
    "n_steps": 128,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}


def resolve_device(device: str) -> str:
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested device={device}, but CUDA is unavailable; falling back to CPU.")
        return "cpu"
    return device


class RoundCurveLogger:
    def __init__(self, output_path: str, round_length: int):
        self.output_path = output_path
        self.round_length = round_length
        self.global_steps = 0
        self.round_steps = 0
        self.reward_sum = 0.0
        self.utility_sum = 0.0
        self.handle = open(self.output_path, "w", encoding="utf-8")
        self.handle.write("round reward utility\n")

    def log_step(self, reward: float, utility: float) -> None:
        self.global_steps += 1
        self.round_steps += 1
        self.reward_sum += reward
        self.utility_sum += utility

        if self.round_steps >= self.round_length:
            self._flush_round()

    def finalize(self) -> None:
        if self.round_steps > 0:
            self._flush_round()
        self.handle.close()

    def _flush_round(self) -> None:
        round_idx = self.global_steps // self.round_length
        mean_reward = self.reward_sum / self.round_steps
        mean_utility = self.utility_sum / self.round_steps
        self.handle.write(f"{round_idx} {mean_reward:.6f} {mean_utility:.6f}\n")
        self.handle.flush()
        self.round_steps = 0
        self.reward_sum = 0.0
        self.utility_sum = 0.0


class StepTraceLogger:
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.handle = open(self.output_path, "w", encoding="utf-8")
        self.handle.write(
            "step action reward utility "
            "s0 s1 s2 "
            "b0 b1 b2 "
            "ns0 ns1 ns2\n"
        )

    @staticmethod
    def _triplet(value) -> np.ndarray:
        if value is None:
            return np.array([np.nan, np.nan, np.nan], dtype=np.float32)

        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size < 3:
            padded = np.full(3, np.nan, dtype=np.float32)
            padded[:arr.size] = arr
            return padded
        return arr[:3]

    def log_step(self, *, global_step: int, action: int, reward: float, info: Dict[str, Any]) -> None:
        state = self._triplet(info.get("current_state"))
        next_state = self._triplet(info.get("next_state"))
        bandwidth = self._triplet(info.get("applied_action_mhz"))
        utility = float(info.get("utility", 0.0))

        self.handle.write(
            f"{global_step} {action} {reward:.6f} {utility:.6f} "
            f"{state[0]:.6f} {state[1]:.6f} {state[2]:.6f} "
            f"{bandwidth[0]:.6f} {bandwidth[1]:.6f} {bandwidth[2]:.6f} "
            f"{next_state[0]:.6f} {next_state[1]:.6f} {next_state[2]:.6f}\n"
        )

        if global_step % 100 == 0:
            self.handle.flush()

    def close(self) -> None:
        self.handle.close()


class DiscreteTrainingCallback(BaseCallback):
    def __init__(self, curve_logger: RoundCurveLogger, step_logger: StepTraceLogger, verbose: int = 0):
        super().__init__(verbose)
        self.curve_logger = curve_logger
        self.step_logger = step_logger

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        rewards = self.locals.get("rewards", [0.0])
        actions = self.locals.get("actions", [0])

        info = infos[0] if infos else {}
        reward = float(rewards[0]) if len(rewards) > 0 else 0.0
        action = int(np.asarray(actions).reshape(-1)[0])

        self.curve_logger.log_step(
            reward=reward,
            utility=float(info.get("utility", 0.0)),
        )
        self.step_logger.log_step(
            global_step=self.num_timesteps,
            action=action,
            reward=reward,
            info=info,
        )
        return True


def evaluate_policy_distribution(
    model: PPO,
    *,
    seed: int,
    eval_steps: int,
    output_path: str,
    deterministic: bool,
) -> Dict[str, Any]:
    env = NetworkSlicingGymStarsBars()
    obs, _ = env.reset(seed=seed)
    counts = np.zeros(env.action_space.n, dtype=np.int64)

    for _ in range(eval_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        action_idx = int(action)
        counts[action_idx] += 1
        obs, _, terminated, truncated, _ = env.step(action_idx)
        if terminated or truncated:
            obs, _ = env.reset(seed=seed)

    lookup = env.action_space_lookup.actions
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("action count fraction u0 u1 u2 mhz0 mhz1 mhz2\n")
        for action_idx in range(env.action_space.n):
            units = lookup[action_idx]
            mhz = units * 0.2
            count = int(counts[action_idx])
            fraction = count / float(eval_steps) if eval_steps > 0 else 0.0
            handle.write(
                f"{action_idx} {count} {fraction:.8f} "
                f"{units[0]} {units[1]} {units[2]} "
                f"{mhz[0]:.1f} {mhz[1]:.1f} {mhz[2]:.1f}\n"
            )

    top_idx = np.argsort(counts)[::-1][:10]
    top_actions = [
        {
            "action": int(idx),
            "count": int(counts[idx]),
            "fraction": float(counts[idx] / float(eval_steps)) if eval_steps > 0 else 0.0,
            "units": lookup[idx].tolist(),
            "mhz": (lookup[idx] * 0.2).tolist(),
        }
        for idx in top_idx
        if counts[idx] > 0
    ]
    return {
        "eval_steps": eval_steps,
        "deterministic": deterministic,
        "top_actions": top_actions,
    }


def train_discrete(args: argparse.Namespace) -> Dict[str, Any]:
    args.device = resolve_device(args.device)

    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    run_stem = f"{run_id}_seed{args.seed}_discrete4851"
    run_dir = os.path.join(args.artifact_root, run_id)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)

    learning_curve_path = os.path.join(args.results_dir, f"{run_stem}.dat")
    step_trace_path = os.path.join(args.results_dir, f"{run_stem}_steps.dat")
    action_hist_path = os.path.join(args.results_dir, f"{run_stem}_action_hist.dat")
    model_path = os.path.join(run_dir, f"{run_stem}.zip")
    summary_path = os.path.join(run_dir, f"{run_stem}_summary.json")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_random_seed(args.seed)

    env = NetworkSlicingGymStarsBars()
    callback = DiscreteTrainingCallback(
        curve_logger=RoundCurveLogger(learning_curve_path, args.round_length),
        step_logger=StepTraceLogger(step_trace_path),
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        device=args.device,
        tensorboard_log=os.path.join(run_dir, "tensorboard"),
        **DEFAULT_PPO_KWARGS,
    )

    print("=" * 60)
    print("Training discrete PPO on 4851-action stars-and-bars space")
    print(f"Seed: {args.seed}")
    print(f"Timesteps: {args.timesteps}")
    print("=" * 60)

    model.learn(
        total_timesteps=args.timesteps,
        callback=callback,
        progress_bar=False,
        tb_log_name="ppo_discrete_4851",
    )
    callback.curve_logger.finalize()
    callback.step_logger.close()

    model.save(model_path)
    eval_summary = evaluate_policy_distribution(
        model,
        seed=args.eval_seed,
        eval_steps=args.eval_steps,
        output_path=action_hist_path,
        deterministic=args.eval_deterministic,
    )

    summary = {
        "run_id": run_id,
        "run_dir": run_dir,
        "run_stem": run_stem,
        "seed": args.seed,
        "device": args.device,
        "timesteps": args.timesteps,
        "round_length": args.round_length,
        "learning_curve_path": learning_curve_path,
        "step_trace_path": step_trace_path,
        "action_hist_path": action_hist_path,
        "model_path": model_path,
        "eval_seed": args.eval_seed,
        "eval_steps": args.eval_steps,
        "eval_deterministic": args.eval_deterministic,
        "top_actions": eval_summary["top_actions"],
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("=" * 60)
    print("Discrete training complete")
    print(f"Results: {args.results_dir}")
    print(f"Artifacts: {run_dir}")
    print(f"Learning curve: {learning_curve_path}")
    print(f"Action histogram: {action_hist_path}")
    print(f"Model: {model_path}")
    print("=" * 60)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on the 4851-action stars-and-bars discrete env.")
    parser.add_argument("--artifact-root", default="artifacts/discrete_runs")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--round-length", type=int, default=400)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-seed", type=int, default=12345)
    parser.add_argument("--eval-steps", type=int, default=10_000)
    parser.add_argument("--eval-deterministic", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train_discrete(parse_args())
