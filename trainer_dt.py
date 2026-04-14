from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from digital_twin.env import DigitalTwinEnv
from digital_twin.utilities import ensure_dirs, save_json
from simulator.gym_env import NetworkSlicingGymContinuous


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

DEFAULT_ROUND_LENGTH = 400
DEFAULT_TOTAL_ROUNDS = 30_000
DEFAULT_BOOTSTRAP_ITERATIONS = 100
DEFAULT_TWIN_ROUNDS_PER_ITER = 150
DEFAULT_REAL_ROUNDS_PER_ITER = 150
DEFAULT_TWIN_TRAIN_STEPS = DEFAULT_TWIN_ROUNDS_PER_ITER * DEFAULT_ROUND_LENGTH
DEFAULT_REAL_TRAIN_STEPS = DEFAULT_REAL_ROUNDS_PER_ITER * DEFAULT_ROUND_LENGTH

CONFIG_SHARDS: Dict[str, Dict[str, int]] = {
    "dt0_real100": {
        "bootstrap_iterations": 100,
        "twin_train_steps": 0 * DEFAULT_ROUND_LENGTH,
        "real_train_steps": 300 * DEFAULT_ROUND_LENGTH,
    },
    "dt10_real90": {
        "bootstrap_iterations": 100,
        "twin_train_steps": 30 * DEFAULT_ROUND_LENGTH,
        "real_train_steps": 270 * DEFAULT_ROUND_LENGTH,
    },
    "dt25_real75": {
        "bootstrap_iterations": 100,
        "twin_train_steps": 75 * DEFAULT_ROUND_LENGTH,
        "real_train_steps": 225 * DEFAULT_ROUND_LENGTH,
    },
    "dt50_real50": {
        "bootstrap_iterations": 100,
        "twin_train_steps": 150 * DEFAULT_ROUND_LENGTH,
        "real_train_steps": 150 * DEFAULT_ROUND_LENGTH,
    },
    "dt75_real25": {
        "bootstrap_iterations": 100,
        "twin_train_steps": 225 * DEFAULT_ROUND_LENGTH,
        "real_train_steps": 75 * DEFAULT_ROUND_LENGTH,
    },
    "dt90_real10": {
        "bootstrap_iterations": 100,
        "twin_train_steps": 270 * DEFAULT_ROUND_LENGTH,
        "real_train_steps": 30 * DEFAULT_ROUND_LENGTH,
    },
    "dt100_real0": {
        "bootstrap_iterations": 100,
        "twin_train_steps": 300 * DEFAULT_ROUND_LENGTH,
        "real_train_steps": 0 * DEFAULT_ROUND_LENGTH,
    },
    # Backward-compatible aliases.
    "config1": {
        "bootstrap_iterations": 100,
        "twin_train_steps": 225 * DEFAULT_ROUND_LENGTH,
        "real_train_steps": 75 * DEFAULT_ROUND_LENGTH,
    },
    "config2": {
        "bootstrap_iterations": 100,
        "twin_train_steps": 150 * DEFAULT_ROUND_LENGTH,
        "real_train_steps": 150 * DEFAULT_ROUND_LENGTH,
    },
    "config3": {
        "bootstrap_iterations": 100,
        "twin_train_steps": 0,
        "real_train_steps": 300 * DEFAULT_ROUND_LENGTH,
    },
}

CONFIG_LABELS: Dict[str, str] = {
    "dt0_real100": "dt0_real100",
    "dt10_real90": "dt10_real90",
    "dt25_real75": "dt25_real75",
    "dt50_real50": "dt50_real50",
    "dt75_real25": "dt75_real25",
    "dt90_real10": "dt90_real10",
    "dt100_real0": "dt100_real0",
    "config1": "heavy",
    "config2": "balanced",
    "config3": "none",
}


class PhaseStatsCallback(BaseCallback):
    def __init__(self, curve_logger: "RoundCurveLogger", verbose: int = 0):
        super().__init__(verbose)
        self.curve_logger = curve_logger

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        rewards = self.locals.get("rewards", [0.0])

        info = infos[0] if infos else {}
        reward = float(rewards[0]) if len(rewards) > 0 else 0.0

        self.curve_logger.log_step(
            reward=reward,
            utility=float(info.get("utility", 0.0)),
        )
        return True


class RoundCurveLogger:
    def __init__(self, output_path: str, round_length: int = 400):
        self.output_path = output_path
        self.round_length = round_length
        self.global_steps = 0
        self.round_reward_sum = 0.0
        self.round_utility_sum = 0.0
        self.round_steps = 0

        with open(self.output_path, "w", encoding="utf-8") as handle:
            handle.write("round reward utility\n")

    def log_step(self, reward: float, utility: float) -> None:
        self.global_steps += 1
        self.round_steps += 1
        self.round_reward_sum += reward
        self.round_utility_sum += utility

        if self.round_steps >= self.round_length:
            self._flush_round()

    def finalize(self) -> None:
        if self.round_steps > 0:
            self._flush_round()

    def _flush_round(self) -> None:
        round_idx = self.global_steps // self.round_length
        mean_reward = self.round_reward_sum / self.round_steps
        mean_utility = self.round_utility_sum / self.round_steps

        with open(self.output_path, "a", encoding="utf-8") as handle:
            handle.write(f"{round_idx} {mean_reward:.6f} {mean_utility:.6f}\n")

        self.round_reward_sum = 0.0
        self.round_utility_sum = 0.0
        self.round_steps = 0


class NullCallback(BaseCallback):
    def _on_step(self) -> bool:
        return True


def build_twin_config(args: argparse.Namespace, run_dir: str) -> Dict[str, Any]:
    return {
        "device": args.twin_device,
        "artifact_root": os.path.join(run_dir, "twin_versions"),
        "seq_len": args.seq_len,
        "state_dim": 3,
        "action_dim": 3,
        "state_use_action": True,
        "normalized_input": True,
        "num_collect_steps": args.collect_steps,
        "agent_path": None,
        "deterministic_action": True,
        "seed": args.seed,
        "stride": args.stride,
        "train_ratio": args.train_ratio,
        "batch_size": args.twin_batch_size,
        "shuffle": False,
        "state_hidden_dim": args.state_hidden_dim,
        "state_num_layers": args.state_num_layers,
        "state_dropout": args.state_dropout,
        "reward_hidden_dim": args.reward_hidden_dim,
        "reward_num_layers": args.reward_num_layers,
        "reward_mlp_hidden_dims": tuple(args.reward_mlp_hidden_dims),
        "epochs": args.twin_epochs,
        "lr": args.twin_lr,
        "grad_clip_norm": args.grad_clip_norm,
        "rollout_start_idx": 0,
        "rollout_num_chunks": args.rollout_num_chunks,
        "rollout_horizon": args.rollout_horizon,
        "rollout_sliding_window": args.rollout_sliding_window,
        "rollout_lag_max": args.rollout_lag_max,
        "plot_show": False,
        "plot_max_points": args.plot_max_points,
        "plot_smooth_window": args.plot_smooth_window,
    }


def rebuild_twin(
    twin_env: DigitalTwinEnv,
    *,
    policy_path: str | None,
    seed: int,
    run_rollout: bool,
    run_plot: bool,
) -> Dict[str, Any]:
    twin_env.set_config(
        {
            "agent_path": policy_path,
            "seed": seed,
        }
    )

    result = twin_env.train(
        rebuild=True,
        run_collect=True,
        run_rollout=run_rollout,
        run_plot=run_plot,
        sim_agent=policy_path,
    )
    return result


def copy_policy(source: PPO, target: PPO) -> None:
    target.policy.load_state_dict(source.policy.state_dict())


def polyak_mix_policy(agent1: PPO, agent2: PPO, zeta: float) -> None:
    mixed_state = {}
    state1 = agent1.policy.state_dict()
    state2 = agent2.policy.state_dict()

    with torch.no_grad():
        for key in state1:
            mixed_state[key] = zeta * state1[key] + (1.0 - zeta) * state2[key]

    agent1.policy.load_state_dict(mixed_state)


def apply_config_shard(args: argparse.Namespace) -> argparse.Namespace:
    shard = CONFIG_SHARDS[args.config_shard]
    args.bootstrap_iterations = shard["bootstrap_iterations"]
    args.twin_train_steps = shard["twin_train_steps"]
    args.real_train_steps = shard["real_train_steps"]
    return args


def train_bootstrap(args: argparse.Namespace) -> Dict[str, Any]:
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    config_label = CONFIG_LABELS[args.config_shard]
    run_stem = f"{run_id}_{config_label}"
    run_dir = os.path.join(args.artifact_root, run_id)
    ensure_dirs(args.artifact_root)
    ensure_dirs(args.results_dir)
    ensure_dirs(run_dir)

    learning_curve_path = os.path.join(args.results_dir, f"{run_stem}.dat")
    summary_path = os.path.join(run_dir, f"{run_stem}_summary.json")
    final_agent1_path = os.path.join(run_dir, f"{run_stem}_agent1_final.zip")
    final_agent2_path = os.path.join(run_dir, f"{run_stem}_agent2_final.zip")
    curve_logger = RoundCurveLogger(
        output_path=learning_curve_path,
        round_length=args.round_length,
    )

    use_twin = args.twin_train_steps > 0
    twin_env = None
    initial_twin = None
    if use_twin:
        twin_env = DigitalTwinEnv()
        twin_env.set_config(build_twin_config(args, run_dir))

    bootstrap_history: List[Dict[str, Any]] = []
    if use_twin:
        print("=" * 60)
        print("Bootstrap Stage 0: random collection -> initial twin training")
        print("=" * 60)
        initial_twin = rebuild_twin(
            twin_env,
            policy_path=None,
            seed=args.seed,
            run_rollout=args.run_rollout,
            run_plot=args.run_plot,
        )
        bootstrap_history.append(
            {
                "iteration": 0,
                "phase": "initial_twin_build",
                "twin_version": initial_twin["version_id"],
                "paths": initial_twin["paths"],
            }
        )

    real_env = NetworkSlicingGymContinuous()
    agent1 = PPO(
        "MlpPolicy",
        real_env,
        verbose=1,
        tensorboard_log=os.path.join(run_dir, "tensorboard"),
        device=args.ppo_device,
        **DEFAULT_PPO_KWARGS,
    )
    agent2 = None
    if use_twin:
        agent2 = PPO(
            "MlpPolicy",
            twin_env,
            verbose=1,
            tensorboard_log=os.path.join(run_dir, "tensorboard"),
            device=args.ppo_device,
            **DEFAULT_PPO_KWARGS,
        )
        copy_policy(agent1, agent2)

    for iteration in range(1, args.bootstrap_iterations + 1):
        print("=" * 60)
        print(f"Bootstrap Iteration {iteration}/{args.bootstrap_iterations}")
        print("=" * 60)

        agent1_collect_path = os.path.join(run_dir, f"{run_stem}_agent1_collect_iter_{iteration}.zip")
        agent1.save(agent1_collect_path)

        twin_result = None
        if use_twin:
            print(f"[Iteration {iteration}] Collecting fresh simulator data with Agent 1 and rebuilding twin")
            twin_result = rebuild_twin(
                twin_env,
                policy_path=agent1_collect_path,
                seed=args.seed + iteration,
                run_rollout=args.run_rollout,
                run_plot=args.run_plot,
            )

            agent2.set_env(twin_env)
            print(f"[Iteration {iteration}] Training Agent 2 on digital twin for {args.twin_train_steps} steps")
            agent2.learn(
                total_timesteps=args.twin_train_steps,
                reset_num_timesteps=False,
                callback=NullCallback(),
                progress_bar=False,
                tb_log_name="agent2_twin",
            )

            print(f"[Iteration {iteration}] Mixing Agent 2 into Agent 1 with zeta={args.zeta}")
            polyak_mix_policy(agent1, agent2, args.zeta)

        if args.real_train_steps > 0:
            agent1.set_env(real_env)
            real_callback = PhaseStatsCallback(curve_logger=curve_logger)
            print(f"[Iteration {iteration}] Fine-tuning Agent 1 on simulator for {args.real_train_steps} steps")
            agent1.learn(
                total_timesteps=args.real_train_steps,
                reset_num_timesteps=False,
                callback=real_callback,
                progress_bar=False,
                tb_log_name="agent1_real",
            )
        else:
            print(f"[Iteration {iteration}] Skipping simulator fine-tuning (real_train_steps=0)")

        if use_twin:
            print(f"[Iteration {iteration}] Copying Agent 1 parameters back to Agent 2")
            copy_policy(agent1, agent2)

        agent1_iter_path = os.path.join(run_dir, f"{run_stem}_agent1_iter_{iteration}.zip")
        agent2_iter_path = os.path.join(run_dir, f"{run_stem}_agent2_iter_{iteration}.zip")
        agent1.save(agent1_iter_path)
        if use_twin:
            agent2.save(agent2_iter_path)

        record = {
            "iteration": iteration,
            "agent1_collect_path": agent1_collect_path,
            "agent1_policy_path": agent1_iter_path,
        }
        if use_twin:
            record["agent2_policy_path"] = agent2_iter_path
            record["twin_version"] = twin_result["version_id"]
            record["paths"] = twin_result["paths"]
        bootstrap_history.append(record)

    curve_logger.finalize()
    agent1.save(final_agent1_path)
    if use_twin:
        agent2.save(final_agent2_path)

    summary = {
        "run_id": run_id,
        "run_dir": run_dir,
        "run_stem": run_stem,
        "config_shard": args.config_shard,
        "config_label": config_label,
        "use_twin": use_twin,
        "learning_curve_path": learning_curve_path,
        "final_agent1_path": final_agent1_path,
        "final_agent2_path": final_agent2_path if use_twin else None,
        "bootstrap_iterations": args.bootstrap_iterations,
        "twin_train_steps": args.twin_train_steps,
        "real_train_steps": args.real_train_steps,
        "collect_steps": args.collect_steps,
        "round_length": args.round_length,
        "scheduled_total_rounds": (
            args.bootstrap_iterations
            * (args.twin_train_steps + args.real_train_steps)
            // args.round_length
        ),
        "zeta": args.zeta,
        "history": bootstrap_history,
    }
    save_json(summary, summary_path)

    print("=" * 60)
    print("Bootstrap training complete")
    print(f"Results directory: {args.results_dir}")
    print(f"Artifacts directory: {run_dir}")
    print(f"Learning curve: {learning_curve_path}")
    print(f"Final Agent 1: {final_agent1_path}")
    if use_twin:
        print(f"Final Agent 2: {final_agent2_path}")
    print("=" * 60)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap PPO with a digital twin.")
    parser.add_argument("--artifact-root", default="artifacts/bootstrap_runs")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--device", default=None)
    parser.add_argument("--ppo-device", default=None)
    parser.add_argument("--twin-device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config-shard", choices=sorted(CONFIG_SHARDS.keys()), default="config2")

    parser.add_argument("--bootstrap-iterations", type=int, default=DEFAULT_BOOTSTRAP_ITERATIONS)
    parser.add_argument("--collect-steps", type=int, default=5000)
    parser.add_argument("--twin-train-steps", type=int, default=DEFAULT_TWIN_TRAIN_STEPS)
    parser.add_argument("--real-train-steps", type=int, default=DEFAULT_REAL_TRAIN_STEPS)
    parser.add_argument("--round-length", type=int, default=DEFAULT_ROUND_LENGTH)
    parser.add_argument("--zeta", type=float, default=0.8)

    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--twin-batch-size", type=int, default=16)
    parser.add_argument("--twin-epochs", type=int, default=200)
    parser.add_argument("--twin-lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)

    parser.add_argument("--state-hidden-dim", type=int, default=64)
    parser.add_argument("--state-num-layers", type=int, default=1)
    parser.add_argument("--state-dropout", type=float, default=0.0)
    parser.add_argument("--reward-hidden-dim", type=int, default=64)
    parser.add_argument("--reward-num-layers", type=int, default=1)
    parser.add_argument("--reward-mlp-hidden-dims", nargs=2, type=int, default=[128, 64])

    parser.add_argument("--rollout-num-chunks", type=int, default=20)
    parser.add_argument("--rollout-horizon", type=int, default=200)
    parser.add_argument("--rollout-sliding-window", type=int, default=32)
    parser.add_argument("--rollout-lag-max", type=int, default=20)
    parser.add_argument("--plot-max-points", type=int, default=500)
    parser.add_argument("--plot-smooth-window", type=int, default=20)
    parser.add_argument("--run-rollout", action="store_true")
    parser.add_argument("--run-plot", action="store_true")

    args = parser.parse_args()
    if args.ppo_device is None and args.twin_device is None:
        if args.device is not None:
            args.ppo_device = args.device
            args.twin_device = args.device
        else:
            args.ppo_device = "cpu"
            args.twin_device = "cpu"
    else:
        if args.ppo_device is None:
            args.ppo_device = args.device if args.device is not None else "cpu"
        if args.twin_device is None:
            args.twin_device = args.device if args.device is not None else "cpu"

    return apply_config_shard(args)


if __name__ == "__main__":
    train_bootstrap(parse_args())
