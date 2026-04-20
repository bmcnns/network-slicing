import gymnasium as gym
from .environment import NetworkSlicingEnv
from .config import Config
import numpy as np
from .action import ActionSpace
from typing import Optional
from stable_baselines3.common.callbacks import BaseCallback


class GnuplotLoggerCallback(BaseCallback):
    """
    Callback for logging training metrics to file for visualization
    
    Records spectrum efficiency, utility, and reward at each training step.
    Compatible with gnuplot or matplotlib for plotting training curves.
    """
    
    def __init__(self, filename="training.dat", verbose=0):
        """
        Initialize logger callback
        
        Args:
            filename (str): Output file path for training logs
            verbose (int): Verbosity level (0 = silent, 1 = info)
        """
        super(GnuplotLoggerCallback, self).__init__(verbose)
        self.filename = filename
        self.file = None

    def _on_training_start(self) -> None:
        """Open log file and write header"""
        self.file = open(self.filename, "w")
        self.file.write("# Step Efficiency Utility Reward\n")
        self.file.flush()
        
    def _on_step(self) -> bool:
        """
        Log metrics at each training step
        
        Returns:
            bool: True to continue training, False to stop
        """
        infos = self.locals.get("infos", [{}])
        info = infos[0]
        reward = self.locals.get("rewards", [0])[0]

        se   = info.get("spectrum_efficiency", 0.0)
        util = info.get("utility", 0.0)
        ssr  = info.get("ssr", {s: 0.0 for s in ['VoLTE', 'Video', 'URLLC']})
        bw   = info.get("bandwidth_allocation", {s: 0.0 for s in ['VoLTE', 'Video', 'URLLC']})
        arr  = info.get("total_arrived_packets", {s: 0 for s in ['VoLTE', 'Video', 'URLLC']})
        suc  = info.get("total_successful_packets", {s: 0 for s in ['VoLTE', 'Video', 'URLLC']})
        drp  = info.get("dropped_packets", {s: 0 for s in ['VoLTE', 'Video', 'URLLC']})

        slices = ['VoLTE', 'Video', 'URLLC']

        line = (
            f"{self.num_timesteps} {se:.6f} {util:.6f} {reward:.6f} "
            f"{ssr['VoLTE']:.4f} {ssr['Video']:.4f} {ssr['URLLC']:.4f} "
            f"{bw['VoLTE']/1e6:.3f} {bw['Video']/1e6:.3f} {bw['URLLC']/1e6:.3f} "
            f"{arr['VoLTE']} {arr['Video']} {arr['URLLC']} "
            f"{suc['VoLTE']} {suc['Video']} {suc['URLLC']} "
            f"{drp['VoLTE']} {drp['Video']} {drp['URLLC']}\n"
        )
        self.file.write(line)

        if self.n_calls % 100 == 0:
            self.file.flush()

        return True

    def _on_training_end(self) -> None:
        """Close log file at end of training"""
        if self.file:
            self.file.close()


class NetworkSlicingGymDiscrete(gym.Env):
    def __init__(self):
        self.environment = NetworkSlicingEnv(seed=Config.RANDOM_SEED)

        self.observation_space = gym.spaces.Box(
            low=0.0, 
            high=1000.0,
            shape=(3,), 
            dtype=np.float32
        )

        # Action space: 21 discrete bandwidth allocation strategies
        self.action_space = gym.spaces.Discrete(21)
        
        # Predefined bandwidth allocations (VoLTE, Video, URLLC) in MHz
        # Total bandwidth = 20 MHz across all three slices
        self.actions = [
            # Balanced allocations (actions 0-3)
            [6.67, 6.67, 6.66],  # 0: Equal split
            [7, 7, 6],           # 1: Slight VoLTE+Video preference
            [7, 6, 7],           # 2: Slight VoLTE+URLLC preference
            [6, 7, 7],           # 3: Slight Video+URLLC preference
            
            # VoLTE-prioritized allocations (actions 4-6)
            [10, 5, 5],          # 4: Moderate VoLTE priority
            [12, 4, 4],          # 5: High VoLTE priority
            [14, 3, 3],          # 6: Very high VoLTE priority
            
            # Video-prioritized allocations (actions 7-9)
            [5, 10, 5],          # 7: Moderate Video priority
            [4, 12, 4],          # 8: High Video priority
            [3, 14, 3],          # 9: Very high Video priority
            
            # URLLC-prioritized allocations (actions 10-12)
            [5, 5, 10],          # 10: Moderate URLLC priority
            [4, 4, 12],          # 11: High URLLC priority
            [3, 3, 14],          # 12: Very high URLLC priority
            
            # Mixed strategies (actions 13-20)
            [8, 8, 4],           # 13: VoLTE+Video emphasis
            [8, 4, 8],           # 14: VoLTE+URLLC emphasis
            [4, 8, 8],           # 15: Video+URLLC emphasis
            [10, 7, 3],          # 16: VoLTE primary, Video secondary
            [10, 3, 7],          # 17: VoLTE primary, URLLC secondary
            [7, 10, 3],          # 18: Video primary, VoLTE secondary
            [7, 3, 10],          # 19: Video primary, URLLC secondary
            [3, 10, 7],          # 20: Video primary, URLLC secondary
        ]

    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
        
        state = self.environment.reset()
        obs = np.array(state, dtype=np.float32)
        return obs, {}

    def step(self, action):
        # Map discrete action to bandwidth allocation
        action_mhz = self.actions[action]
        
        # Execute step in underlying environment
        obs, rew, term, trunc, info = self.environment.step(action_mhz)
        obs = np.array(obs, dtype=np.float32)
        
        # Periodic logging for monitoring training progress
        if self.environment.current_step % 100 == 0:
            ssr = info['ssr']
            print(f"Step {self.environment.current_step}: "
                  f"Action={action}({action_mhz}), "
                  f"SSR=[{ssr['VoLTE']:.2f}, {ssr['Video']:.2f}, {ssr['URLLC']:.2f}], "
                  f"Reward={rew:.1f}")
        
        return obs, rew, term, trunc, info


class NetworkSlicingGymContinuous(gym.Env):
    def __init__(self):
        """Initialize continuous action space environment"""
        self.environment = NetworkSlicingEnv(seed=Config.RANDOM_SEED)
        self.last_obs = None

        self.observation_space = gym.spaces.Box(
            low=0.0, 
            high=100000.0,
            shape=(3,), 
            dtype=np.float32
        )

        # Continuous action space: bandwidth ratio for each slice
        self.action_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )

    @staticmethod
    def _normalize_action_ratio(action):
        action = np.array(action, dtype=np.float32)
        action = np.maximum(action, 0.0)

        action_sum = action.sum()
        if action_sum > 0:
            return action / action_sum

        return np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Reset environment"""
        if seed is not None:
            super().reset(seed=seed)
        
        state = self.environment.reset()
        obs = np.array(state, dtype=np.float32)
        self.last_obs = obs.copy()
        return obs, {}

    def step(self, action):
        current_obs = None if self.last_obs is None else self.last_obs.copy()

        # Normalize action to ensure sum equals 1.0
        action = self._normalize_action_ratio(action)

        # Scale to 20 MHz total bandwidth
        action_mhz = action * 20.0
        
        # Execute step
        obs, rew, term, trunc, info = self.environment.step(action_mhz)
        obs = np.array(obs, dtype=np.float32)
        self.last_obs = obs.copy()

        info = dict(info)
        info["current_state"] = current_obs
        info["next_state"] = obs.copy()
        info["applied_action_ratio"] = action.copy()
        info["applied_action_mhz"] = action_mhz.copy()
        
        # Periodic logging
        if self.environment.current_step % 100 == 0:
            print(f"Step {self.environment.current_step}: "
                  f"Action={action_mhz}, Utility={info['utility']:.3f}, "
                  f"Reward={rew:.3f}")
        
        return obs, rew, term, trunc, info


class NetworkSlicingGymStarsBars(gym.Env):
    def __init__(self):
        self.environment = NetworkSlicingEnv(seed=Config.RANDOM_SEED)
        self.last_obs = None

        self.observation_space = gym.spaces.Box(
            low=0.0, 
            high=1000.0,
            shape=(3,), 
            dtype=np.float32
        )

        self.action_space = gym.spaces.Discrete(n=4851)
        self.action_space_lookup = ActionSpace()

    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)

        state = self.environment.reset()
        obs = np.array(state, dtype=np.float32)
        self.last_obs = obs.copy()
        return obs, {}

    def step(self, action):
        current_obs = None if self.last_obs is None else self.last_obs.copy()

        # Lookup bandwidth allocation from action index
        action_units = self.action_space_lookup.lookup_action(action)

        # Convert to MHz (each unit = 0.2 MHz)
        action_mhz = np.array([0.2 * a for a in action_units], dtype=np.float32)

        # Execute step
        obs, rew, term, trunc, info = self.environment.step(action_mhz)
        obs = np.array(obs, dtype=np.float32)
        self.last_obs = obs.copy()

        info = dict(info)
        info["current_state"] = current_obs
        info["next_state"] = obs.copy()
        info["action_index"] = int(action)
        info["applied_action_units"] = np.array(action_units, dtype=np.int32)
        info["applied_action_mhz"] = action_mhz.copy()
        info["applied_action_ratio"] = action_mhz / float(action_mhz.sum())

        return obs, rew, term, trunc, info


# Backward-compatible alias for older code that imports NetworkSlicingGym.
NetworkSlicingGym = NetworkSlicingGymStarsBars
