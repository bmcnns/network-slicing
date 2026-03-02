import numpy as np
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from config import Config


_DEFAULTS = {
    'VIDEO_LAMBDA':       Config.VIDEO_LAMBDA,
    'VIDEO_MAX_INTERVAL': Config.VIDEO_MAX_INTERVAL,
    'URLLC_LAMBDA':       Config.URLLC_LAMBDA,
    'VOLTE_MIN_INTERVAL': Config.VOLTE_MIN_INTERVAL,
    'VOLTE_MAX_INTERVAL': Config.VOLTE_MAX_INTERVAL,
    'SLICE_USER_COUNTS':  Config.SLICE_USER_COUNTS,
}


def apply_scenario(params: dict):
    """Override Config class attributes with scenario-specific values."""
    for key, val in _DEFAULTS.items():
        setattr(Config, key, val)
    for key, val in params.items():
        setattr(Config, key, val)


SCENARIOS = {
    'default': {
        'VIDEO_LAMBDA':       2.0,
        'VIDEO_MAX_INTERVAL': 40,
        'URLLC_LAMBDA':       4.0,
        'VOLTE_MIN_INTERVAL': 1,
        'VOLTE_MAX_INTERVAL': 3,
        'SLICE_USER_COUNTS':  [100, 50, 30],
        'description': 'Default — reference baseline',
    },

    'heavy_video': {
        'VIDEO_LAMBDA':       4.0,
        'VIDEO_MAX_INTERVAL': 40,
        'URLLC_LAMBDA':       4.0,
        'VOLTE_MIN_INTERVAL': 1,
        'VOLTE_MAX_INTERVAL': 3,
        'SLICE_USER_COUNTS':  [100, 50, 30],
        'description': 'Heavy Video load (lambda x2)',
    },

    'heavy_urllc': {
        'VIDEO_LAMBDA':       2.0,
        'VIDEO_MAX_INTERVAL': 40,
        'URLLC_LAMBDA':       8.0,
        'VOLTE_MIN_INTERVAL': 1,
        'VOLTE_MAX_INTERVAL': 3,
        'SLICE_USER_COUNTS':  [100, 50, 30],
        'description': 'Heavy URLLC load (lambda x2)',
    },

    'urllc_dominant': {
        'VIDEO_LAMBDA':       2.0,
        'VIDEO_MAX_INTERVAL': 40,
        'URLLC_LAMBDA':       4.0,
        'VOLTE_MIN_INTERVAL': 1,
        'VOLTE_MAX_INTERVAL': 3,
        'SLICE_USER_COUNTS':  [50, 30, 100],
        'description': 'URLLC-dominant user distribution [50,30,100]',
    },
}

# ══════════════════════════════════════════════════════════════════════
# CONTROLLER DEFINITIONS
# ══════════════════════════════════════════════════════════════════════
TOTAL_BW_MHZ = 20.0
N_SLICES     = 3
MIN_BW_MHZ   = Config.MIN_BANDWIDTH_PER_SLICE / 1e6   # 1 MHz


def random_controller(obs, env):
    """
    Random: Dirichlet-sampled allocation with 1 MHz floor per slice.
    Lower bound — cannot learn from experience (paper Sec VII-B).
    """
    remaining = TOTAL_BW_MHZ - N_SLICES * MIN_BW_MHZ
    props = np.random.dirichlet(np.ones(N_SLICES))
    return (MIN_BW_MHZ + props * remaining).tolist()


def equal_controller(obs, env):
    """
    Equal split: fixed 1/3 per slice regardless of traffic or priority.
    Naive reference — ignores all state information.
    """
    bw = TOTAL_BW_MHZ / N_SLICES
    return [bw, bw, bw]


def proportional_controller(obs, env):
    """
    Proportional-to-traffic: allocate BW proportional to arrived packets
    per slice (obs = packet counts from previous step).
    Responds to load but ignores slice priority weights (BETA).
    Falls back to equal split on first step when obs is all zeros.
    """
    obs_arr = np.array(obs, dtype=float)
    if obs_arr.sum() == 0:
        return equal_controller(obs, env)
    remaining = TOTAL_BW_MHZ - N_SLICES * MIN_BW_MHZ
    props = obs_arr / obs_arr.sum()
    return (MIN_BW_MHZ + props * remaining).tolist()


def weighted_proportional_controller(obs, env):
    """
    Beta-weighted proportional: weight each slice's packet count by
    its SSR importance (Config.BETA=[1,1,5]) before allocating.
    URLLC gets 5x amplification per unit traffic — directly encodes
    the paper's priority structure. Should outperform plain Proportional
    in heavy_urllc and urllc_dominant scenarios.
    """
    obs_arr  = np.array(obs, dtype=float)
    betas    = np.array(Config.BETA, dtype=float)
    weighted = obs_arr * betas
    if weighted.sum() == 0:
        return equal_controller(obs, env)
    remaining = TOTAL_BW_MHZ - N_SLICES * MIN_BW_MHZ
    props = weighted / weighted.sum()
    return (MIN_BW_MHZ + props * remaining).tolist()


def urllc_priority_controller(obs, env):
    """
    Static URLLC-priority: always give URLLC 60% BW, rest split equally.
    Upper-bound reference for URLLC SSR; shows the cost to other slices.
    """
    urllc_bw = TOTAL_BW_MHZ * 0.6
    rest      = (TOTAL_BW_MHZ - urllc_bw) / 2
    return [rest, rest, urllc_bw]


CONTROLLERS = {
    'Random':        random_controller,
    'Equal':         equal_controller,
    'Proportional':  proportional_controller,
    # 'WeightedProp':  weighted_proportional_controller,
    # 'URLLCPriority': urllc_priority_controller,
}

# ══════════════════════════════════════════════════════════════════════
# (2)  DATA OUTPUT 
# ══════════════════════════════════════════════════════════════════════
SLICE_NAMES = ['VoLTE', 'Video', 'URLLC']
DAT_HEADER  = (
    "# step se utility reward "
    "ssr_VoLTE ssr_Video ssr_URLLC "
    "bw_VoLTE bw_Video bw_URLLC "
    "arr_VoLTE arr_Video arr_URLLC "
    "suc_VoLTE suc_Video suc_URLLC "
    "drp_VoLTE drp_Video drp_URLLC\n"
)


def write_dat_row(f, step, info, reward):
    ssr = info['ssr']
    bw  = info['bandwidth_allocation']
    arr = info['total_arrived_packets']
    suc = info['total_successful_packets']
    drp = info['dropped_packets']
    se      = info['spectrum_efficiency']
    utility = info['utility']
    row = (
        f"{step} {se:.6f} {utility:.6f} {reward:.6f} "
        f"{ssr['VoLTE']:.4f} {ssr['Video']:.4f} {ssr['URLLC']:.4f} "
        f"{bw['VoLTE']/1e6:.3f} {bw['Video']/1e6:.3f} {bw['URLLC']/1e6:.3f} "
        f"{arr['VoLTE']} {arr['Video']} {arr['URLLC']} "
        f"{suc['VoLTE']} {suc['Video']} {suc['URLLC']} "
        f"{drp['VoLTE']} {drp['Video']} {drp['URLLC']}\n"
    )
    f.write(row)


# ══════════════════════════════════════════════════════════════════════
# EVALUATION LOOP
# ══════════════════════════════════════════════════════════════════════

def run_evaluation(controller_fn, dat_path, n_steps=300, seed=2022):
    """Run controller for n_steps, write one .dat row per step."""
    from environment import NetworkSlicingEnv   # import after Config patched
    env = NetworkSlicingEnv(seed=seed)
    obs = env.reset()

    utilities = []
    ssrs = {s: [] for s in SLICE_NAMES}

    os.makedirs(os.path.dirname(dat_path) or '.', exist_ok=True)
    with open(dat_path, 'w') as f:
        f.write(DAT_HEADER)
        for step in range(1, n_steps + 1):
            action = controller_fn(obs, env)
            obs, reward, done, truncated, info = env.step(action)
            write_dat_row(f, step, info, reward)
            utilities.append(info['utility'])
            for s in SLICE_NAMES:
                ssrs[s].append(info['ssr'][s])
            if done or truncated:
                obs = env.reset()

    return {
        'utility': np.mean(utilities),
        'ssr':     {s: np.mean(ssrs[s]) for s in SLICE_NAMES},
    }


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    N_STEPS = 300
    SEED    = 2022
    OUT_DIR = 'eval_data'

    print(f"Output directory : {OUT_DIR}/")
    print(f"Steps per run    : {N_STEPS}\n")
    print(f"{'Scenario':<18} {'Controller':<16} "
          f"{'Utility':>8} {'SSR_V':>7} {'SSR_Vi':>7} {'SSR_U':>7}  File")
    print("-" * 85)

    for sc_name, scenario in SCENARIOS.items():
        apply_scenario(scenario)    # apply this scenario's overrides

        for ctrl_name, ctrl_fn in CONTROLLERS.items():
            dat_path = os.path.join(OUT_DIR, f"{sc_name}_{ctrl_name}.dat")
            summary  = run_evaluation(ctrl_fn, dat_path,
                                      n_steps=N_STEPS, seed=SEED)
            print(
                f"{sc_name:<18} {ctrl_name:<16} "
                f"{summary['utility']:>8.3f} "
                f"{summary['ssr']['VoLTE']:>7.3f} "
                f"{summary['ssr']['Video']:>7.3f} "
                f"{summary['ssr']['URLLC']:>7.3f}  "
                f"{dat_path}"
            )

    print("\nDone.")