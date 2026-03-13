import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════
FILE_PATH   = "training_ppo.dat"
# FILE_PATH   = "eval_data/default_Equal.dat"
# FILE_PATH   = "eval_data/default_Proportional.dat"
# FILE_PATH   = "eval_data/default_Random.dat"

SLICE_NAMES = ['VoLTE', 'Video', 'URLLC']
SMOOTH_WIN  = 300

SLICE_COLOR  = {'VoLTE': '#1d6fa4', 'Video': '#2a8c4a', 'URLLC': '#c95f1a'}
METRIC_COLOR = {'reward': '#7b2d8b', 'se': '#b85c00', 'utility': '#006d6d'}
DROP_COLOR   = '#c0392b'

# ══════════════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════════════
data = {
    'step': [], 'se': [], 'utility': [], 'reward': [],
    'ssr':     {s: [] for s in SLICE_NAMES},
    'bw':      {s: [] for s in SLICE_NAMES},
    'arrived': {s: [] for s in SLICE_NAMES},
    'success': {s: [] for s in SLICE_NAMES},
    'dropped': {s: [] for s in SLICE_NAMES},
}

with open(FILE_PATH, 'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.strip().split()
        if len(parts) != 19:
            continue
        data['step'].append(int(parts[0]))
        data['se'].append(float(parts[1]))
        data['utility'].append(float(parts[2]))
        data['reward'].append(float(parts[3]))
        for i, s in enumerate(SLICE_NAMES):
            data['ssr'][s].append(float(parts[4 + i]))
            data['bw'][s].append(float(parts[7 + i]))
            data['arrived'][s].append(float(parts[10 + i]))
            data['success'][s].append(float(parts[13 + i]))
            data['dropped'][s].append(float(parts[16 + i]))

steps = np.array(data['step'])

# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════
def smooth(y, w=SMOOTH_WIN):
    y = np.array(y)
    if len(y) < w:
        return steps, y
    return steps[w-1:], np.convolve(y, np.ones(w)/w, mode='valid')

STYLE = {
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 10,
    'axes.facecolor': '#f8f8f6', 'figure.facecolor': 'white',
    'axes.edgecolor': '#aaaaaa', 'axes.labelcolor': '#333333',
    'xtick.color': '#555555', 'ytick.color': '#555555',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': '#dddddd',
    'grid.linewidth': 0.7, 'grid.linestyle': '--',
    'axes.titlepad': 10, 'xtick.major.size': 3, 'ytick.major.size': 3,
}
plt.rcParams.update(STYLE)

fmt_k     = FuncFormatter(lambda x, _: f'{int(x/1000)}k' if x >= 1000 else f'{int(x)}')
fmt_comma = FuncFormatter(lambda x, _: f'{int(x):,}' if abs(x) >= 1000 else f'{x:.3g}')
SUBTITLE  = f'Shaded: raw data    |    Line: rolling mean (window = {SMOOTH_WIN} steps)'

def add_titles(fig, title, subtitle=SUBTITLE):
    fig.text(0.5, 0.978, title, ha='center', va='top', fontsize=15,
             fontweight='bold', color='#111111')
    fig.text(0.5, 0.948, subtitle, ha='center', va='top', fontsize=9,
             color='#888888', style='italic')

def plot_series(ax, y_raw, color, alpha_fill=0.15, alpha_line=0.22):
    sx, sy = smooth(y_raw)
    ax.fill_between(steps, y_raw, alpha=alpha_fill, color=color, linewidth=0)
    ax.plot(steps, y_raw, color=color, alpha=alpha_line, linewidth=0.5)
    ax.plot(sx, sy, color=color, linewidth=2.0, solid_capstyle='round')

def finish_ax(ax, is_bottom, ylabel=None, ylim=None, yfmt=None):
    ax.set_xlim(steps[0], steps[-1])
    ax.xaxis.set_major_formatter(fmt_k)
    if not is_bottom:
        ax.tick_params(labelbottom=False)
    else:
        ax.set_xlabel('Step', fontsize=9, color='#555555')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color='#333333')
    if ylim is not None:
        ax.set_ylim(*ylim)
    if yfmt:
        ax.yaxis.set_major_formatter(yfmt)

# ══════════════════════════════════════════════════════════════════════
# Figure 1: Utility & Reward
# ══════════════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(2, 1, figsize=(11, 6))
fig1.subplots_adjust(top=0.90, bottom=0.08, left=0.10, right=0.97, hspace=0.15)
add_titles(fig1, 'Utility & Reward')

# --- Utility (top) ---
plot_series(ax1[0], data['utility'], METRIC_COLOR['utility'])
finish_ax(ax1[0], False,
          ylabel='Utility',
          ylim=(0, None),
          yfmt=FuncFormatter(lambda x, _: f'{x:.2f}'))

# --- Reward (bottom) ---
plot_series(ax1[1], data['reward'], METRIC_COLOR['reward'])
finish_ax(ax1[1], True,
          ylabel='Reward',
          yfmt=fmt_comma)

fig1.savefig('figure/plot_utility_reward.png',
             dpi=180, bbox_inches='tight', facecolor='white')
print("Saved: plot_utility_reward.png")

# ══════════════════════════════════════════════════════════════════════
# Figure 2: Per-Slice Packet Statistics
# ══════════════════════════════════════════════════════════════════════
ROW_LABELS = ['Arrived', 'Success', 'Dropped']
ROW_KEYS   = ['arrived', 'success', 'dropped']

fig2 = plt.figure(figsize=(15, 10))
fig2.subplots_adjust(top=0.90, bottom=0.08, left=0.07, right=0.97,
                     hspace=0.20, wspace=0.28)
add_titles(fig2, 'Per-Slice Packet Statistics')

for row in range(3):
    for col, s in enumerate(SLICE_NAMES):
        ax = fig2.add_subplot(3, 3, row*3+col+1)
        color = DROP_COLOR if row == 2 else SLICE_COLOR[s]
        plot_series(ax, data[ROW_KEYS[row]][s], color)
        if row == 0:
            ax.set_title(s, color=SLICE_COLOR[s], fontsize=12,
                         fontweight='bold', pad=10)
        if col == 0:
            ax.set_ylabel(ROW_LABELS[row], fontsize=10, color='#333333')
        if row == 2:
            ax.spines['left'].set_color(DROP_COLOR)
            ax.spines['left'].set_linewidth(1.4)
        ax.yaxis.set_major_formatter(fmt_comma)
        finish_ax(ax, row == 2)

fig2.savefig('figure/plot_packets.png', dpi=180, bbox_inches='tight', facecolor='white')
print("Saved: plot_packets.png")

# ══════════════════════════════════════════════════════════════════════
# Figure 3: SSR & SE
# ══════════════════════════════════════════════════════════════════════
fig3, ax3 = plt.subplots(2, 1, figsize=(11, 7))
fig3.subplots_adjust(top=0.90, bottom=0.08, left=0.10, right=0.97, hspace=0.15)
add_titles(fig3, 'SSR & Spectrum Efficiency')

# --- SSR (top) ---
for s in SLICE_NAMES:
    c = SLICE_COLOR[s]
    sx, sy = smooth(data['ssr'][s])
    ax3[0].plot(steps, data['ssr'][s], color=c, alpha=0.15, linewidth=0.5)
    ax3[0].plot(sx, sy, color=c, linewidth=2.0,
                solid_capstyle='round', label=s)

finish_ax(ax3[0], False,
          ylabel='SSR',
          ylim=(0, 1.08),
          yfmt=FuncFormatter(lambda x, _: f'{x:.1f}'))

ax3[0].legend(framealpha=0.6, edgecolor='#cccccc',
              fontsize=9, loc='lower right')

# --- SE (bottom) ---
plot_series(ax3[1], data['se'], METRIC_COLOR['se'])
finish_ax(ax3[1], True,
          ylabel='Spectrum Efficiency',
          ylim=(0, None),
          yfmt=FuncFormatter(lambda x, _: f'{x:.2f}'))

fig3.savefig('figure/plot_ssr_se.png',
             dpi=180, bbox_inches='tight', facecolor='white')
print("Saved: plot_ssr_se.png")


# ══════════════════════════════════════════════════════════════════════
# Figure 4: Bandwidth Allocation
# ══════════════════════════════════════════════════════════════════════
fig4, ax4 = plt.subplots(figsize=(11, 4.5))
fig4.subplots_adjust(top=0.88, bottom=0.12, left=0.10, right=0.97)
add_titles(fig4, 'Bandwidth Allocation')

for s in SLICE_NAMES:
    c = SLICE_COLOR[s]
    sx, sy = smooth(data['bw'][s])
    ax4.plot(steps, data['bw'][s], color=c, alpha=0.15, linewidth=0.5)
    ax4.plot(sx, sy, color=c, linewidth=2.0,
             solid_capstyle='round', label=s)

finish_ax(ax4, True,
          ylabel='Bandwidth (MHz)',
          yfmt=FuncFormatter(lambda x, _: f'{x:.0f}'))

ax4.legend(framealpha=0.6, edgecolor='#cccccc',
           fontsize=9, loc='upper right')

fig4.savefig('figure/plot_bandwidth.png',
             dpi=180, bbox_inches='tight', facecolor='white')
print("Saved: plot_bandwidth.png")